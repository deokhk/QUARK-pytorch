import torch
import numpy
import torch.nn.functional as F
import torch.nn as nn
import ujson as json
import random
import time
import torch.optim as optim
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertForQuestionAnswering
from QA_module import preprocess_single_qapair, find_best_answer
class Quark(nn.Module):
    def __init__(self, sentence_scorer_without_answer, sentence_scorer_with_answer, QA_model, ss_tokenizer, qa_tokenizer):
        super(Quark, self).__init__()
        self.rnas = sentence_scorer_with_answer
        self.ras = sentence_scorer_without_answer
        self.QA_model = QA_model
        self.ss_tokenizer = ss_tokenizer
        self.qa_tokenizer = qa_tokenizer

    def forward(self, hotpot_qa_pairs):
        rnas = self.rnas
        ras = self.ras
        QA_model = self.QA_model
        ss_tokenizer = self.ss_tokenizer
        qa_tokenizer = self.qa_tokenizer
        


        inputs_ids =[]
        attention_masks =[]
        segment_ids =[]

        for qapair in hotpot_qa_pairs:
            rnas_sorted_sentences = preprocess_single_qapair(qapair, rnas, ss_tokenizer, "[MASK]")
            line_before_E_tokens =qa_tokenizer.convert_tokens_to_ids(qa_tokenizer.tokenize("[CLS] " + qapair['question'] + " [SEP] "))
            line_after_E_tokens = qa_tokenizer.convert_tokens_to_ids(qa_tokenizer.tokenize(" yes no noans [SEP]"))
            
            E_tokens=[]        
            is_new_paragraphs = [1 for _ in range(len(qapair['context']))]
            for single_sentence_info in rnas_sorted_sentences:
                E_tokens += qa_tokenizer.convert_tokens_to_ids(qa_tokenizer.tokenize(single_sentence_info['sentence']))
                para_idx = single_sentence_info['para']
                if is_new_paragraphs[para_idx] == 1:
                    para_first_sentence_and_title = single_sentence_info['first_sentence'] + " <t> " +single_sentence_info['title'] + " </t> "
                    E_tokens += qa_tokenizer.convert_tokens_to_ids(qa_tokenizer.tokenize(para_first_sentence_and_title))
                    is_new_paragraphs[para_idx] = 0
                
                if len(line_before_E_tokens + E_tokens + line_after_E_tokens) > 512:
                    # truncate E_tokens to fit in 512
                    E_tokens = E_tokens[:512-len(line_before_E_tokens + line_after_E_tokens)]
                    break

            qa_line = line_before_E_tokens + E_tokens + line_after_E_tokens
            

            segment_id = [0 for _ in range(len(line_before_E_tokens))] + [1 for _ in range(len(E_tokens + line_after_E_tokens))]
            attention_mask = [1 for _ in range(len(qa_line))]

            # pad the tokens 
            qa_line = qa_line + [0 for _ in range(512-len(qa_line))]
            segment_id = segment_id + [0 for _ in range(512-len(segment_id))]
            attention_mask = attention_mask + [0 for _ in range(512-len(attention_mask))]

            inputs_ids.append(qa_line)
            segment_ids.append(segment_id)
            attention_masks.append(attention_mask)
        
        b_inputs_ids = torch.Tensor(inputs_ids).cuda().long()
        b_segment_ids = torch.Tensor(segment_ids).cuda().long()
        b_attention_masks = torch.Tensor(attention_masks).cuda().long()
        

        with torch.no_grad():
            start_scores, end_scores = QA_model(input_ids = b_inputs_ids, attention_mask=b_attention_masks, token_type_ids=b_segment_ids)

        start_tokens_idxs, end_tokens_idxs = find_best_answer(start_scores, end_scores)

        qapairs_predicted_answers_with_id = {}
        predicted_answers = []

        for i in range(len(hotpot_qa_pairs)):
            token_line = inputs_ids[i]
            answer_part = token_line[start_tokens_idxs[i]:end_tokens_idxs[i]+1]
            answer = ' '.join(qa_tokenizer.convert_ids_to_tokens(answer_part))

            corrected_answer = ''
            for word in answer.split():
                #If it's a subword token
                if word[0:2] == '##':
                    corrected_answer += word[2:]
                else:
                    corrected_answer += ' ' + word
            predicted_answers.append(corrected_answer)
            qapair_id = hotpot_qa_pairs[i]['_id']
            qapairs_predicted_answers_with_id[qapair_id] = corrected_answer



        # Now, with predicted answers, we use ras model to score each sentence in D to find sentences supporting the chosen answer.
        qapairs_predicted_supporting_facts_with_id = {}
        for qapair_idx, qapair in enumerate(hotpot_qa_pairs):
            ras_sorted_sentences = preprocess_single_qapair(qapair, ras, ss_tokenizer, predicted_answers[qapair_idx])
            
            paragraphs_score_info = []
            
            for para in qapair['context']:
                para_info = {}
                para_info['title'] = para[0]
                para_info['sum_score'] = 0
                para_info['sentences'] = []
                paragraphs_score_info.append(para_info)
                        
            for single_sentence_info in ras_sorted_sentences:
                para_idx = single_sentence_info['para']
                sentence_idx_in_para = single_sentence_info['sentence_idx_in_para']
                supporting_fact_score = single_sentence_info['score']
                if supporting_fact_score >=0:
                    paragraphs_score_info[para_idx]['sum_score']+=supporting_fact_score
                    paragraphs_score_info[para_idx]['sentences'].append(sentence_idx_in_para)

            
            score_sorted_paragraphs = sorted(paragraphs_score_info, key=(lambda x: x['sum_score']), reverse=True)

            supporting_facts = []

            for sentence_idx in score_sorted_paragraphs[0]['sentences']:
                supporting_fact = [score_sorted_paragraphs[0]['title'], sentence_idx]
                supporting_facts.append(supporting_fact)

            for sentence_idx in score_sorted_paragraphs[1]['sentences']:
                supporting_fact = [score_sorted_paragraphs[1]['title'], sentence_idx]
                supporting_facts.append(supporting_fact)
                    

            qapair_id = hotpot_qa_pairs[qapair_idx]['_id']
            qapairs_predicted_supporting_facts_with_id[qapair_id] = supporting_facts


        return qapairs_predicted_answers_with_id, qapairs_predicted_supporting_facts_with_id 

    
        






                



        



        
        