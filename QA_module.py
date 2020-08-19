import torch
import numpy
import torch.nn.functional as F
import ujson as json
import torch.optim as optim
import numpy as np
import time
import datetime
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, BertForQuestionAnswering, get_linear_schedule_with_warmup, AdamW, AutoModel

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def preprocess_single_qapair(single_hotpot_qapair):
    question = single_hotpot_qapair['question']
    answer = single_hotpot_qapair['answer']
    paragraphs = single_hotpot_qapair['context']
    line_before_sentence_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + question + " [SEP] "))
    line_after_sentence_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" [SEP] [MASK] [SEP]"))
    sentences = []
    
    for idx, para in enumerate(paragraphs):
        for sentence in para[1]:
            single_sentence_info ={}
            single_sentence_info['token'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
            single_sentence_info['score'] = 0
            single_sentence_info['para'] = idx
            sentences.append(single_sentence_info)
    
    pos = 0
    for single_batch in batch(sentences, 8):
        
        inputs_ids = []
        attention_masks = []
        segment_ids = []

        for single_sentence_info in single_batch:
            sentence_line_tokens = single_sentence_info['token']
            indexed_tokens = line_before_sentence_tokens + sentence_line_tokens + line_after_sentence_tokens
            indexed_tokens = indexed_tokens + [0 for _ in range(512-len(indexed_tokens))]
            attention_mask  = [1 for _ in range(len(indexed_tokens))] + [0 for _ in range(512-len(indexed_tokens))]
            token_type_id = [0 for _ in range(len(line_before_sentence_tokens))] + [1 for _ in range(len(sentence_line_tokens))] 
            token_type_id = token_type_id + [0 for _ in range(512-len(token_type_id))]
            inputs_ids.append(indexed_tokens)
            attention_masks.append(attention_mask)
            segment_ids.append(token_type_id)

        b_inputs_ids = torch.Tensor(inputs_ids).cuda().long()
        b_segment_ids = torch.Tensor(segment_ids).cuda().long()
        b_attention_masks = torch.Tensor(attention_masks).cuda().long()
        with torch.no_grad():
            logits =rnas_model(input_ids = b_inputs_ids, token_type_ids=b_segment_ids, attention_mask=b_attention_masks)
            logits_list = list(logits[0])
        for logit in logits_list:
            sentences[pos]['score'] = logit[1].item()
            pos=pos+1
    score_sorted_sentences = sorted(sentences, key=(lambda x : x['score']), reverse=True)
    return score_sorted_sentences


def prepare_file_for_rnas(original_hotpotqa_file, data_category):
    print("Loading {} ...".format(original_hotpotqa_file))
    data = json.load(open(original_hotpotqa_file, 'r'))
    print("Successfully loaded the data!")
    Num_total_datas = len(data)
    start_time = time.time()
    prepared_datas = []
    for myidx, qapair in enumerate(data):
        print(myidx)
        single_qa_line={}
        sorted_sentences = preprocess_single_qapair(qapair)
        line_before_E_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + qapair['question'] + " [SEP] "))
        line_after_E_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" yes no noans [SEP]"))
        line_len_without_E = len(line_before_E_tokens) + len(line_after_E_tokens)
        E_tokens=[]        
        for single_sentence_info in sorted_sentences:
            E_tokens = E_tokens + single_sentence_info['token']
            if len(E_tokens) + line_len_without_E > 512:
                E_tokens = E_tokens[:512-line_len_without_E]
                break
        qa_line = line_before_E_tokens + E_tokens + line_after_E_tokens

        tokenized_answer = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(qapair['answer']))
        answer_len = len(tokenized_answer)
        start_position = 0
        end_position = 0

        for idx in range(len(line_before_E_tokens), len(qa_line)-answer_len+1):
            candidate_span = qa_line[idx:idx+answer_len]
            if candidate_span == tokenized_answer:
                start_position = idx
                end_position = idx + answer_len - 1
                break
        
        # If we cannot find answer within the context, mark the answer as no answer.
        if start_position == 0 and end_position == 0:
            # index of noans token => [len(qa_line)-3 : len(qa_line)-2]
            start_position = len(qa_line)-3
            end_position = start_position+1

        segment_id = [0 for _ in range(len(line_before_E_tokens))] + [1 for _ in range(len(E_tokens + line_after_E_tokens))]
        attention_mask = [1 for _ in range(len(qa_line))]

        # pad the tokens 
        qa_line = qa_line + [0 for _ in range(512-len(qa_line))]
        segment_id = segment_id + [0 for _ in range(512-len(segment_id))]
        attention_mask = attention_mask + [0 for _ in range(512-len(attention_mask))]

        single_qa_line['line'] = qa_line
        single_qa_line['segment_id'] = segment_id
        single_qa_line['attention_mask'] = attention_mask
        single_qa_line['start_position'] = start_position
        single_qa_line['end_position'] = end_position
        prepared_datas.append(single_qa_line)

        if myidx%500 == 0:
            time_taken = time.time()-start_time
            print("Done [{}/{}], elapsed = {}".format(idx, Num_total_datas, format_time(time_taken)))
    print("Saving {}_data_for_rnas".format(data_category))
    with open(data_category+"_data_for_rnas.json", "w") as fh:
        json.dump(prepared_datas, fh)


print("Loading tokenizer..")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
batch_size = 16
num_epoch = 3
print("Loading model..")
rnas_model = BertForSequenceClassification.from_pretrained("./rnas_test/")
rnas_model.cuda()
rnas_model.eval()

print("Preparing training data..")
prepare_file_for_rnas("hotpot_train_v1.1.json", "Training")
print("Preparing dev data..")
prepare_file_for_rnas("hotpot_dev_distractor_v1.json", "Dev")

# optimizer = AdamW(lr=1e-5, weight_decay=0.01)
# total_training_steps = len(train_dataset) // batch_size if len(train_dataset) % batch_size ==0 else (len(train_dataset) // batch_size)+1
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= total_training_steps//10, num_training_steps=total_training_steps)
