import torch
import numpy
import torch.nn.functional as F
import ujson as json
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, BertForQuestionAnswering, get_linear_schedule_with_warmup, AdamW, AutoModel
# from sentence_scorer_without_answer import batch

print("Loading tokenizer..")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
batch_size = 16
num_epoch = 3
print("Loading model..")
config = BertConfig.from_pretrained('./rnas_test/config.json')
rnas_model = BertForSequenceClassification.from_pretrained("./rnas_test/", config=config)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def prepare_single_training_data(original_hotpot_qapair):
    question = original_hotpot_qapair['question']
    answer = original_hotpot_qapair['answer']
    paragraphs = original_hotpot_qapair['context']
    line_before_E_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + question + " [SEP] "))
    line_before_sentence_tokens = line_before_E_tokens
    line_after_sentence_tokens = " [SEP] [MASK] [SEP]"
    sentences = []
    
   
    for para in paragraphs:
        for sentence in para[1]:
            single_sentence_info ={}
            single_sentence_info['raw_sentence'] = sentence
            single_sentence_info['token'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
            single_sentence_info['score'] = 0
            sentences.append(single_sentence_info)
    
    pos = 0
    for single_batch in batch(sentences, 8):
        
        inputs_ids = []
        attention_masks = []
        segment_ids = []

        for single_sentence_info in single_batch:
            sentence_line_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize( single_sentence_info['token']))
            indexed_tokens = line_before_sentence_tokens + sentence_line_tokens + line_before_sentence_tokens + [0 for _ in range(512-len(sentence_line_tokens))]
            attention_mask  = [1 for _ in range(len(indexed_tokens))] + [0 for _ in range(512-len(indexed_tokens))]
            token_type_id = [0 for _ in range(len(line_before_sentence_tokens))] + [1 for _ in range(len(sentence_line_tokens))] 
            token_type_id = token_type_id + [0 for _ in range(512-len(token_type_id))]
            inputs_ids.append(indexed_tokens)
            attention_masks.append(attention_mask)
            segment_ids.append(token_type_id)

        b_inputs_ids = torch.Tensor(inputs_ids).cuda().long()
        b_segment_ids = torch.Tensor(segment_ids).cuda().long()
        b_attention_masks = torch.Tensor(attention_masks).cuda().long()

        _, logits =rna_model(input_ids = b_inputs_ids, token_type_ids=b_segment_ids, attention_mask=b_attention_masks)

        for logit in logits:
            sentences[pos]['score'] = logit
            pos=pos+1
    return sentences
        

data = json.load(open("hotpot_train_v1.1.json", 'r'))
sentences = prepare_single_training_data(data[0])
print(sentences)
    

# optimizer = AdamW(lr=1e-5, weight_decay=0.01)
# total_training_steps = len(train_dataset) // batch_size if len(train_dataset) % batch_size ==0 else (len(train_dataset) // batch_size)+1
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= total_training_steps//10, num_training_steps=total_training_steps)
