import torch
import numpy
import torch.nn.functional as F
import ujson as json
import random
import time
import torch.optim as optim
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from util import batch, flat_accuracy
from sentence_scorer_without_answer import preprocess_file, prepare_datas

sentence_scorer_wa_model = BertForSequenceClassification.from_pretrained("./model/rnas")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

print("Preprocess dev data")
preprocess_file("hotpot_dev_distractor_v1.json")
print("Prepare dev data")
prepare_datas("preprocessed_wa_hotpot_dev_distractor_v1.json", "Dev")

print("Loading dev datasets")
dev_dataset = json.load(open("Dev_data_wa.json"))

sentence_scorer_wa_model.cuda()
print("Now validating...")
sentence_scorer_wa_model.eval()
validation_epoch_start_time = time.time()

total_eval_accuracy = 0
total_eval_loss = 0
step = 0
batch_size = 3
MAX_batch_token_size = 5625

for single_batch in batch(dev_dataset, batch_size):

    # cap the batch size at 5625 tokens
    inputs_ids=[]
    attention_masks=[]
    segment_ids=[]
    labels=[]
    for question in single_batch:
        for para in question:
            for sentence in para[2]:
                inputs_ids.append(para[0])
                attention_masks.append(para[1])
                segment_ids.append(sentence['segment_id'])
                labels.append(sentence['label'])
    current_batch_token_size = 0
    for sentence_token in inputs_ids:
        current_batch_token_size+=len(sentence_token)
    
    while current_batch_token_size > MAX_batch_token_size:
        drop_sentence_idx = random.randint(0, len(inputs_ids)-1)

        current_batch_token_size -=len(inputs_ids[drop_sentence_idx])
        del inputs_ids[drop_sentence_idx]
        del attention_masks[drop_sentence_idx]
        del segment_ids[drop_sentence_idx]
        del labels[drop_sentence_idx]

    b_inputs_ids = torch.Tensor(inputs_ids).cuda().long()
    b_segment_ids = torch.Tensor(segment_ids).cuda().long()
    b_attention_masks = torch.Tensor(attention_masks).cuda().long()
    b_labels = torch.Tensor(labels).cuda().long()

    with torch.no_grad():
        loss, logits = sentence_scorer_wa_model(input_ids = b_inputs_ids, token_type_ids=b_segment_ids, attention_mask=b_attention_masks, labels=b_labels)        
    
    total_eval_loss += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    total_eval_accuracy += flat_accuracy(logits, label_ids)

    step+=1

avg_eval_loss = total_eval_loss / step
avg_eval_accuracy = total_eval_accuracy / step
Validation_time = time.time()-validation_epoch_start_time

print("Epoch 1 average validation loss : {}".format(avg_eval_loss))
print("Epoch 1 average validation accuracy : {}".format(avg_eval_accuracy))
