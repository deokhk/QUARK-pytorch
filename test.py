import torch
import numpy
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import ujson as json
import random
from joblib import Parallel, delayed

def count_sentence_num_on_single_qa_pair(qapair):
    num_sentence_in_para_cnt = []
    for para in qapair:
        num_sentence_in_para_cnt.append(len(para[2]))
    return num_sentence_in_para_cnt



data = json.load(open("Training_data.json", 'r'))
outputs = Parallel(n_jobs=12, verbose=10)(delayed(count_sentence_num_on_single_qa_pair)(qapair) for qapair in data)      
num_sentence_in_paras = sum(outputs, [])
num_sentence_in_paras.sort()
max_sentence_num = num_sentence_in_paras[-1]
min_sentence_num = num_sentence_in_paras[0]

for i in range(min_sentence_num, max_sentence_num+1):
    print("Number of sentence :", i,"Number of paragraph with corresponding sentence number: ", num_sentence_in_paras.count(i))


