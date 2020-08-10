import torch
import numpy
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import ujson as json
import random
from joblib import Parallel, delayed


data = json.load(open("hotpot_train_v1.1.json", 'r'))
data_num = len(data)
print("Total number of QA pair: ", len(data))
para_num_count = []

for a in range(10):
    para_num_count.append(0)
for i in range(data_num):
    para_num = len(data[i]['context'])
    para_num_count[para_num-1]+=1
    if para_num == 2:
        print("Detected qapair consists only of 2 pargraph")
        print(data[i])
        break
    if i% 1000 == 0:
        print("Done %d jobs, [ %d / %d]" % (i, i, data_num))

print(para_num_count)
count_sum  = 0
for i in para_num_count:
    count_sum+=i

if count_sum == data_num:
    print("OK all paragraphgs are within 10 paragraph")
else:
    print("SOMETHING WRONG..")
