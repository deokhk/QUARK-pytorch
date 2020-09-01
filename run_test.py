import torch
import numpy
import ujson as json
import numpy as np
import random
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, BertForQuestionAnswering, get_linear_schedule_with_warmup
from QUARK import Quark
from util import batch

def predict(data_source, prediction_file):
    print("Loading datasets..")
    dataset = json.load(open(data_source))

    print("Loading rnas, ras, QA model and tokenizer..")
    rnas_model = BertForSequenceClassification.from_pretrained("./model/rnas")
    ras_model = BertForSequenceClassification.from_pretrained("./model/ras/")
    QA_model = BertForQuestionAnswering.from_pretrained("./model/qa/")
    ss_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    qa_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    rnas_model.cuda()
    rnas_model.eval()
    ras_model.cuda()
    ras_model.eval()
    QA_model.cuda()
    QA_model.eval()

    print("Loading QUARK pipeline..")
    QP = Quark(rnas_model, ras_model, QA_model, ss_tokenizer, qa_tokenizer)

    print("All ready, start prediction")
    batch_size = 1
    predicted_answers = {}
    predicted_supporting_facts = {}

    total_batch_num = 0
    if len(dataset) %4 ==0:
        total_batch_num = len(dataset) // 4
    else:
        total_batch_num = len(dataset) //4 +1
    
    batch_count = 0
    for single_batch in batch(dataset, batch_size):
        batch_predicted_answers, batch_predicted_supporting_facts = QP.forward(single_batch)
        predicted_answers = {**predicted_answers, **batch_predicted_answers}
        predicted_supporting_facts = {**predicted_supporting_facts, **batch_predicted_supporting_facts}
        batch_count+=1
        
        if batch_count % 10 ==0:
            print("[Predicted / Total ] : {} / {}".format(batch_count, total_batch_num))
        break
    prediction_data = {}
    prediction_data['answer'] = predicted_answers
    prediction_data['sp'] = predicted_supporting_facts

    with open(prediction_file, 'w') as f:
        json.dump(prediction_data, f)

    print(prediction_data)
predict("hotpot_dev_distractor_v1.json", "dev_distractor_pred.json")


