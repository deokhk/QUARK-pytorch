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
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertForQuestionAnswering, get_linear_schedule_with_warmup

class Quark(nn.Module):
    def __init__(self, sentence_scorer_with_answer, sentence_scorer_without_answer, QA_model, tokenizer):
        super(Quark, self).__init__()
        self.rnas = sentence_scorer_with_answer
        self.ras = sentence_scorer_without_answer
        self.QA_model = QA_model
        self.tokenizer = tokenizer
    
    def forward(self, qa_pairs):
        



        
        