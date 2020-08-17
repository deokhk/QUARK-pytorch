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
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering, get_linear_schedule_with_warmup
from sentence_scorer import sentence_scorer_model

class Quark(nn.Module):
    def __init__(self,sentence_scorer_with_answer, sentence_scorer_without_answer):
        super(Quark, self).__init__()
        self.sentence_scorer_with_answer = sentence_scorer_with_answer
        self.sentence_scorer_without_answer = sentence_scorer_without_answer

        
        