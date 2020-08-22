import torch
import numpy
import torch.nn.functional as F
import ujson as json
import random
from joblib import Parallel, delayed
from sentence_scorer_without_answer import batch

x = list(range(0, 50))
for single_batch in batch(x, 8):
    print(single_batch)
