import torch
import numpy
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import ujson as json
import random
from joblib import Parallel, delayed


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) avaialbe." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

def process_article(qapair):
        single_data = {}
        single_data['question'] = qapair['question']
        single_data['answer'] = qapair['answer']
        paragraphs = qapair['context']
        spfacts = qapair['supporting_facts']
        spfacts_titles = []
        for spfact in spfacts:
            if spfact[0] not in spfacts_titles:
                spfacts_titles.append(spfact[0])

        sentences=[]
        numlist = list(range(len(paragraphs)))

        # process 2 paragraphs with supporting sentences
        for para in paragraphs:
            if para[0] in spfacts_titles:
                supporting_sentence_idx = []
                for spfact in spfacts:
                    if spfact[0] == para[0]:
                        supporting_sentence_idx.append(spfact[1])
                for sentence_idx in range(len(para[1])):
                    single_sentence = {}
                    if sentence_idx in supporting_sentence_idx:
                        single_sentence['label'] = 1
                    else:
                        single_sentence['label'] = 0
                    single_sentence['sentence'] = para[1][sentence_idx]
                    sentences.append(single_sentence)
                numlist.remove(paragraphs.index(para))
        """
        If possible, randomly sample 2 paragraphs without supporting sentences.
        Some article does not contain 10 paragraphs. Below is the number of examples with corresponding paragraph number.
        [0, 262, 156, 94, 88, 53, 77, 60, 48, 89609]
        """
        remaining_para_num = len(numlist)
        sampled_para_idx = random.sample(numlist,min(remaining_para_num, 2))

        for para_idx in sampled_para_idx:
            para = paragraphs[para_idx]
            for sentence in para[1]:
                    single_sentence = {}
                    single_sentence['label'] = 0
                    single_sentence['sentence']=sentence
                    sentences.append(single_sentence)

        single_data['sentences']=sentences
        return single_data

def preprocess_file(filename):
    data = json.load(open(filename, 'r'))

    training_datas = []

    total_data_nums = len(data)
    data_count = 0
    outputs = Parallel(n_jobs=12, verbose=10)(delayed(process_article)(article) for article in data)
    training_datas = [e for e in outputs]
    print("Saving preprocessed_{}".format(filename))
    with open("preprocessed"+filename, "w") as fh:
        json.dump(training_datas, fh)

preprocess_file("hotpot_train_v1.1.json")