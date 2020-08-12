import torch
import numpy
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import ujson as json
import random
from joblib import Parallel, delayed


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
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

        preprocessed_paragraphs=[]
        numlist = list(range(len(paragraphs)))

        # process 2 paragraphs with supporting sentences
        for para in paragraphs:
            if para[0] in spfacts_titles:
                _para = {}
                _para['title']=para[0]
                _para['sentences']=[]
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
                    _para['sentences'].append(single_sentence)
                preprocessed_paragraphs.append(_para)
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
            _para = {}
            _para['title']=para[0]
            _para['sentences']=[]
            for sentence in para[1]:
                    single_sentence = {}
                    single_sentence['label'] = 0
                    single_sentence['sentence']=sentence
                    _para['sentences'].append(single_sentence)
            preprocessed_paragraphs.append(_para)

        single_data['paragraphs']=preprocessed_paragraphs
        return single_data

def preprocess_file(filename):
    print('Preprocessing ', filename)
    data = json.load(open(filename, 'r'))

    preprocessed_datas = []

    outputs = Parallel(n_jobs=12, verbose=10)(delayed(process_article)(article) for article in data)
    preprocessed_datas = [e for e in outputs]
    print("Saving preprocessed_{}".format(filename))
    with open("preprocessed_"+filename, "w") as fh:
        json.dump(preprocessed_datas, fh)

def prepare_single_qapair(qapair, tokenizer):
    MAX_LEN = 512
    question = qapair['question']
    answer = qapair['answer']
    paragraphs = qapair['paragraphs']
    qapair_for_training = []
    for para in paragraphs:
        para_for_training= []
        indexed_tokens = []
        line_before_para_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + question + " [SEP] "))
        segment_before_para = [0 for _ in range(len(line_before_para_tokens))]

        line_after_para_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" [SEP] " + answer + " [SEP]"))
        segment_after_para = [0 for _ in range(len(line_after_para_tokens))]

        para_tokens = []

        for sentence in para['sentences']:
            sentence_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence['sentence']))
            para_tokens += sentence_token

        len_without_para_tokens = len(line_before_para_tokens) + len(line_after_para_tokens)
        # If a input has more than 512 tokens, we restrict the input to 512.

        allowed_para_length = len(para_tokens)
        if len(para_tokens)+len_without_para_tokens>512:
            para_tokens = para_tokens[0:512-len_without_para_tokens]
            allowed_para_length = len(para_tokens)

        indexed_tokens = line_before_para_tokens + para_tokens + line_after_para_tokens
        attention_mask = [1 for _ in range(len(indexed_tokens))] + [0 for _ in range(MAX_LEN-len(indexed_tokens))]

        # Pad the tokens to MAX_LEN
        indexed_tokens += [0 for _ in range(MAX_LEN-len(indexed_tokens))]

        para_for_training.append(indexed_tokens)
        para_for_training.append(attention_mask)
        para_for_training_sentence_list=[]
        pos = 0

        for sentence in para['sentences']:
            sentence_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence['sentence']))
            segment_para = [0 for _ in range(allowed_para_length)]
            if pos+len(sentence_token) > allowed_para_length:
                if pos < allowed_para_length:
                    segment_para[pos:] = [1 for _ in range(allowed_para_length - pos)]
            else:
                segment_para[pos:pos+len(sentence_token)] = [1 for _ in range(len(sentence_token))]
                pos = pos + len(sentence_token)
            sentence_for_training = {}
            sentence_for_training['label']=sentence['label']
            sentence_segment_id = segment_before_para + segment_para + segment_after_para
            
            # Pad the tokens to MAX_LEN
            sentence_segment_id += [0 for _ in range(MAX_LEN-len(sentence_segment_id))]
            sentence_for_training['segment_id']= sentence_segment_id
            para_for_training_sentence_list.append(sentence_for_training)
            

        para_for_training.append(para_for_training_sentence_list)
        qapair_for_training.append(para_for_training)
    return qapair_for_training
            
def prepare_training_datas(preprocessed_file):
    print('Loading preprocessed file...')
    data = json.load(open(preprocessed_file, 'r'))
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
    training_datas =[]
    outputs = Parallel(n_jobs=12, verbose=10)(delayed(prepare_single_qapair)(qapair, tokenizer) for qapair in data)      
    training_datas = [e for e in outputs]
    print("Saving training_data")
    with open("Training_data", "w") as fh:
        json.dump(training_datas, fh)


#preprocess_file("hotpot_train_v1.1.json")
prepare_training_datas("preprocessed_hotpot_train_v1.1.json")
