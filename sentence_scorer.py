import torch
import numpy
import torch.nn.functional as F
import ujson as json
import random
import time
import torch.optim as optim
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup



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
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    training_datas =[]
    outputs = Parallel(n_jobs=12, verbose=10)(delayed(prepare_single_qapair)(qapair, tokenizer) for qapair in data)      
    training_datas = [e for e in outputs]
    print("Saving training_data")
    with open("Training_data", "w") as fh:
        json.dump(training_datas, fh)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


batch_size = 3
num_epochs= 4
MAX_batch_token_size = 5625


# preprocess_file("hotpot_train_v1.1.json")
# prepare_training_datas("preprocessed_hotpot_train_v1.1.json")
print("Loading datasets..")
train_dataset = json.load(open("Training_data.json", 'r'))
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False, 
)

model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
total_training_steps = len(train_dataset) // batch_size if len(train_dataset) % batch_size ==0 else (len(train_dataset) // batch_size)+1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_training_steps//10, num_training_steps= total_training_steps)

training_start_time = time.time()
for epoch in range(num_epochs):
    print("Shuffling dataset...")
    random.shuffle(train_dataset)
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')
    total_train_loss = 0
    model.train()
    step = 0
    total_train_loss = 0
    for single_batch in batch(train_dataset, batch_size):

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

        model.zero_grad()
        loss, logits = model(input_ids = b_inputs_ids, token_type_ids=b_segment_ids, attention_mask=b_attention_masks, labels=b_labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        step+=1
        if step % 100 ==0 and step != 0:
            elapsed_epoch_time = time.time()-training_start_time
            print("Batch [ {} / {} ] , loss = {} , elapsed = {}".format(step, total_training_steps, loss.item(), len(train_dataset)))

# Save the fine-tuned model
model.save_pretrained('./')



