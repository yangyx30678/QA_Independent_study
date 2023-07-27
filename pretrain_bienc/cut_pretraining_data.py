from data_pretrain import pretrain_dataset
from data_wiki import WIKI
import random
import torch
import transformers as T
from torch.utils.data import Dataset, DataLoader
from question_generation import pipeline
from replace_punc import punctuations
import json
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = T.AutoTokenizer.from_pretrained("algolet/mt5-base-chinese-qg")
model = T.AutoModelForSeq2SeqLM.from_pretrained("algolet/mt5-base-chinese-qg")
d_id = 3
device = torch.device(f'cuda:{d_id}' if torch.cuda.is_available() else 'cpu')
# qg = pipeline("question-generation", model=model, tokenizer=tokenizer, device=f'cuda:{d_id}')

def qg(model, questions, tokenizer, device):
    inp = tokenizer.batch_encode_plus(questions, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    model = model.to(device)
    with torch.no_grad():
        questions = model.generate(
            input_ids = inp["input_ids"],
            attention_mask=inp["attention_mask"],
            max_length=128,
            no_repeat_ngram_size=4,
            num_beams=2
        )
    questions = tokenizer.batch_decode(questions.tolist())
    results = []
    for each in questions:
        qlist = each.split('<sep> ')
        if len(qlist) > 1 and qlist[0][6:] not in results:
            results.append(qlist[0][6:])
        else:
            results.append('')
    return results

s = 0
# pd = pretrain_dataset(new=0, start=0, end=30)
pd = WIKI(new=True)
print('--Data complete--')

roberta_tokenizer = T.AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# qg = pipeline('question-generation', device=device)
def getbatch(sample):
    d = [each for each in sample]
    # context = [tokenizer.encode(each) for each in d]
    question = qg(model, [each['text'] for each in d], tokenizer, f'cuda:{d_id}')
    # question = [tokenizer.encode(each) for each in question]
    return [{'title':d[i]['title'], 'text':d[i]['text'], 'question':question[i]} for i in range(len(d))]
dl = DataLoader(pd, batch_size=32, collate_fn=getbatch, shuffle=False)

import numpy


from tqdm import tqdm

# data = pd.data
part = []
counter = s

for i, d in enumerate(tqdm(dl, leave=True)):
# for i in range(len(data)):
    # context = tokenizer.encode(d)
    # question = qg(d, tokenizer=tokenizer)[0]
    # question = tokenizer.encode(question)
    part.extend(d)
    # if i % 1000 == 0:
    #     print(f'{i}/{len(data)}')
    if (i+1) % 10000 == 0:
        with open(f'./data4pretraining/wiki/part{counter}_{d_id}.json', 'w') as f:
            json.dump(part, f)
        part = []
        counter += 1
with open(f'./data4pretraining/wiki/part{counter}_{d_id}.json', 'w') as f:
    json.dump(part, f)