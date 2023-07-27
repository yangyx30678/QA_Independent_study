from data_cmrc import CMRC
from bienc_model import biencoder_model
import transformers as T
import torch
from tqdm import tqdm
from copy import copy
import math
import jieba
device = 'cpu'
import json

def get_vectors(model, tokenizer):
    
    with open(f'./data4finetuning/cmrc_train_t.json', 'r') as f:
        dic = json.load(f)
    vecs = {'contexts':[], 'queries':[], 'pairs':[]}#{'contexts_vec':[], 'queries_vec':[]}
    # print(123)
    arts = []
    for i in tqdm(range(len(dic['contexts'])), leave=True):
        c = tokenizer.encode(dic['contexts'][i][:505], return_tensors='pt').to(device)
        if c.size(-1) <= 512: 
            c = copy(model.context_encoder(c)).squeeze()
            vecs['contexts'].append(c)
        else:
            vecs['contexts'].append(None)
    for i in tqdm(range(len(dic['queries'])), leave=True):
        q = tokenizer.encode(dic['queries'][i], return_tensors='pt').to(device)
        q = copy(model.question_encoder(q)).squeeze()
        vecs['queries'].append(q)
    
    vecs['pairs'] = copy(dic['pairs'])

    torch.save(vecs, './vecs')

def get_scores():
    vecs = torch.load('./vecs', map_location='cpu')
    
    counter = 1
    recall = [0, 0, 0]
    for i in tqdm(range(len(vecs['pairs']))):
        if vecs['contexts'][vecs['pairs'][i][0]] == None: continue
        ranks = 1
        s = []
        for j in range(len(vecs['contexts'])):
            if vecs['contexts'][j] == None: continue
            s.append(copy(torch.dot(vecs['queries'][vecs['pairs'][i][1]], vecs['contexts'][j])/math.sqrt(vecs['contexts'][j].size(-1))))
        correct = s[vecs['pairs'][i][0]]
        for j in range(len(s)):
            if s[j] > correct:
                ranks += 1
        if ranks <= 20: recall[2] += 1
        if ranks <= 5: recall[1] += 1
        if ranks == 1: recall[0] += 1 
    recall[0] /= len(vecs['pairs'])
    recall[1] /= len(vecs['pairs'])
    recall[2] /= len(vecs['pairs'])
    print(f"recall@1 {recall[0]}, recall@5 {recall[1]}, recall@20 {recall[2]}")
    torch.save(ranks, './recall')

if __name__ == "__main__":
    model = torch.load('./saved_model/roberta_wikiretriever_1_60000', map_location=device)
    tokenizer = T.AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir='./cache')
    get_vectors(model, tokenizer)
    # get_scores()
