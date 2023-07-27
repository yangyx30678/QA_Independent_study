from data_wiki import WIKI
from copy import copy
import transformers as T
import torch
from tqdm import tqdm
import time
from jieba import analyse
data = WIKI(new=False)
device = 'cuda:3'
def get_vectors(model, tokenizer):
    
    vecs = []#{'contexts_vec':[], 'queries_vec':[]}
    for each in tqdm(data.data, leave=True):
    # for i in tqdm(range(len(dic['contexts'])), leave=True):
        c = tokenizer.encode(each['text'], max_length=512, return_tensors='pt').to(device)
        c = copy(model.context_encoder(c)).squeeze()
        vecs.append(c)

    torch.save(vecs, './wiki_vecs')


def get_scores(query, model, tokenizer):

    vecs = torch.load('./wiki_vecs', map_location=device)
    kwds = analyse.textrank(query)[:2]
    q = tokenizer.encode(query, max_length=512, return_tensors='pt').to(device)
    q = copy(model.question_encoder(q)).squeeze()
    scores = []
    start = time.time()

    for i in tqdm(range(len(vecs)), leave=True):
        txt = data.data[i]['text']
        if kwds[0] in txt or kwds[1] in txt:
            s = torch.dot(q, vecs[i])
            if s > 11.0:
                scores.append((i, s))
    
    print(f"len: {len(scores)}")
    print(time.time()-start)
    start = time.time()

    # scores = [each for each in zip(range(len(scores)), scores)]
    sorted(scores, key=lambda x:x[1], reverse=True)

    print(time.time()-start)
    start = time.time()

    with open('results.txt', 'w') as f:
        for i in tqdm(range(100)):
            maxidx = scores[i][0]#max(range(len(scores)), key=scores.__getitem__)
            # scores[maxidx] -= 10000
            f.write(f'{i}: {data.data[maxidx]}\n')

    print(time.time()-start)
    # print(data.data[maxidx])


if __name__ == "__main__":
    model = torch.load('./saved_model/roberta_wikiretriever_1_60000', map_location=device)
    tokenizer = T.AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir='./cache')
    get_vectors(model, tokenizer)
    # query = "泰坦尼克號是哪國電影？"
    # get_scores(query, model, tokenizer)