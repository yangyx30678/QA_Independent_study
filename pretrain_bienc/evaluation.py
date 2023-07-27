import torch
from bienc_model import biencoder_model
device = 'cuda:2'
# model = biencoder_model().to(device)
# model.load_state_dict(torch.load('./saved_model/roberta_retriever_1_1400', map_location=device)['model'])
model = torch.load('./saved_model/roberta_retriever_1', map_location=device)
# model.eval()
print('---Model Complete!---')

from data_mt5gen import data_mt5gen_eval
dl = data_mt5gen_eval(split='dev')
print('---Data Complete!---')

def tensorize(q, c):
    query = torch.tensor(q).unsqueeze(0)
    cmax = max([len(each) for each in c])
    context = torch.zeros((len(c), cmax), dtype=torch.int64)
    for i in range(len(c)):
        for j in range(len(c[i])):
            context[i, j] = c[i][j]
    context = context.unsqueeze(0)
    return query, context

import copy
import numpy
ranks = []
for i in range(dl.len()[0]):
    scores = []
    # for j in range(dl.len()[1]):
        # if j % 1000 == 0:
        #     print(f'{j}/{dl.len()[1]}')
    q, c = dl.eval_qc(i)
    q, c = tensorize(q, c)
    for j in range(0, c.size(1), 4):
        score = model(q.to(device), copy.copy(c[:, j:j+4, :]).to(device))
        # scores.append(score.data)
        scores.extend([each.item() for each in list(score[0])])
    true = scores[i]
    # print()
    scores.sort(reverse=True)
    true_idx = scores.index(true)
    print(true_idx)
    ranks.append(true_idx)
numpy.save(f'./data4pretraining/eval_result', ranks)


# import transformers as T
# tokenizer = T.AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# for i in range(0, 5):
#     print(tokenizer.decode(dl.queries[i]))

# q = torch.tensor(dl.queries[6]).unsqueeze(0)
# print(f'\tquery: {tokenizer.decode(dl.queries[6])}')
# s = 0
# e = 20
# context = dl.contexts[s:e]
# maxlen = max([len(each) for each in context])
# c = torch.zeros((e-s, maxlen), dtype=torch.int64)
# for i in range(e-s):
#     print(f'\tcontext {i}: {tokenizer.decode(context[i])}')
#     for j in range(len(context[i])):
#         c[i, j] = context[i][j]
# c = c.unsqueeze(0)
# print(model(q, c))