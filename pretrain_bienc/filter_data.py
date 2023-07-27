import os
import numpy
from tqdm import tqdm
import json
import transformers as T
from replace_punc import punctuations
data = []
for each in os.listdir('./data4pretraining/wiki/'):
    with open('./data4pretraining/wiki/'+each, 'r') as f:
        data.extend(json.load(f))
# f = [list(each) for each in numpy.load(f'./data4pretraining/wiki/{each}', allow_pickle=True)]
tokenizer = T.AutoTokenizer.from_pretrained('hfl/chinese-lert-base', cache_dir='./cache')
# print(each, len(data))
data_filter = []
for d in tqdm(data, leave=True):
    d['text'] = d['text'].replace('\n', '').replace(' ', '')
    d['text'] = d['text'].lower()
    for each in punctuations:
        d["question"] = d["question"].replace(each[0], each[1])
        d["text"] = d["text"].replace(each[0], each[1])
    # qe = tokenizer.convert_tokens_to_ids(list(d["question"]))
    # te = tokenizer.convert_tokens_to_ids(list(d["text"]))
    # if 100 in te:
    #     a = tokenizer.decode(te)
    #     xxx = 10
    if d["question"] != "" and d["text"] != "":#and 100 not in qe and 100 not in te:
        data_filter.append(d)

# with open('./data4pretraining/wiki/'+each, 'r') as f:
#     data.extend(json.load(f))
with open(f'./data4pretraining/wiki_filter.json', 'w') as f:
    json.dump(data_filter, f)
# import numpy
# data = numpy.load(f'./data4pretraining/biencoder_data_Qnews.npy', allow_pickle=True)
# # numpy.save(f'./data4pretraining/biencoder_data_train', data[:2000000])
# # numpy.save(f'./data4pretraining/biencoder_data_dev', data[2000000:])
# print()