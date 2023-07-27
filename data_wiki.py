# from datasets import load
from torch.utils.data import Dataset
# import json
import os
# import numpy
# from public_params import articles
from replace_punc import punctuations
root = '../datasets/zhwiki_odqa/'
import json
from multiprocessing import Pool
import random
from copy import copy
# from numpy.random import default_rng
# rng = default_rng()
# from replace_punc import punctuations
# articles = []
import transformers as T
# test_tokenizer = T.BertTokenizer.from_pretrained('hfl/chinese-lert-base', cache_dir='./cache')
# print(test_tokenizer.encode("[UNK]"))
def read_file(file_dir):
    with open(file_dir, 'r') as pf:
        data = [json.loads(each) for each in pf.readlines()]
        data = [{'title':each['title'], 'text':each['text']} for each in data]
    return data

def cut_data(data):
    d = []
    for each in punctuations:
        data['text'] = data['text'].replace(each[0], each[1])
    for i in range(20, len(data['text']), 490):
        d.append({'title':data['title'], 'text':data['text'][i-20:i+490]})
    return d

class WIKI(Dataset):
    def __init__(self, new=False, samples=4) -> None:
        self.new = new
        self.samples = samples
        if new:
            data = []
            files = []
            for folder in os.listdir(root):
                for f in os.listdir(root+folder):
                    files.append(f"{root}{folder}/{f}")
            with Pool(24) as p:
                [data.extend(each) for each in p.map(read_file, files)]
            self.data = []
            with Pool(24) as p:
                [self.data.extend(each) for each in p.map(cut_data, data)]

        else:
            # self.data = []
            # for each in os.listdir('./data4pretraining/wiki/'):
            #     with open('./data4pretraining/wiki/'+each, 'r') as f:
            #         self.data.extend(json.load(f))
            with open('./QA_ft/data4pretraining/wiki_filter.json', 'r') as f:
                self.data = json.load(f)
            # for each in self.data:
            #     for p in punctuations:
            #         each['text'].replace(p[0], p[1])
            #         each['question'].replace(p[0], p[1])
            # self.data = list(filter(lambda x: x["question"] != "" and x["text"] != "", self.data))
            # d = copy(self.data)
            # self.data = []
            # for each in d:
            #     qe = test_tokenizer.encode(list(each["question"]))
            #     te = test_tokenizer.encode(list(each["text"]))
            #     if 100 not in qe and 100 not in te:
            #         self.data.append(each)
            # self.data = list(filter(lambda x: (100 not in test_tokenizer.encode(list(x["question"]))) and (100 not in test_tokenizer.encode(list(x["text"]))), self.data))
            # for i in range(len(self.data)):
            #     self.data[i]['text'] = self.data[i]['text'].replace('\n', '')
            self.queries = [each['question'] for each in self.data]
            self.contexts = [each['text'] for each in self.data]
            # self.data = [each.replace() for each in self.data]
            # print()
        # self.data = self.data[20000:]
        # print()
    def __getitem__(self, index):
        q = self.queries[index]
        # maxlen = 509 - len(q)
        
        if random.random() < 0.5:
            c = self.contexts[index]
            l = 1
        else:
            # cidx = list(range(len(self.contexts)))
            # cidx.remove(index)
            # c = self.contexts[random.choice(cidx)]
            c = self.contexts[(index+random.randint(1, len(self.contexts)-1))%len(self.contexts)]
            l = 0

        return [c, q, l]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # import torch
    # print(torch.__version__)
    import statistics as s
    wiki = WIKI()
    for each in wiki:
        print(wiki)
    # ave_len = s.mean([len(each) for each in wiki.data])
    # max_len = max([len(each) for each in wiki.data])
    # counter = 0
    # for each in wiki.data:
    #     if len(each['text']) >= 512:
    #         counter += 1
    # print(max_len)
    # print(counter / len(wiki.data))
    # print(ave_len)
