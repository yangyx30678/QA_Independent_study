# from datasets import load
from torch.utils.data import Dataset
# import json
import os
# import numpy
# from public_params import articles
from replace_punc import punctuations
root = './datasets/zhwiki_odqa/'
import json
from multiprocessing import Pool
import random
from numpy.random import default_rng
rng = default_rng()
# from replace_punc import punctuations
# articles = []

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
            self.data = []
            for each in os.listdir('./data4pretraining/wiki/'):
                with open('./data4pretraining/wiki/'+each, 'r') as f:
                    self.data.extend(json.load(f))
            self.data = list(filter(lambda x: x["question"] != "", self.data))
            for i in range(len(self.data)):
                self.data[i]['text'] = self.data[i]['text'].replace('\n', '')
            # self.data = [each.replace() for each in self.data]
            print()
        # self.data = self.data[20000:]
        # print()
    def __getitem__(self, index):
        if self.new:
            return self.data[index]
        else:
            query = self.data[index]['question']
            context = [self.data[index]['text']]
            for idx in rng.choice(len(self.data), size=self.samples-1, replace=False):
                text = self.data[(index+idx+1) % len(self.data)]['text']
                # if len(text)>510: 
                    # start = random.randint(0, len(text)-510)
                #     context.append(text[start: start+510])
                # else:
                context.append(text)
            random.shuffle(context)
            label = context.index(self.data[index]['text'])
            return query, context, label

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
