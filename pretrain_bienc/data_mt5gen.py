from torch.utils.data import Dataset, DataLoader
import torch
import numpy
import random

samples = 4

class data_mt5gen(Dataset):
    def __init__(self, split='train') -> None:
        super().__init__()
        self.split = split
        self.data = []
        if split == 'train':
            self.data = numpy.load(f'./data4pretraining/biencoder_data_train.npy', allow_pickle=True)
        elif split == 'dev':
            self.data = numpy.load(f'./data4pretraining/biencoder_data_dev.npy', allow_pickle=True)
        self.contexts = [each[0] for each in self.data]
        self.queries = [each[1] for each in self.data]
    def __getitem__(self, index):
        query = self.queries[index]
        context = [self.contexts[index][1:-1]]
        idx_lst = list(range(len(self.contexts)))
        idx_lst.remove(index)
        # for i in range(samples-1):
        context.extend([self.contexts[idx][1:-1] for idx in random.sample(idx_lst, samples-1)])
        random.shuffle(context)
        label = context.index(self.contexts[index][1:-1])
        for i in range(samples):
            if len(context[i]) > 510:
                start = random.randint(0, len(context[i]) - 510)
                context[i] = [101]+context[i][start: start+510]+[102]
            else:
                context[i] = [101]+context[i]+[102]
        return [query, context, label]

    def __len__(self):
        return len(self.queries)


class data_mt5gen_eval():
    def __init__(self, split='dev') -> None:
        super().__init__()
        self.split = split
        self.data = []
        if split == 'train':
            self.data = numpy.load(f'./data4pretraining/biencoder_data_train.npy', allow_pickle=True)
        elif split == 'dev':
            self.data = numpy.load(f'./data4pretraining/biencoder_data_dev.npy', allow_pickle=True)
        self.contexts = [each[0] for each in self.data]
        self.queries = [each[1] for each in self.data]
    def eval_qc(self, qi):#, ci):
        query = self.queries[qi]
        context = []
        for ci in range(len(self.contexts)):
            c_raw = self.contexts[ci][1:-1]
            for j in range(10, len(c_raw), 510):
                context.append([101]+c_raw[j-10: j+500]+[102])
                break
            
        return query, context

    def len(self):
        return len(self.queries), len(self.contexts)