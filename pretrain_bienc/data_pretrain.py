from data_news import news2016zh, news_batch
from data_cna import CNA, cna_batch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy
import random
from replace_punc import punctuations

class pretrain_dataset(Dataset):
    def __init__(self, new, start=0, end=113) -> None:
        super().__init__()
        self.new = new
        if new == True:
            news = news2016zh()
            cna = CNA() 
            self.data = news.data + cna.data
            random.shuffle(self.data)
        else:
            # self.data.extend(list(numpy.load('./data4pretraining2/part0.npy', allow_pickle=True)))
            root = './data4pretraining/news_cna/'
            self.data = []
            for i in range(start, end+1):
                print(i)
                self.data.extend(numpy.load(f'{root}part{i}.npy', allow_pickle=True))
            self.data = self.data[new:]
            # self.data1 = numpy.load(root+'pert_part0.npy', allow_pickle=True)
            # self.data2 = numpy.load(root+'pert_part1.npy', allow_pickle=True)
            # for f in os.listdir(root):
            #     print(f)
            #     self.data.extend(list(numpy.load('./data4pretraining/'+f, allow_pickle=True)))
            #     break
        
    def __getitem__(self, index):
        out = self.data[index]
        for each in punctuations:
            out.replace(each[0], each[1])
        return out

    def __len__(self):
        return len(self.data)

# def get_batch(sample):

#     return sample
if __name__ == '__main__':
    dl = pretrain_dataset(new=True)
    import tqdm
    for i in tqdm.tqdm(range(len(dl.data)//100000), leave=True):
        numpy.save(f'./data4pretraining/news_cna/part{i}', dl.data[i*100000:(i+1)*100000])
