import os
import json
import urllib
from torch.utils.data import Dataset, DataLoader
import re

file_dict = './datasets/CNA/data/'
folder_list = os.listdir(file_dict)

class CNA(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []
        for folder in folder_list:
            files = os.listdir(file_dict + folder + '/')
            for file in files:
                with open(f'{file_dict}{folder}/{file}', 'r', encoding='utf8') as f:
                    d = f.read().replace('\n', '').replace(' ', '')
                    headlines = d.split('<HEADLINE>')
                    # print(headlines[-4:])
                    for article in headlines[1:]:
                        article = article.replace('</HEADLINE>', '<>').replace('<DATELINE>', '<>').replace('</DATELINE>', '<>').replace('<P>', '<>').replace('</P>', '<>').replace('<TEXT>', '<>').replace('</TEXT>', '<>')
                        patterns = article.split('<>')[:-1]
                        while '' in patterns: patterns.remove('')
                        for each in patterns[2:]:
                            if 100 <= len(each):
                                self.data.append(each)
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def cna_batch(sample):
    return sample

if __name__ == '__main__':
    ds = CNA()
    import statistics as s
    l = s.mean([len(each) for each in ds])
    print(l)
    # d0 = ds[0]
    # print(len(ds))
    # print()
