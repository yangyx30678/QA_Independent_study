import torch
from torch.utils.data import Dataset, DataLoader
import json

train_filename = '../datasets/news2016zh/news2016zh_train.json'
valid_filename = '../datasets/news2016zh/news2016zh_valid.json'

class news2016zh(Dataset):
    def __init__(self) -> None:
        super().__init__()

        with open(train_filename, 'r', encoding='utf8') as file:
            jsons = file.readlines()
        self.data = []
        for each in jsons:
            j = json.loads(each)['content']
            # self.data.extend([each+'。' for each in j.split('。')])
            # if len(j)>253: 
            #     for i in range(20, len(j), 253):
            #         self.data.append(j[i-20:i+233])
            #         if i+233 >= len(j):
            #             break
            # elif len(j)<=100:
            #     continue
            # else:
            if 100 >= len(j):
                continue
            for i in range(0, len(j), 510):
                self.data.append(j[i:i+510])


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def news_batch(sample):
    return sample

if __name__ == '__main__':
    ds = news2016zh()
    import statistics as s
    l = s.mean([len(each) for each in ds])
    print(l)