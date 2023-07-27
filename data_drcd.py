# from datasets import load
from torch.utils.data import Dataset
import json
import numpy
from public_params import articles
from replace_punc import punctuations
file_dict = './data4finetuning/'

articles = []

class DRCD(Dataset):
    def __init__(self, split) -> None:
        super().__init__()
        self.name = 'DRCD'
        with open(f'{file_dict}drcd_{split}.json', 'r') as f:
            self.data = json.load(f)
        # self.data = numpy.load(f'{file_dict}cmrc_{split}.json', allow_pickle=True)

    def __getitem__(self, index):
        data_piece = self.data[index]
        for each in punctuations:
            data_piece['c'] = data_piece['c'].replace(each[0], each[1])
            data_piece['q'] = data_piece['q'].replace(each[0], each[1])
            # data_piece['c'] = data_piece['c'].replace(each[0], each[1])
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    import statistics as s
    # cmrc = CMRC(split='train')
    drcd = DRCD(split='test')
    # cmrc = CMRC(split='test')
    c = s.mean([len(each['c']) for each in drcd])
    q = s.mean([len(each['q']) for each in drcd])
    print(q, c)
    # for each in cmrc:
    #     print(f"  context:{each[0]}\n  query:{each[1]}\n  label:{each[0][each[2][0]:each[2][1]]}\n")

