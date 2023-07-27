import json
from public_params import articles
# from datasets import load
from torch.utils.data import Dataset
import json
import numpy
file_dict = './data4finetuning/'


class Xquad(Dataset):
    def __init__(self, split) -> None:
        super().__init__()
        # self.data = numpy.load(f'{file_dict}{split}.npy', allow_pickle=True)
        self.data = []
        with open('../datasets/xquad/xquad.zh.json') as file:
            data_in_file = json.load(file)
        for paragraphs in data_in_file['data']:
            for paragraph in paragraphs['paragraphs']:
                # articles.append(paragraph['context'])
                for qa in paragraph['qas']:
                    self.data.append([paragraph['context'], qa['question'], [qa['answers'][0]['answer_start'], qa['answers'][0]['answer_start']+len(qa['answers'][0]['text'])]])
        print()
                
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # cmrc = CMRC(split='train')
    xquad = Xquad(split='dev')
    # cmrc = CMRC(split='test')
    # c = max([len(each[0]) for each in cmrc])
    # q = max([len(each[1]) for each in cmrc])

    for each in xquad:
        print(f"  context:{each[0]}\n  query:{each[1]}\n  label:{each[0][each[2][0]:each[2][1]]}\n")

