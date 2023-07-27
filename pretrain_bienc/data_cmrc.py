# from datasets import load
from torch.utils.data import Dataset
import json
import numpy
from public_params import articles
from replace_punc import punctuations
file_dict = '../data4finetuning/'

articles = []

class CMRC(Dataset):
    def __init__(self, split) -> None:
        super().__init__()
        with open(f'{file_dict}cmrc_{split}.json', 'r') as f:
            self.data = json.load(f)
        # self.data = numpy.load(f'{file_dict}cmrc_{split}.json', allow_pickle=True)
        d = []
        for each in self.data:
            if each['q'] != '' and each['c'] != '':
                for p in punctuations:
                    each['c'] = each['c'].replace(p[0], p[1])
                    each['q'] = each['q'].replace(p[0], p[1])
                d.append(each)
        self.data = d
        self.queries = [each['q'] for each in self.data]
        self.contexts = [each['c'] for each in self.data]

        

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
        return len(self.data)

if __name__ == '__main__':
    import statistics as s
    # cmrc = CMRC(split='train')
    cmrc = CMRC(split='test')
    # cmrc = CMRC(split='test')
    c = s.mean([len(each['c']) for each in cmrc])
    q = s.mean([len(each['q']) for each in cmrc])
    print(q, c)
    # for each in cmrc:
    #     print(f"  context:{each[0]}\n  query:{each[1]}\n  label:{each[0][each[2][0]:each[2][1]]}\n")

