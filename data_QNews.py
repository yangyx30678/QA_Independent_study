from torch.utils.data import Dataset
import numpy
import random
data_root = './data4pretraining'
import json
# random.seed(None)
import random

class QNews(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open(f'{data_root}/QNews_cn.json', 'r') as f:
            self.data = json.load(f)
        d = []
        random.seed(0)
        cy = [0.01, 0.02, 0.03]#[0.003, 0.005, 0.007]#
        for each in self.data:
            if len(each['c']) > 400:
                d.append(each)
            elif len(each['c']) < 200 and random.random() < cy[0]:
                d.append(each)
            elif len(each['c']) < 300 and random.random() < cy[1]:
                d.append(each)
            elif random.random() < cy[2]:
                d.append(each)
        random.seed(0)
        self.data = d
        self.contexts = [each['c'] for each in self.data]
        self.queries = [each['q'] for each in self.data]

    def __getitem__(self, index):
        q = self.queries[index]
        # maxlen = 509 - len(q)
        
        if random.random() < 0.3:
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


if __name__=='__main__':
    import statistics as s
    d = QNews()

    c = s.mean([len(each) for each in d.contexts])
    q = s.mean([len(each) for each in d.queries])
    dic = {'how':0, 'who':0,'when':0,'where':0, 'what':0,'which':0,'why':0,'other':0}
    for each in d.queries:
        if "如何" in each or "怎么" in each:
            dic['how'] += 1
        elif "什么" in each and "什么时候" not in each and "为什么" not in each:
            dic['what'] += 1
        elif "何时" in each or "什么时候" in each:
            dic['when'] += 1
        elif "在哪" in each or "何处" in each:
            dic['where'] += 1
        elif "谁" in each:
            dic['who'] += 1
        elif "为何" in each or "为什么" in each or "因何" in each:
            dic['why'] += 1
        elif "哪个" in each or "何者" in each:
            dic['which'] += 1
        else:
            each = each.replace("什麼", "甚麼")
            if "如何" in each or "怎麼" in each:
                dic['how'] += 1
            elif "甚麼" in each and "甚麼時候" not in each and "為甚麼" not in each:
                dic['what'] += 1
            elif "何時" in each or "甚麼時候" in each:
                dic['when'] += 1
            elif "在哪" in each or "何處" in each:
                dic['where'] += 1
            elif "誰" in each:
                dic['who'] += 1
            elif "為何" in each or "為甚麼" in each:
                dic['why'] += 1
            elif "哪個" in each or "何者" in each or "何種" in each:
                dic['which'] += 1
            else:
                dic['other'] += 1
    for each in dic:
        dic[each] /= len(d.queries)
    print(dic)
    print(q, c, len(d))
    # import transformers as T
    # from tqdm import tqdm
    # tokenizer = T.AutoTokenizer.from_pretrained('hfl/chinese-lert-base', cache_dir='./cache')
    # data = numpy.load(f'{data_root}/QNews.npy', allow_pickle=True)
    # d = []
    # for i in tqdm(range(len(data[:3600000])), leave=True):
    #     e = {'c':tokenizer.decode(data[i][0][1:-1]).replace(' ', ''), 'q':tokenizer.decode(data[i][1][1:-1]).replace(' ', '')}
    #     d.append(e)
    #     # if i == 3600000:
    #     #     break
    # import json
    # with open(f'{data_root}/QNews_cn.json', 'w') as f:
    #     json.dump(d, f)