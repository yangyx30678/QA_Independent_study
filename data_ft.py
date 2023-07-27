from torch.utils.data import Dataset
import numpy

file_list = {
    'cmrc_train.npy':1,
    'xquad.npy':1,
    'zhongyi.npy':1,
    'NCPPolicies.npy':1
}

class data4finetuning(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []
        for each in file_list:
            if file_list[each] == 1:
                self.data.extend(numpy.load('./data4finetuning/'+each, allow_pickle=True))
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    d = data4finetuning()
    print(len(d))