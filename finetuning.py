from data_cmrc import CMRC
from data_drcd import DRCD
from data_cail import CAIL
from data_squad import Squad
from torch.utils.data import DataLoader
from model_MRC import ModelMRC
from data_webqa import WebQA
import torch
import transformers as T
import random
import numpy

record = []
def init_seed():
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)
def main(huggingface_name, model_name=None, train_set=None, dev_set=None, test_set=None, ep=2, device='cpu'):
    init_seed()
    # tokenizer = FullTokenizer('./QA_paper_1/pert_vocab.txt')
    tokenizer = T.BertTokenizer.from_pretrained(huggingface_name, cache_dir='./cache')
    # tokenizer = T.AutoTokenizer.from_pretrained('bert_base_chinese', cache_dir='./cache')
    # print(T.__version__)
    def get_batch(sample):
        # lengths = [len(each[0])+1 for each in sample]
        # qc = [[101]+each[0]+[102]+each[1]+[102] for each in sample]
        # da = [[list(each['q']), list(each['c'])] for each in sample]
        encoded = tokenizer.batch_encode_plus([[list(str(each['c'])), list(str(each['q']))] for each in sample], padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoded['input_ids']
        mask = encoded['attention_mask']
        token_type = encoded['token_type_ids']
        labels = [[each['a'][0]+1, each['a'][1]+1] for each in sample]
        # max_len = max([len(each) for each in qc])
        # input_ids = torch.zeros((len(sample), max_len), dtype=torch.int64)
        # mask = torch.zeros((len(sample), max_len), dtype=torch.int64)
        # token_type = torch.zeros((len(sample), max_len), dtype=torch.int64)
        # for i in range(len(sample)):
        #     for j in range(len(qc[i])):
        #         input_ids[i, j] = qc[i][j]
        #         mask[i, j] = 1
        #         if j >= lengths[i]:
        #             token_type[i, j] = 1
        labels = torch.tensor([each[0] for each in labels]), torch.tensor([each[1] for each in labels])

        return input_ids, mask, labels, token_type

    device = device# torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # model = ModelMRC('hfl/chinese-roberta-wwm-ext')
    # model = T.BertModel.from_pretrained('hfl/chinese-lert-base', cache_dir='./cache')
    if model_name != '':
        model = torch.load(f'./saved_model/{model_name}')
    else:
        model = ModelMRC(huggingface_name)
    model = model.to(device)
    epochs = ep
    lr = 3e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    import random
    # loss_weight = [2, 0.2]
    # for _ in range(510):
    #     loss_weight.append(2)
    # loss_weight = torch.tensor(loss_weight, dtype=torch.float).to(device)
    def loss_func(scores, labels):
        s, e = scores
        sl = torch.tensor([each[0] for each in labels], device=device)
        el = torch.tensor([each[1] for each in labels], device=device)
        total_loss = 0
        for i in range(s.size(0)):
            # lw = loss_weight[:s[i].size(0)]
            # if sl[i]==1 and el[i]==1:
            #     # if random.random()<0.15:
            #     total_loss += torch.nn.CrossEntropyLoss()(s[i], sl[i])
            #     total_loss += torch.nn.CrossEntropyLoss()(e[i], el[i])
            # else:
            total_loss += torch.nn.CrossEntropyLoss()(s[i], sl[i])
            total_loss += torch.nn.CrossEntropyLoss()(e[i], el[i])
            
        return total_loss

    from tqdm import tqdm
    ce_loss = torch.nn.CrossEntropyLoss()
    # train_set = CMRC(split='train')
    # dev_set = CMRC(split='dev')
    for epoch in range(epochs):
        dl = DataLoader(train_set, batch_size=24, collate_fn=get_batch, shuffle=True)
        model.train()
        for i, batch in enumerate(tqdm(dl, leave=True)):
            input_ids, mask, labels, token_type = batch
            input_ids, mask, token_type = input_ids.to(device), mask.to(device), token_type.to(device)
            start_label, end_label = labels
            start_label, end_label = start_label.to(device), end_label.to(device)
            optimizer.zero_grad()
            model_out = model(input_ids, mask=mask, token_type_ids=token_type)
            start, end, clas, feat = model_out#['start_logits'], model_out['end_logits']
            loss = ce_loss(start, start_label)+ce_loss(end, end_label)
            loss.backward()
            optimizer.step()
        
        if dev_set != None:
            el = DataLoader(dev_set, batch_size=4, collate_fn=get_batch, shuffle=False)
            model.eval()
            acc_true = 0
            acc_total = 0
            em_true = 0
            em_total = 0
            for i, batch in enumerate(tqdm(el, leave=True)):
                input_ids, mask, labels, token_type = batch
                input_ids, mask, token_type = input_ids.to(device), mask.to(device), token_type.to(device)
                # start_label, end_label = labels
                # start_label, end_label = start_label.to(device), end_label.to(device)
                scores = model(input_ids, mask=mask, token_type_ids=token_type)
                s, e, c, f = scores#['start_logits'], scores['end_logits']
                s = torch.argmax(s, dim=-1)
                e = torch.argmax(e, dim=-1)
                for j in range(len(labels[0])):
                    if not (labels[0][j] == 1 and labels[1][j] == 1):
                        if labels[0][j] == s[j]:
                            acc_true+=1
                        acc_total+=1
                        if labels[1][j] == e[j]:
                            acc_true+=1
                        acc_total+=1

                        if labels[0][j] == s[j] and labels[1][j] == e[j]:
                            em_true+=1
                        em_total+=1
            print(f'\n{dev_set.name} eval epoch {epoch}, {model_name}, acc {float(acc_true)/float(acc_total)}, EM {float(em_true)/float(em_total)}')
            record.append(f'\n{dev_set.name} eval epoch {epoch}, {model_name if model_name != "" else huggingface_name}, acc {float(acc_true)/float(acc_total)}, EM {float(em_true)/float(em_total)}')
            # torch.save(model, f'./saved_model/MRC_CMRC_ft{epoch}')

        if test_set != None:
            tl = DataLoader(test_set, batch_size=4, collate_fn=get_batch, shuffle=False)
            model.eval()
            acc_true = 0
            acc_total = 0
            em_true = 0
            em_total = 0
            for i, batch in enumerate(tqdm(tl, leave=True)):
                input_ids, mask, labels, token_type = batch
                input_ids, mask, token_type = input_ids.to(device), mask.to(device), token_type.to(device)
                # start_label, end_label = labels
                # start_label, end_label = start_label.to(device), end_label.to(device)
                scores = model(input_ids, mask=mask, token_type_ids=token_type)
                s, e, c, f = scores#['start_logits'], scores['end_logits']
                s = torch.argmax(s, dim=-1)
                e = torch.argmax(e, dim=-1)
                for j in range(len(labels[0])):
                    if not (labels[0][j] == 1 and labels[1][j] == 1):
                        if labels[0][j] == s[j]:
                            acc_true+=1
                        acc_total+=1
                        if labels[1][j] == e[j]:
                            acc_true+=1
                        acc_total+=1

                        if labels[0][j] == s[j] and labels[1][j] == e[j]:
                            em_true+=1
                        em_total+=1
            print(f'\n{test_set.name} test epoch {epoch}, {model_name}, acc {float(acc_true)/float(acc_total)}, EM {float(em_true)/float(em_total)}')
            record.append(f'\n{test_set.name} test epoch {epoch}, {model_name if model_name != "" else huggingface_name}, acc {float(acc_true)/float(acc_total)}, EM {float(em_true)/float(em_total)}')
            # torch.save(model, f'./saved_model/MRC_CMRC_ft{epoch}')

if __name__ == '__main__':
    file = open('./record.txt', 'w')
    datasets_list = [
        [WebQA(split='train'), WebQA(split='dev'), WebQA(split='test'), 1],
        [DRCD(split='train'), DRCD(split='dev'), DRCD(split='test'), 2],
        [CMRC(split='train'), CMRC(split='dev'), CMRC(split='test'), 2]
    ]
    model_names = [
        # ['bert-base-chinese', ''],tev
        # ['hfl/chinese-pert-base', '', 2, 'cuda:3'],
        # ['hfl/chinese-lert-base', '', 2, 'cuda:3'],
        # ['hfl/chinese-roberta-wwm-ext', '', 2, 'cuda:3'],
        # ['hfl/chinese-lert-base', 'lertsampled6000_1.1']
        # ['hfl/chinese-lert-base', 'MRClert6000_1.2'],
        # ['hfl/chinese-lert-base', 'MRClert6000_1.3'],

        ['hfl/chinese-roberta-wwm-ext', 'robertasampled6000_1.1', 'cuda:3'],
        ['hfl/chinese-pert-base', 'pertsampled6000_1.1', 'cuda:3'],
        ['bert-base-chinese', 'bertsampled6000_1.1', 'cuda:3'],
        ['hfl/chinese-lert-base', 'lertsampled6000_1.1', 'cuda:3'],
        # ['hfl/chinese-pert-base', 'MRC6000_1.1', 'cuda:3'],
        # ['hfl/chinese-pert-base', 'MRC6000_1.2', 'cuda:3'],
        # ['hfl/chinese-pert-base', 'MRC6000_1.3', 'cuda:3'],
        # ['hfl/chinese-pert-base', '', 'cuda:3'],
        # ['bert-base-chinese', 'bert6000_1.0', 'cuda:3'],
        # ['bert-base-chinese', 'bert6000_1.1', 'cuda:3'],
        # ['bert-base-chinese', 'bert6000_1.2', 'cuda:3'],
        # ['bert-base-chinese', 'bert6000_1.3', 'cuda:3']
    ]
    # ep = 2
    params = []
    for d in datasets_list:
        for each in model_names:
            main(each[0], each[1], d[0], d[1], d[2], ep=d[3], device=each[2])
    for each in record:
        file.write(each)

    file.close()