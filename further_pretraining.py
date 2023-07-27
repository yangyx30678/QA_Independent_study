from data_QNews import QNews
from torch.utils.data import DataLoader
from model_MRC import ModelMRC
import torch
import transformers as T
import copy
import random
from data_wiki import WIKI
from decimal import *
# getcontext().prec = 8

T.logging.set_verbosity_error()
# transformers.logging.set_verbosity_error()
#/home/FengZQ/Anaconda3/envs/py3713/bin/pip install sentencepiece
def main(lamb, model_name, save_name):

    # torch.manual_seed(0)
    # random.seed(0)

    tokenizer = T.BertTokenizer.from_pretrained(model_name, cache_dir='./cache')
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer = T.AutoTokenizer.from_pretrained('bert_base_chinese', cache_dir='./cache')
    def get_batch(sample):
        encoded = tokenizer.batch_encode_plus([[list(str(each[0])), list(str(each[1]))] for each in sample], padding=True, truncation=True, max_length=512, return_tensors='pt')
        # lengths = [len(each[0])+2 for each in sample]
        # qc = []
        # for each in sample:
        #     if len(each[0]) + len(each[1]) > 509: startidx = random.choice(range(len(each[0]) - (509 - len(each[1]))))
        #     else: startidx = 0
        #     qc.append([101] + each[0][startidx: startidx+(509-len(each[1]))] + [102] + each[1] + [102])
        # maxlen = max([len(each) for each in qc])
        input_ids = encoded['input_ids']
        mask = encoded['attention_mask']
        token_type = encoded['token_type_ids']
        labels = torch.tensor([each[2] for each in sample], dtype=torch.int64)
        # input_ids = torch.zeros((len(sample), maxlen), dtype=torch.int64)
        # mask = torch.zeros((len(sample), maxlen), dtype=torch.int64)
        # token_type = torch.zeros((len(sample), maxlen), dtype=torch.int64)
        # for i in range(len(sample)):
        #     for j in range(len(qc[i])):
        #         input_ids[i, j] = qc[i][j]
        #         mask[i, j] = 1
        #         if j >= lengths[i]:
        #             token_type[i, j] = 1

        return input_ids, mask, labels, token_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModelMRC(model_name=model_name).to(device)
    # model = torch.load('./saved_model/epoch4')
    # model = torch.nn.DataParallel(model, device_ids=[0, 3])
    model_copy = copy.copy(model.pretrained_model).to(device)
    epochs = 1
    lr = 5e-6
    lamb = lamb
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    import random
    # loss_weight = [2, 0.2]
    # for _ in range(510):
    #     loss_weight.append(2)
    # loss_weight = torch.tensor(loss_weight, dtype=torch.float).to(device)
    def loss_func(scores, labels, feat, feat_copy):
        # s, e = scores
        # sl = torch.tensor([each[0] for each in labels], device=device)
        # el = torch.tensor([each[1] for each in labels], device=device)
        total_loss = 0
        for i in range(scores.size(0)):
            distance = torch.nn.functional.pairwise_distance(feat[i], feat_copy[i])
            distance = torch.sum(distance)/distance.size(0)
            total_loss += torch.nn.functional.cross_entropy(scores[i], labels[i])/(lamb**distance)

            
        return total_loss

    from tqdm import tqdm
    ce_loss = loss_func#torch.nn.CrossEntropyLoss()
    qnews = WIKI(new=False)#QNews()
    dl = DataLoader(qnews, batch_size=24, collate_fn=get_batch, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_bar = tqdm(dl, leave=True)
        epoch_bar.set_description(f"loss:{1000}")
        for i, batch in enumerate(epoch_bar):
            input_ids, mask, labels, token_type = batch
            input_ids, mask, labels, token_type = input_ids.to(device), mask.to(device), labels.to(device), token_type.to(device)
            optimizer.zero_grad()
            
            model_out = model(input_ids, mask=mask)#, token_type_ids=token_type)
            with torch.no_grad():
                feat_copy = model_copy(input_ids, attention_mask=mask, token_type_ids=token_type)
            start, end, clas, feat = model_out#['start_logits'], model_out['end_logits']
            feat, feat_copy = feat.last_hidden_state, feat_copy.last_hidden_state
            
            loss = ce_loss(clas, labels, feat, feat_copy)#ce_loss(clas, labels)/(lamb**(torch.nn.functional.kl_div(feat, feat_copy)))
            loss.backward()
            epoch_bar.set_description(f"loss:{loss}")
            optimizer.step()
            if (i+1) % 3000 == 0:
                torch.save(model, f'./saved_model/{save_name}{i+1}_{lamb}')
            if (i+1) == 9000:
                break
        
        # el = DataLoader(dev_set, batch_size=4, collate_fn=get_batch, shuffle=False)
        # model.eval()
        # acc_true = 0
        # acc_total = 0
        # em_true = 0
        # em_total = 0
        # for i, batch in enumerate(tqdm(el, leave=True)):
        #     input_ids, mask, labels, token_type = batch
        #     input_ids, mask, token_type = input_ids.to(device), mask.to(device), token_type.to(device)
        #     # start_label, end_label = labels
        #     # start_label, end_label = start_label.to(device), end_label.to(device)
        #     scores = model(input_ids, mask=mask, token_type_ids=token_type)
            
        #     s, e, c = scores#['start_logits'], scores['end_logits']
        #     s = torch.argmax(s, dim=-1)
        #     e = torch.argmax(e, dim=-1)
        #     for j in range(len(labels[0])):
        #         if not (labels[0][j] == 1 and labels[1][j] == 1):
        #             if labels[0][j] == s[j]:
        #                 acc_true+=1
        #             acc_total+=1
        #             if labels[1][j] == e[j]:
        #                 acc_true+=1
        #             acc_total+=1

        #             if labels[0][j] == s[j] and labels[1][j] == e[j]:
        #                 em_true+=1
        #             em_total+=1
        # print(f'\nepoch {epoch}, acc {float(acc_true)/float(acc_total)}, EM {float(em_true)/float(em_total)}')
        # torch.save(model, f'./saved_model/roberta_ft{epoch}')

if __name__=='__main__':
    random.seed(0)
    main(1.1, 'hfl/chinese-lert-base', 'lertsampled_final')
    # random.seed(0)
    # main(1.1, 'hfl/chinese-roberta-wwm-ext', 'robertasampled')
    # random.seed(0)
    # main(1.1, 'bert-base-chinese', 'bertsampled')
    # random.seed(0)
    # main(1.1, 'bert-base-chinese', 'bert')
    # random.seed(0)
    # main(1.2, 'bert-base-chinese', 'bert')
    # random.seed(0)
    # main(1.3, 'bert-base-chinese', 'bert')
    # random.seed(0)
    # main(1.0, 'hfl/chinese-roberta-wwm-ext', 'roberta')
    # random.seed(0)
    # main(1.1, 'hfl/chinese-roberta-wwm-ext', 'roberta')
    # random.seed(0)
    # main(1.2, 'hfl/chinese-roberta-wwm-ext', 'roberta')
    # random.seed(0)
    # main(1.3, 'hfl/chinese-roberta-wwm-ext', 'roberta')