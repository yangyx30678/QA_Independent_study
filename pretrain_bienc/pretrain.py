from bienc_model import biencoder_model
from data_mt5gen import data_mt5gen
from data_wiki import WIKI
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import transformers as T
import math
import torch
samples=4
dl = WIKI(new=False, samples=samples)
tokenizer = T.AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
def get_batch(sample):
    query = [each[0] for each in sample]
    context = [each[1] for each in sample]
    label = [each[2] for each in sample]
    # qmax = max([len(each) for each in query])
    # cmax = max([max([len(c) for c in q]) for q in context])
    # q = torch.zeros((len(query), qmax), dtype=torch.int64)
    # c = torch.zeros((len(context), len(context[0]), cmax), dtype=torch.int64)
    q = tokenizer.batch_encode_plus(query, max_length=512, padding=True, truncation=True, return_tensors='pt')['input_ids']
    cont = []
    for each in context:
        cont.extend(each)
    c = tokenizer.batch_encode_plus(cont, max_length=512, padding=True, truncation=True, return_tensors='pt')['input_ids'].view([len(sample), samples, -1])
    l = torch.tensor(label)
    # for i in range(len(query)):
    #     for j in range(len(query[i])):
    #         q[i, j] = query[i][j]
    #     for j in range(len(context[i])):
    #         for k in range(len(context[i][j])):
    #             c[i, j ,k] = context[i][j][k]
    return q, c, l

model = biencoder_model()
# model = torch.load('./saved_model/roberta_retriever_1_8800')
# model.load_state_dict(torch.load('./saved_model/roberta_retriever_1_1400', map_location='cpu')['model'])
model = torch.nn.DataParallel(model, device_ids=[1,3])
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train_dataloader = DataLoader(dl, batch_size=12, collate_fn=get_batch, shuffle=True)


model = model.to(device)
epochs = 900
PATH = './roberta_retriver'
model.train()
class custom_loss(torch.nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
        
    def forward(self, output, target):
        # temp = 0
        # for i in output:
        #     temp += torch.exp(i)
            #print(math.exp(i))
        #print(math.exp(output[target]) ,temp)
        loss_sum = 0
        for i in range(output.size(0)):
            loss_sum += -torch.log(torch.exp(output[i, target[i]]) / torch.sum(torch.exp(output[i]), dim=-1).requires_grad_())
        return loss_sum#- torch.log(torch.exp(output[target]) / torch.sum(torch.exp(output), dim=-1).requires_grad_())

#criterion = nn.MSELoss()
# criterion = torch.nn.CrossEntropyLoss()
criterion = custom_loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
optimizer.zero_grad()
min_loss = 1000
min_batch = 0
for epoch in range(epochs):
    
    # if (epoch) % 4 == 0:
    #     for param in model.module.parameters():
    #         param.requires_grad = True
    # else:      # 加強對query的訓練
    #     for param in model.module.parameters():
    #         param.requires_grad = False
    #     for param in model.module.context_encoder.parameters():
    #         param.requires_grad = True
    
    acc_loss_n = 2  # greadient acc times
    running_loss = 0.0
    epoch_loss = 0.0  # 紀錄用
    epoch_bar = tqdm(train_dataloader)
    epoch_bar.set_description("epoch: %d , loss: %f , min: %f, min batch: %d, batch: " %(epoch + 1, running_loss, min_loss, min_batch))
    
    for  i, data in enumerate(epoch_bar):
        query, context, label = data
        outputs = model(query.to(device), context.to(device))#, label.to(device))
        #print(outputs)
        #print(same_r, diff_r)
        # print(outputs)
        loss = criterion(outputs, label.to(device))
        #loss = outputs
        #loss = loss / acc_loss_n
        #print(loss)
        loss.backward()
        
        running_loss += loss.item() * acc_loss_n
        if ((i + 1) % acc_loss_n == 0):
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss = running_loss / (i + 1)
            epoch_bar.set_description("epoch: %d , loss: %f , min: %f, min batch: %d, batch: " %(epoch + 1, epoch_loss, min_loss, min_batch))
        #break  # test
        if (i % 6000) == 0 and i != 0:
            if epoch_loss <= min_loss:
                min_loss = epoch_loss
                min_batch = i
                epoch_bar.set_description("epoch: %d , loss: %f , min: %f, min batch: %d, batch: " %(epoch + 1, epoch_loss, min_loss, min_batch))
            torch.save(model.module, f"./saved_model/roberta_wikiretriever_{str(epoch + 1)}_{i}")
    
    # if (epoch_loss <= min_loss) or (epoch == 0):
    #     min_loss = epoch_loss
    #     epoch_bar.set_description("epoch: %d , loss: %f , min: %f, batch: " %(epoch + 1, epoch_loss, min_loss))
    #     torch.save({'model': model.module.state_dict()}, './saved_model/roberta_retriver_min.pth.tar')
    
    if (epoch % 1) == 0:
        torch.save(model.module, "./saved_model/roberta_wikiretriever_" + str(epoch + 1))
        
    epoch_bar.close()
    #break  # test
#     if (epoch % 1) == 0:
#         torch.save({'model': model.state_dict()}, 'roberta_retriver.pth.tar')
        #torch.save(model, PATH + "_" + str(epoch + 1) + ".pth")
print('Finished Training')
# ==================================================================
# 儲存模型
#torch.save({'model': model.state_dict()}, 'roberta_retriver.pth.tar')
#torch.save(model, PATH + "_" + str(epoch + 1) + ".pth")