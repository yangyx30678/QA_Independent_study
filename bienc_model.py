import torch
import transformers as T
import math
class roberta_fc(torch.nn.Module):   # one roberta with fc       cls fc pooling
    def __init__(self):
        super(roberta_fc, self).__init__()
        self.encoder = T.BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext") 
        self.fc1 = torch.nn.Linear(in_features=768, out_features=768, bias=True)
        # self.norm = torch.nn.LayerNorm(768)
    def forward(self, context):
        c_rep = self.encoder(context)
        
        cls_rep = []  # cls pooling
        for rep in c_rep[0]:
            cls_rep.append(torch.unsqueeze(rep[0], dim=0))
        cls_rep = torch.cat(cls_rep, dim=0)
        
        out = self.fc1(cls_rep)
        # out = self.norm(out)   # batch_size*512*768
        #out = self.fc1(c_rep[0])   # batch_size*512*768
        return out
    
class biencoder_model(torch.nn.Module):   #Q+---  Crossentropy
    def __init__(self):
        super(biencoder_model, self).__init__()
        self.question_encoder = roberta_fc()
        self.context_encoder = roberta_fc()             
    
    def forward(self, query, context):
        similarity = []
        batch_size = len(query)   
        q_rep = self.question_encoder(query)
        
        for i, c in enumerate(context):   # batch中每筆data的正負段落集分次進入
            para_num = len(c)  # 正負段落數
            c_rep = self.context_encoder(c)
            temp = []

            for p in range(para_num):    # 對每筆段落算sim   
                sim = torch.dot(q_rep[i], c_rep[p])/math.sqrt(q_rep.size(-1))     # cls  fc
                #sim = torch.dot(q_rep[i][0], c_rep[p][0])     # cls  fc
                temp.append(torch.unsqueeze(sim, dim=0))
                
            temp = torch.cat(temp, dim=0)
            similarity.append(torch.unsqueeze(temp, dim=0)) # 回傳batch size數的sim值
        
        similarity = torch.cat(similarity, dim=0)
        return similarity