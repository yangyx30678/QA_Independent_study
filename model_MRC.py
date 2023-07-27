import transformers as T
from transformers.models.bert.modeling_bert import *
from transformers.models.roberta.modeling_roberta import *
import torch
import copy
from torch.utils.checkpoint import checkpoint

class ModelMRC(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.pretrained_model = T.AutoModel.from_pretrained(model_name)
        self.answer_linear = torch.nn.Linear(768, 2)
        self.classifier = torch.nn.Linear(768, 2)

    # def module_forward(self, m, x, mask=None, tti=None):
    #     if m == self.pretrained_model:
    #         return m(x, attention_mask=mask, token_type_ids=tti)
    #     else:
    #         return m(x)

    def forward(self, x, mask, token_type_ids=None):
        # feat = self.pretrained_model(x, attention_mask=mask, token_type_ids=token_type_ids)
        # ans = self.answer_linear(feat.last_hidden_state)
        pretr_LM = lambda x, mask, token_type_ids : self.pretrained_model(x, attention_mask=mask, token_type_ids=token_type_ids).last_hidden_state
        # a_classifier = lambda x : self.answer_linear(x)
        # feat = checkpoint(pretr_LM, x, mask, token_type_ids)
        # ans = checkpoint(a_classifier, feat.last_hidden_state)
        feat = self.pretrained_model(x, attention_mask=mask, token_type_ids=token_type_ids)
        ans = self.answer_linear(feat.last_hidden_state)
        start, end = torch.split(ans, 1, 2)

        start = start.squeeze(-1).contiguous()
        end = end.squeeze(-1).contiguous()
        clas = self.classifier(feat.last_hidden_state[:, 0, :])
        # clas = torch.nn.Flatten()(clas)
        return start, end, clas, feat
        # return torch.cat((start, end))



if __name__ == '__main__':
    m = Model4QA()
    seq =  torch.tensor([[1,2,3,4,5,21000], [1,2,3,4,5,21000]])
    mask =  torch.tensor([[1,1,1,0,0,0], [1,1,1,0,0,0]])
    tt = torch.tensor([[0,0,1,1,1,1], [0,0,1,1,1,1]])

    m(seq, mask, tt)