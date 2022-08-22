import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, num_out=5, return_CLS_representation=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768,num_out)
        self.return_CLS_representation = return_CLS_representation
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
        #embeddings = embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        pred = self.linear(cls_representation)
        if self.return_CLS_representation:
            return cls_representation
        return self.softmax(pred)