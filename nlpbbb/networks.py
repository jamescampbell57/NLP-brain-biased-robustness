import torch.nn as nn
from transformers import BertModel

class PlaceHolderBERT(nn.Module):
    def __init__(self, model_config):
        #num_out=5, sigmoid=False, return_CLS_representation=False
        super().__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768, model_config["num_out"])
        self.return_CLS_representation = model_config["return_CLS_rep"]
        self.sigmoid_bool = model_config["sigmoid"]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
        #embeddings.to(device)
        representations = self.bert(**x).last_hidden_state
        cls_representation = representations[:,0,:]
        pred = self.linear(cls_representation)
        if self.return_CLS_representation:
            return cls_representation
        if self.sigmoid_bool:
            return self.sigmoid(pred)
        return pred