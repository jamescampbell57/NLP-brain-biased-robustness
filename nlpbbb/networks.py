import torch.nn as nn
from transformers import BertModel

class AmazonBERT(nn.Module):
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

    
class MNLIBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768*2, 3)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x, y):
        x_embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        y_embeddings = self.tokenizer(y, return_tensors='pt', padding=True, truncation=True)
        x_embeddings.to(self.device)
        y_embeddings.to(self.device)
        x_representations = self.bert(**x_embeddings).last_hidden_state
        x_cls_representation = x_representations[:,0,:]
        y_representations = self.bert(**y_embeddings).last_hidden_state
        y_cls_representation = y_representations[:,0,:]
        input_vec = torch.cat((x_cls_representation, y_cls_representation), axis=1)
        pred = self.linear(input_vec)
        return pred
    
class SST2BERT(nn.Module):
    def __init__(self, num_out=1, sigmoid=False, return_CLS_representation=False):
        super().__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 
        self.bert = BertModel.from_pretrained('bert-base-cased')
        state_path = '/home/ubuntu/NLP-brain-biased-robustness/notebooks/fine_tuned_model'
        pre_odict = torch.load(state_path)
        filtered_odict = change_all_keys(pre_odict)
        self.bert.load_state_dict(filtered_odict, strict=True)
        self.linear = nn.Linear(768,num_out)
        self.return_CLS_representation = return_CLS_representation
        self.sigmoid_bool = sigmoid
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