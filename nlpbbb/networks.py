# hf imports
from transformers import BertModel
from transformers import AutoTokenizer

# torch imports
import torch
import torch.nn as nn

# bbb imports
from nlpbbb.paths import PATHS

# random imports
import os

def change_all_keys(pre_odict):
    def change_key(odict, old, new):
        for _ in range(len(odict)):
            k, v = odict.popitem(False)
            odict[new if old == k else k] = v
            return odict
    for key in pre_odict.keys():
        if key[:5] == 'bert.':
            post_odict = change_key(pre_odict, key, key[5:])
            return change_all_keys(post_odict)
        if key[:7] == 'linear.':
            del pre_odict[key]
            return change_all_keys(pre_odict)
    return pre_odict

class AmazonBERT(nn.Module):
    def __init__(self, model_config):
        #num_out=5, sigmoid=False, return_CLS_representation=False
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if model_config["brain_biased"]:
            state_path = os.path.join(PATHS["root"], model_config["state_path"])
            pre_odict = torch.load(state_path)["model_state_dict"]
            filtered_odict = change_all_keys(pre_odict)
            self.bert.load_state_dict(filtered_odict, strict=True)
        self.linear = nn.Linear(768, 5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
        #embeddings.to(device)
        representations = self.bert(**x).last_hidden_state
        cls_representation = representations[:,0,:]
        pred = self.linear(cls_representation)
        #if self.return_CLS_representation:
        #    return cls_representation
        #if self.sigmoid_bool:
        #    return self.sigmoid(pred)
        return pred

    
class MNLIBert(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if model_config["brain_biased"]:
            state_path = os.path.join(PATHS["root"], model_config["state_path"])
            pre_odict = torch.load(state_path)["model_state_dict"]
            filtered_odict = change_all_keys(pre_odict)
            self.bert.load_state_dict(filtered_odict, strict=True)
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
    def __init__(self, model_config, num_out=1, sigmoid=False, return_CLS_representation=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if model_config["brain_biased"]:
            state_path = os.path.join(PATHS["root"], model_config["state_path"])
            pre_odict = torch.load(state_path)["model_state_dict"]
            filtered_odict = change_all_keys(pre_odict)
            self.bert.load_state_dict(filtered_odict, strict=True)
        self.linear = nn.Linear(768,num_out)
        self.return_CLS_representation = return_CLS_representation
        self.sigmoid_bool = sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        representations = self.bert(**x).last_hidden_state
        cls_representation = representations[:,0,:]
        pred = self.linear(cls_representation)
        if self.return_CLS_representation:
            return cls_representation
        if self.sigmoid_bool:
            return self.sigmoid(pred)
        return pred
    
    
class STSBBERT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if model_config["brain_biased"]:
            state_path = os.path.join(PATHS["root"], model_config["state_path"])
            pre_odict = torch.load(state_path)["model_state_dict"]
            filtered_odict = change_all_keys(pre_odict)
            self.bert.load_state_dict(filtered_odict, strict=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        return cls_representation
    
    
class YelpBERT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if model_config["brain_biased"]:
            state_path = os.path.join(PATHS["root"], model_config["state_path"])
            pre_odict = torch.load(state_path)["model_state_dict"]
            filtered_odict = change_all_keys(pre_odict)
            self.bert.load_state_dict(filtered_odict, strict=True)
        self.linear = nn.Linear(768,5)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        pred = self.linear(cls_representation)
        return pred
    

class ReCoRDBERT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if model_config["brain_biased"]:
            state_path = os.path.join(PATHS["root"], model_config["state_path"])
            pre_odict = torch.load(state_path)["model_state_dict"]
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
    

class BrainBiasedBERT(nn.Module):
    def __init__(self, num_voxels=37913):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768,num_voxels)
    def forward(self, x):
        representations = self.bert(**x).last_hidden_state
        cls_representation = representations[:,0,:]
        pred_fmri = self.linear(cls_representation)
        return pred_fmri
    
    
                         
class NSDBiasedBERT(nn.Module):
    def __init__(self, num_voxels=23):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768,num_voxels)
    def forward(self, x):
        representations = self.bert(**x).last_hidden_state
        cls_representation = representations[:,0,:]
        pred_fmri = self.linear(cls_representation)
        return pred_fmri
