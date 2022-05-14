import os

os.system('pip install torch')
os.system('pip install transformers')
os.system('pip install datasets')
os.system('pip install tqdm')
os.system('pip install wandb')


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
from datasets import load_dataset

from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import wandb

################################################################################################

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

class PlaceHolderBERT(nn.Module):
    def __init__(self, num_out=1, sigmoid=False, return_CLS_representation=False, brain=True):
        super().__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if brain:
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
    
    
def train(model, dataloader, num_epochs=8):
    
    
    wandb.init(project="imdb sst2", entity="nlp-brain-biased-robustness")

    wandb.config = {
      "learning_rate": 5e-5,
      "epochs": 10,
      "batch_size": 8
    }
    
    #optimizer as usual
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_function = torch.nn.MSELoss()
    #learning rate scheduler
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    #auto logging; progress bar
    progress_bar = tqdm(range(num_training_steps))

    #training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader: #tryin unpacking text from 'labels' as in model development
            batch = {k: v.to(device) for k, v in batch.items()}
            features = {k: v for k, v in batch.items() if k != 'labels'}
            preds = model(features)
            loss = loss_function(preds, batch['labels'].float()) #replace .loss
            loss.backward()
            
            wandb.log({"loss": loss})
            #wandb.watch(model)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        IID_score = evaluate(model, imdb_test_loader)
        OOD_score = evaluate(model, sst2_test_loader)
        wandb.log({"IID score": IID_score})
        wandb.log({"OOD score": OOD_score})

def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_samples = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        features = {k: v for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            preds = model(features)
            preds = torch.where(preds < .5, 0, 1)
            labels = batch['labels'].reshape(preds.shape)
            num_correct += (preds==labels).sum()
            num_samples += preds.size(0)
    return float(num_correct)/float(num_samples)*100 




imdb = load_dataset('imdb')
sst2 = load_dataset('glue','sst2')

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#tokenize function
def tokenize_imdb(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def tokenize_sst2(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

#pre-tokenize entire dataset
tokenized_imdb = imdb.map(tokenize_imdb, batched=True)
tokenized_sst2 = sst2.map(tokenize_sst2, batched=True)

tokenized_imdb = tokenized_imdb.remove_columns(["text"])
tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
tokenized_imdb.set_format("torch")

tokenized_sst2 = tokenized_sst2.remove_columns(["sentence","idx"])
tokenized_sst2 = tokenized_sst2.rename_column("label", "labels")
tokenized_sst2.set_format("torch")


### Only for practice
imdb_small_train = tokenized_imdb['train'].shuffle(seed=42).select(range(1000))
imdb_small_test = tokenized_imdb['test'].shuffle(seed=42).select(range(500))
###
imdb_train_loader = DataLoader(imdb_small_train, shuffle=True, batch_size=8)
imdb_test_loader = DataLoader(imdb_small_test, shuffle=True, batch_size=8)

sst2_small_train = tokenized_sst2["train"].shuffle(seed=42).select(range(1000))
sst2_small_test = tokenized_sst2["validation"].shuffle(seed=42).select(range(500)) #actual test set is fucked up

sst2_train_loader = DataLoader(sst2_small_train, shuffle=True, batch_size=8)
sst2_test_loader = DataLoader(sst2_small_test, shuffle=True, batch_size=8)


model = PlaceHolderBERT(brain=True)
train(model, imdb_train_loader)
model2 = PlaceHolderBERT(brain=False)
train(model, imdb_train_loader)

