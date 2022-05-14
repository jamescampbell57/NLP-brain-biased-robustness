import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
from datasets import load_dataset

from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm



import json
data_path = '/home/ubuntu/NLP-brain-biased-robustness/data/yelp/'
f1 = open(data_path+'american.json')
f2 = open(data_path+'italian.json')
f3 = open(data_path+'japanese.json')
f4 = open(data_path+'chinese.json')

american = []
for line in f1:
    american.append(json.loads(line))

italian = []
for line in f2:
    italian.append(json.loads(line))
    
japanese = []
for line in f3:
    japanese.append(json.loads(line))
    
chinese = []
for line in f4:
    chinese.append(json.loads(line))

f1.close()
f2.close()
f3.close()
f4.close()


american = american[0]
italian = italian[0]
japanese = japanese[0]
chinese = chinese[0]



import torch
import torch.nn.functional as F

na = []
for i in american:
    na.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})

ni = []
for i in italian:
    ni.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})

nj = []
for i in japanese:
    nj.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})

nc = []
for i in chinese:
    nc.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})
    
    
    
american_dataloader = DataLoader(na, shuffle=True, batch_size=8)
italian_dataloader = DataLoader(ni, shuffle=True, batch_size=8)
japanese_dataloader = DataLoader(nj, shuffle=True, batch_size=8)
chinese_dataloader = DataLoader(nc, shuffle=True, batch_size=8)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel

from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

class PlaceHolderBERT(nn.Module):
    def __init__(self, num_out=5, sigmoid=False, return_CLS_representation=False, brain=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        pred = self.linear(cls_representation)
        if self.return_CLS_representation:
            return cls_representation
        if self.sigmoid_bool:
            return self.sigmoid(pred)
        return pred
    
    
def train(model, dataloader, num_epochs=4): #can scrap keyword
    
    
            
    wandb.init(project="yelp prelim", entity="nlp-brain-biased-robustness")

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
            #batch = {k: v.to(device) for k, v in batch.items()}
            #features = {k: v for k, v in batch.items() if k != 'labels'}
            preds = model(batch['text'])
            targets = batch['labels'].float().to(device)
            loss = loss_function(preds, targets) #replace .loss
            loss.backward()
            wandb.log({"loss": loss})
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        IID_score = evaluate(model, japanese_dataloader)
        chinese = evaluate(model, chinese_dataloader)
        american = evaluate(model, american_dataloader)
        italian = evaluate(model, italian_dataloader)
        wandb.log({"japanese score": IID_score})
        wandb.log({"chinese score": chinese})
        wandb.log({"american score": american})
        wandb.log({"italian score": italian})  
            

def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_samples = 0
    for batch in dataloader:
        #batch = {k: v.to(device) for k, v in batch.items()}
        #features = {k: v for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            preds = model(batch['text'])
            preds = torch.argmax(preds, axis=1)
            labels = torch.argmax(batch['labels'], axis=1).to(device)
            num_correct += (preds==labels).sum()
            num_samples += preds.size(0)
    return float(num_correct)/float(num_samples)*100 

model = PlaceHolderBERT(brain=True)
train(model, japanese_dataloader)
model2 = PlaceHolderBERT(brain=False)
train(model2, japanese_dataloader)


