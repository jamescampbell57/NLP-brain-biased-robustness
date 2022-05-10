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
    
    
    wandb.init(project="preliminary results just in case", entity="nlp-brain-biased-robustness")

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
    
    
            
    wandb.init(project="preliminary results just in case", entity="nlp-brain-biased-robustness")

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





import csv
data_path = '/home/ubuntu/NLP-brain-biased-robustness/data/stsb/stsbenchmark/'

#wget https://data.deepai.org/Stsbenchmark.zip

def read_csv(csv_file):
    file = open(csv_file)
    csvreader = csv.reader(file, delimiter="\t")
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows




train_set = read_csv(data_path+'sts-train.csv')
dev_set = read_csv(data_path+'sts-dev.csv')
test_set = read_csv(data_path+'sts-test.csv')


def split_data():
    headlines = []
    images = []
    MSRpar = []
    MSRvid = []
    for dataset in [train_set, dev_set, test_set]:
        for i in range(len(dataset)):
            if dataset[i][1] == 'headlines':
                headlines.append(dataset[i])
            if dataset[i][1] == 'images':
                images.append(dataset[i])
            if dataset[i][1] == 'MSRpar':
                MSRpar.append(dataset[i])
            if dataset[i][1] == 'MSRvid':
                MSRvid.append(dataset[i])
    return headlines, images, MSRpar, MSRvid



headlines, images, MSRpar, MSRvid = split_data()


def create_dataset(split):
    dataset = []
    for example in split:
        if not len(example) < 7:
            data = {}
            data['sentence_1'] = example[5]
            data['sentence_2'] = example[6]
            data['labels'] = float(example[4])
            dataset.append(data)
    return dataset

headlines_dataset = create_dataset(headlines)
images_dataset = create_dataset(images)
MSRpar_dataset = create_dataset(MSRpar)
MSRvid_dataset = create_dataset(MSRvid)

headlines_dataloader = DataLoader(headlines_dataset)
images_dataloader = DataLoader(images_dataset)
MSRpar_dataloader = DataLoader(MSRpar_dataset)
MSRvid_dataloader = DataLoader(MSRvid_dataset)




class PlaceHolderBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        return cls_representation
    
    
def train(model, dataloader, num_epochs=1): #can scrap keyword
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
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    #training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader: #tryin unpacking text from 'labels' as in model development
            #batch = {k: v.to(device) for k, v in batch.items()}
            #features = {k: v for k, v in batch.items() if k != 'labels'}
            vec_1 = model(batch['sentence_1'])
            vec_2 = model(batch['sentence_2'])
            cosine_similarity_times_5 = cos(vec_1, vec_2) * 5
            targets = batch['labels'].float().to(device)
            loss = loss_function(cosine_similarity_times_5, targets) #replace .loss
            loss.backward()
            

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            

def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarities = []
    gold = []
    for batch in dataloader:
        with torch.no_grad():
            vec_1 = model(batch['sentence_1'])
            vec_2 = model(batch['sentence_2'])
            cosine_similarity = cos(vec_1, vec_2)
            golds = batch['labels'].float()
            for idx, similarity in enumerate(cosine_similarity):
                cosine_similarities.append(similarity)
                gold.append(golds[idx])
    torch_cosines = torch.tensor(cosine_similarities)
    torch_gold = torch.tensor(gold)
    
    torch_cosines = torch_cosines.reshape((1,torch_cosines.shape[0]))
    torch_gold = torch_gold.reshape((1,torch_gold.shape[0]))
    
    combined = torch.cat((torch_cosines, torch_gold), axis=0)
    
    return torch.corrcoef(combined)[1,1]

