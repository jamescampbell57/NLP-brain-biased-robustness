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
    
    
def train(model, dataloader, num_epochs=10): #can scrap keyword
    
    
    wandb.init(project="stsb", entity="nlp-brain-biased-robustness")

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
            
            wandb.log({"loss": loss})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        IID_score = evaluate(model, headlines_dataloader)
        images_score = evaluate(model, images_dataloader)
        vid_score = evaluate(model, MSRvid_dataloader)
        par_score = evaluate(model, MSRpar_dataloader)
        wandb.log({"IID score": IID_score})
        wandb.log({"images score": images_score})
        wandb.log({"vid score": vid_score})
        wandb.log({"par score": par_score})
        
            

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


import csv
data_path = '/home/ubuntu/nlp-brain-biased-robustness/data/stsb/stsbenchmark'
if not os.path.exists(data_path):
    dataset_path = '/home/ubuntu/nlp-brain-biased-robustness/data/stsb'
    os.system('mkdir '+dataset_path)
    os.system('wget https://data.deepai.org/Stsbenchmark.zip -P '+dataset_path)
    os.system(f'unzip /home/ubuntu/nlp-brain-biased-robustness/data/stsb/Stsbenchmark.zip -d /home/ubuntu/nlp-brain-biased-robustness/data/stsb/')


def read_csv(csv_file):
    file = open(csv_file)
    csvreader = csv.reader(file, delimiter="\t")
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows

train_set = read_csv(os.path.join(data_path,'sts-train.csv'))
dev_set = read_csv(os.path.join(data_path,'sts-dev.csv'))
test_set = read_csv(os.path.join(data_path,'sts-test.csv'))

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

model = PlaceHolderBERT(brain=True)
train(model, headlines_dataloader)
model2 = PlaceHolderBERT(brain=False)
train(model, headlines_dataloader)

