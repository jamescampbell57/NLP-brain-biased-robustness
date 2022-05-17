import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import h5py
from pycocotools.coco import COCO
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel

data_path = '/home/ubuntu/NLP-brain-biased-robustness/NSD/'

coco3 = COCO(data_path+'annotations/captions_train2017.json')
coco4 = COCO(data_path+'annotations/captions_val2017.json')

def load_csv(csv_file):
    file = open(csv_file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows

nsd_to_coco = load_csv(data_path+'nsd_stim_info_merged.csv')
exp_design = scipy.io.loadmat(data_path+'nsd_expdesign.mat')
ordering = exp_design['masterordering'].flatten() - 1 #fix indexing

data_size = 22500 #trials[subject-1] #can use more than 22500 trials if seems promising
ordering_data = ordering[:data_size]
subjectim = exp_design['subjectim'] - 1

def index_to_captions(my_index, subject):
    index = ordering_data[my_index]
    nsd_id = subjectim[subject-1,index]
    coco_id = nsd_to_coco[nsd_id][1]
    if int(nsd_id) < 2950:
        annotation_ids = coco4.getAnnIds(int(coco_id))
        annotations = coco4.loadAnns(annotation_ids)
    else:
        annotation_ids = coco3.getAnnIds(int(coco_id))
        annotations = coco3.loadAnns(annotation_ids)
    captions = [item['caption'] for item in annotations]
    return captions

NSD_fmri_parcellated = np.empty((22500,23,8))
for subject in range(8):
    X = scipy.io.loadmat(data_path+'X'+str(subject+1)+'.mat')
    NSD_fmri_parcellated[:,:,subject] = X['X']
    

dataset = []
for subject in range(4):
    for my_index in range(22000):
        descriptions = index_to_captions(my_index, 1)
        brain_vec = NSD_fmri_parcellated[my_index,:,0]
        for description in descriptions[:3]:
            example = (description, brain_vec)
            dataset.append(example)
            
import random

random.shuffle(dataset)
split_place = int(.8*len(dataset))
train_dataset = dataset[:split_place]
val_dataset = dataset[split_place:]



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                          
                          
                          
                          
class BrainBiasedBERT(nn.Module):
    def __init__(self, num_voxels=23):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768,num_voxels)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
        embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        pred_fmri = self.linear(cls_representation)
        return pred_fmri
                          
                          
                          
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import wandb

wandb.init(project="NSD finetuning", entity="nlp-brain-biased-robustness")

wandb.config = {
  "learning_rate": 5e-5,
  "epochs": 20,
  "batch_size": 8
}

def evaluate(model, dataloader, loss_function):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    with torch.no_grad():
        test_losses = []
        for batch in dataloader:
            preds = model(list(batch[0]))
            labels = batch[1].to(device)
            test_loss = loss_function(preds, labels.float())
            test_losses.append(test_loss)

    return torch.mean(torch.as_tensor(test_losses)) 

    
def train(model, dataloader, num_epochs=30): 
    last_val_loss = 9223372036854775807
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_function = torch.nn.MSELoss()
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            preds = model(list(batch[0]))
            labels = batch[1].to(device)
            loss = loss_function(preds, labels.float()) #replace .loss
            loss.backward()
            
            wandb.log({"running loss": loss})
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        val_loss = evaluate(model, val_loader, loss_function)
        wandb.log({"training loss": loss})
        wandb.log({"val loss": val_loss})
        #if val_loss > last_val_loss:
        #    print('Stopped early')
        #if epoch % 2 == 0:
        torch.save(model.state_dict(), 'NSD_model_prime_prime_epoch_'+str(epoch))
            #break
        #last_val_loss = val_loss
        
model = BrainBiasedBERT()
train(model, train_loader)
        
