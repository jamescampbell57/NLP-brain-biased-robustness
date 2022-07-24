import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
import numpy as np
from scipy.io import loadmat
import os
import random

import scipy.io
import matplotlib.pyplot as plt
import h5py
from pycocotools.coco import COCO
import time
import csv


class HarryPotterDataset(Dataset):
    def __init__(self, subject):

        data_path = '/home/ubuntu/nlp-brain-biased-robustness/data/harry_potter_brain'
        # install dataset
        if not os.path.exists(data_path):
            os.system('mkdir '+data_path)
            for i in range(1,9):
                os.system(f'wget http://www.cs.cmu.edu/~fmri/plosone/files/subject_{i}.mat -P '+data_path)

        harry_potter = loadmat(os.path.join(data_path, f'subject_{subject}.mat'))

        words = []
        for i in range(5176):
            word = harry_potter['words'][0][i][0][0][0][0]
            words.append(word)

        word_times = []
        for i in range(5176):
            word_time = harry_potter['words'][0][i][1][0][0]
            word_times.append(word_time)

        tr_times = []
        for i in range(1351):
            tr_time = harry_potter['time'][i,0]
            tr_times.append(tr_time)

        #dont_include_indices = []
        #for idx, tr_time in enumerate(tr_times):
        #    if not set(np.arange(tr_time - 10, tr_time, .5)).issubset(set(word_times)):
        #        dont_include_indices.append(idx)

        dont_include_indices = [i for i in range(15)] + [i for i in range(335,355)] + [i for i in range(687,707)] + [i for i in range(966,986)] + [i for i in range(1346,1351)]

        X_fmri = harry_potter['data']

        useful_X_fmri = np.delete(X_fmri, dont_include_indices,axis=0)

        #tr_times_arr = np.asarray(tr_times)

        useful_tr_times = np.delete(np.asarray(tr_times), dont_include_indices)

        sentences = [[]]*1271
        for idx, useful_tr_time in enumerate(useful_tr_times):
            sentence= []
            for word, word_time in zip(words,word_times):
                if useful_tr_time - 10 <= word_time <= useful_tr_time:
                    sentence.append(word)
            sentences[idx] = sentence   

        actual_sentences = ['']*1271
        for idx, sentence in enumerate(sentences):
            for word in sentence:
                if word != '+':
                    actual_sentences[idx] = actual_sentences[idx] + word + ' '


        fmri = torch.as_tensor(useful_X_fmri)
        truth_fmri = fmri[:5,:]


        self.fmri_data = []
        for i in range(1271):
            self.fmri_data.append((actual_sentences[i], fmri[i,:]))


    def __getitem__(self, idx):
        return self.fmri_data[idx]

    def __len__(self):
        return len(self.fmri_data)
    
    
class NSD(Dataset):
    def __init__(self, voxels=False):
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
            X = scipy.io.loadmat(f"{data_path}X{subject+1}.mat")
            NSD_fmri_parcellated[:,:,subject] = X['X']


        self.fmri_data = []
        for my_index in range(22000):
            for subject in range(4):
        #for subject in range(4):
            #for my_index in range(22000):
                descriptions = index_to_captions(my_index, subject+1)
                brain_vec = NSD_fmri_parcellated[my_index,:,subject]
                for description in descriptions[:3]:
                    example = (description, brain_vec)
                    self.fmri_data.append(example)

        if voxels:
            #implement separate function for voxel-wise data
            assert False, "voxels not implemented"

        
    def __getitem__(self, idx):
        return self.fmri_data[idx]

    def __len__(self):
        return len(self.fmri_data)

       

#dataset_1 = HarryPotterDataset(1)
#n_rows = len(dataset_1)
#train_dataset = dataset_1[:int(.8*n_rows)]
#val_dataset = dataset_1[int(.9*n_rows):]

#train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#test_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)


class BrainBiasedBERT(nn.Module):
    def __init__(self, num_voxels=37913):
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
    
    


def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_function = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_losses = []
        for batch in dataloader:
            preds = model(list(batch[0]))
            labels = batch[1].to(device)
            test_loss = loss_function(preds, labels.float())
            test_losses.append(test_loss)

    return torch.mean(torch.as_tensor(test_losses)) 

    
def train(model, dataloader, num_epochs=100): 
    optimizer = AdamW(model.parameters(), lr=1e-5)
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
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            wandb.log({"running loss": loss.item()})
            progress_bar.update(1)
        
        val_loss = evaluate(model, test_dataloader)
        wandb.log({"training loss": loss.item()})
        wandb.log({"val loss": val_loss})
        save_dir = '/home/ubuntu/NLP-brain-biased-robustness/state_dicts'
        if epoch % 10 == 0 or epoch == 5 or epoch == 3 or epoch ==1 or epoch ==2 or epoch==4:
            torch.save(model.state_dict(), os.path.join(save_dir, f'cereberto_NSD_parc_4_subjs_epoch_{epoch}'))
        
        

if __name__ == "__main__":
    
    from torch.optim import AdamW
    from transformers import get_scheduler
    from tqdm.auto import tqdm
    import wandb

    run = wandb.init(project="fMRI pretraining", entity="nlp-brain-biased-robustness")
    wandb.run.name = 'NSD parcellated subjects 1-4'
    wandb.config = {
      "learning_rate": 1e-5,
      "epochs": 40,
      "batch_size": 16
    }
        

    dataset_2 = NSD()
    split_place = int(.8*len(dataset_2))
    train_dataset = dataset_2.fmri_data[:split_place]
    val_dataset = dataset_2.fmri_data[split_place:]


    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
        
    model = BrainBiasedBERT(num_voxels=23)
    train(model, train_dataloader)
