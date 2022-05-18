# hf imports
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import get_scheduler

# torch imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# nlpbbb imports
import nlpbbb as bbb
from nlpbbb.paths import PATHS

# random imports
from scipy.io import loadmat
import numpy as np

import scipy.io
import matplotlib.pyplot as plt
import h5py
from pycocotools.coco import COCO
import time
import csv
import random


class Experiment():
    
    def __init__(self, config):
        self.train_dataset = NSDDataset(split="train")
        self.val_dataset = NSDDataset(split="val")
        
        # handles two cases: you want to validate internally or using another experiment object (dont do this)
        self.train_loaders = [DataLoader(self.train_dataset, batch_size=config["experiment"]["batchsize"], shuffle=True)]
        self.val_loaders = [DataLoader(self.val_dataset, batch_size=config["experiment"]["batchsize"], shuffle=False)]
        
        # really you only want to build a model for an experiment object if it is the train experiment
        self.model = self.get_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["experiment"]["lr"])
        num_iters = sum([len(dl) for dl in self.train_loaders])
        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_iters)
        self.lr_scheduler = None
        self.loss_function = torch.nn.MSELoss()
        
                                           
    def get_model(self):
        return bbb.networks.NSDBiasedBERT()
        
    def train_forward_pass(self, batch, device):
        embeddings = self.model.tokenizer(list(batch[0]), return_tensors='pt', padding=True)
        embeddings.to(device)
        labels = batch[1].to(device)
        preds = self.model(embeddings)
        loss = self.loss_function(preds, labels.float())   
        return loss
    
    def val_forward_pass(self, batch, device):
        embeddings = self.model.tokenizer(list(batch[0]), return_tensors='pt', padding=True)
        embeddings.to(device)
        labels = batch[1].to(device)
        preds = self.model(embeddings)
        loss = self.loss_function(preds, labels.float())
        return loss, preds.shape[0]
          
class NSDDataset(Dataset):
    def __init__(self, split, dataset_config=None):
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

        random.shuffle(dataset)            
        split_place = int(.8*len(dataset))
        
        if split == 'train':
            self.tokenized_data = dataset[:split_place]
        if split == 'val':
            self.tokenized_data = dataset[split_place:]


    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)