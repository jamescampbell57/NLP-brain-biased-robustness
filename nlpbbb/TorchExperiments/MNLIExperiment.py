# hf imports
from datasets import load_dataset
from transformers import AutoTokenizer

# torch imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# nlpbbb imports
import nlpbbb as bbb

# misc imports
import os
import sys
import csv

class Experiment():
    
    def __init__(self, config):
        self.train_datasets = [MNLIDataset(ds, config["dataset"]) for ds in config["dataset"]["train_datasets"]]
        self.val_datasets = [MNLIDataset(ds, config["dataset"]) for ds in config["dataset"]["val_datasets"]]
        
        # handels two cases: you want to validate internally or using another experiment object
        if len(self.train_datasets) == 1 and len(self.val_datasets) == 0:
            total_dset_size = len(self.train_datasets[0])
            train_size = int(0.8 * total_dset_size)
            test_size = total_dset_size - train_size
            training_data, test_data = torch.utils.data.random_split(self.train_datasets[0], [train_size, test_size])
            self.train_loaders = [DataLoader(training_data, batch_size=config["experiment"]["batchsize"], shuffle=True)]
            self.val_loaders = [DataLoader(test_data, batch_size=config["experiment"]["batchsize"], shuffle=False)]
        else:
            self.train_loaders = [DataLoader(ds, batch_size=config["experiment"]["batchsize"], shuffle=True) for ds in self.train_datasets]
            self.val_loaders = [DataLoader(ds, batch_size=config["experiment"]["batchsize"], shuffle=False) for ds in self.val_datasets]
        
        # really you only want to build a model for an experiment object if it is the train experiment
        self.model = self.get_model(config["model"])
                                           
    def get_model(self, model_config):
        return bbb.networks.MNLIBert(model_config)
        
    def train_forward_pass(self, batch, loss_fn, device):
        pred = model(batch['sentence_1'], batch['sentence_2'])
        targets = torch.stack(tuple(batch['labels'])).to(device)
        targets = torch.transpose(targets, 0, 1)
        loss = loss_fn(pred, targets.float())
        return loss
    
    def val_forward_pass(self, batch, device):
        pred = model(batch['sentence_1'], batch['sentence_2'])
        pred = torch.argmax(pred, axis=1)
        targets = torch.stack(tuple(batch['labels'])).to(device)
        targets = torch.transpose(targets, 0, 1)
        labels = torch.argmax(targets, axis=1)
        num_correct = (pred==labels).sum()
        num_samples = pred.size(0)
        return num_correct, num_samples

class MNLIDataset(Dataset):
    def __init__(self, ds, dataset_config):
        data_path = f'{dataset_config["root"]}/data/mnli/multinli_1.0/'
        maxInt = sys.maxsize
        #From stackoverflow
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt/10)

        def load_data(data_file):
            dataset = []
            with open(data_path+data_file) as file:
                tsv_file = csv.reader(file, delimiter="\t")
                for line in tsv_file:
                    dataset.append(line)
            return dataset
           
        #examples of dname: multinli_1.0_train.txt, multinli_1.0_dev_matched.txt, multinli_1.0_dev_mismatched.txt
        train_set = load_data('multinli_1.0_train.txt')
        dev_matched = load_data('multinli_1.0_dev_matched.txt')
        dev_mismatched = load_data('multinli_1.0_dev_mismatched.txt')
        #gather subdatasets
        telephone, letters, facetoface = split_data()
        
        if ds == "train_set":
            self.dataset = simplify_data(train_set)[1:]
        elif ds == "dev_matched":
            self.dataset = simplify_data(dev_matched)[1:]
        elif ds == "dev_mismatched":
            self.dataset = simplify_data(dev_mismatched)[1:]
        elif ds == "telephone":
            self.dataset = simplify_data(telephone)
        elif ds == "letters":
            self.dataset = simplify_data(letters)
        elif ds == "facetoface":
            self.dataset = simplify_data(facetoface)
        else:
            raise ValueError("Dataset not implemented")
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def split_data():
        telephone = []
        letters = []
        facetoface = []

        def extract(dataset):
            for ex in dataset:
                if ex[9] == 'telephone':
                    telephone.append(ex)
                if ex[9] == 'letters':
                    letters.append(ex)
                if ex[9] == 'facetoface':
                    facetoface.append(ex)

        extract(train_set)
        extract(dev_matched)
        extract(dev_mismatched)
        
        return telephone, letters, facetoface
    
    def simplify_data(dataset):
        simplified_dataset = []
        for item in dataset:
            i = 0
            example = {}
            example['sentence_1'] = item[5]
            example['sentence_2'] = item[6]
            if item[0] == 'entailment':
                example['labels'] = [0,0,1]
                i = 1
            if item[0] == 'neutral':
                example['labels'] = [0,1,0]
                i = 1
            if item[0] == 'contradiction':
                example['labels'] = [1,0,0]
                i =1
            if i == 1:
                simplified_dataset.append(example)
        return simplified_dataset