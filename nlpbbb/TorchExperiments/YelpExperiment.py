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

from nlpbbb.paths import PATHS

class Experiment():
    
    def __init__(self, config):
        self.train_datasets = [YelpDataset(ds, config["dataset"]) for ds in config["dataset"]["train_datasets"]]
        self.val_datasets = [YelpDataset(ds, config["dataset"]) for ds in config["dataset"]["val_datasets"]]
        
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

    
class YelpDataset(Dataset):
    def __init__(self, ds, dataset_config):
        data_path = f'{PATHS["root"]}/data/yelp'
        f1 = open(os.path.join(data_path,'yelp_academic_dataset_business.json')) #150346
        f2 = open(os.path.join(data_path,'yelp_academic_dataset_review.json')) #6990280

        business = []
        for line in f1:
            business.append(json.loads(line))

        review = []
        for line in f2:
            review.append(json.loads(line))

        f1.close()
        f2.close()
        
        american_business_ids = []
        japanese_business_ids = []
        chinese_business_ids = []
        italian_business_ids = []

        for example in business:
            if (not example['categories'] is None) and 'American' in example['categories']:
                american_business_ids.append(example['business_id'])
            if (not example['categories'] is None) and 'Japanese' in example['categories']:
                japanese_business_ids.append(example['business_id'])
            if (not example['categories'] is None) and 'Chinese' in example['categories']:
                chinese_business_ids.append(example['business_id'])
            if (not example['categories'] is None) and 'Italian' in example['categories']:
                italian_business_ids.append(example['business_id'])
        
        import time

        american = []
        japanese = []
        chinese = []
        italian = []

        start = time.time()

        for idx, example in enumerate(review):
            if example['business_id'] in american_business_ids:
                american.append(example)
            if example['business_id'] in japanese_business_ids:
                japanese.append(example)
            if example['business_id'] in chinese_business_ids:
                chinese.append(example)
            if example['business_id'] in italian_business_ids:
                italian.append(example)
            if idx%250000 == 0:
                print("Hello")


        with open('american.json', 'w') as f3:
            json.dump(american, f3)
        with open('japanese.json', 'w') as f4:
            json.dump(japanese, f4)
        with open('chinese.json', 'w') as f5:
            json.dump(chinese, f5)
        with open('italian.json', 'w') as f6:
            json.dump(italian, f6)
            
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
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)