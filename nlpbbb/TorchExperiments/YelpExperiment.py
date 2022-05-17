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
import json

from nlpbbb.paths import PATHS

class Experiment():
    
    def __init__(self, config):
        self.train_datasets = [YelpDataset(ds, config["dataset"]) for ds in config["dataset"]["train_datasets"]]
        self.val_datasets = [YelpDataset(ds, config["dataset"]) for ds in config["dataset"]["val_datasets"]]
        
        self.val_loaders = []
        for index, ds in enumerate(self.val_datasets):
            if config["dataset"]["train_datasets"][0] == config["dataset"]["val_datasets"][index]:
                total_dset_size = len(self.train_datasets[0])
                train_size = int(0.8 * total_dset_size)
                test_size = total_dset_size - train_size
                training_data, test_data = torch.utils.data.random_split(self.train_datasets[0], [train_size, test_size])
                self.train_loaders = [DataLoader(training_data, batch_size=config["experiment"]["batchsize"], shuffle=True)]
                self.val_loaders.append(DataLoader(test_data, batch_size=config["experiment"]["batchsize"], shuffle=False))
            else:
                self.val_loaders.append(DataLoader(ds, batch_size=config["experiment"]["batchsize"], shuffle=False))
        
        # really you only want to build a model for an experiment object if it is the train experiment
        self.model = self.get_model(config["model"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["experiment"]["lr"])
        #num_iters = sum([len(dl) for dl in self.train_loaders])
        #self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_iters)
        self.loss_function = torch.nn.MSELoss()
        self.lr_scheduler = None
        
    def get_model(self, model_config):
        return bbb.networks.YelpBERT(model_config)
        
    def train_forward_pass(self, batch, device):
        #batch = {k: v.to(device) for k, v in batch.items()}
        #features = {k: v for k, v in batch.items() if k != 'labels'}
        preds = self.model(batch['text'])
        targets = batch['labels'].float().to(device)
        loss = self.loss_function(preds, targets) #replace .loss
        return loss
    
    def val_forward_pass(self, batch, device):
        #batch = {k: v.to(device) for k, v in batch.items()}
        #features = {k: v for k, v in batch.items() if k != 'labels'}
        preds = self.model(batch['text'])
        preds = torch.argmax(preds, axis=1)
        labels = torch.argmax(batch['labels'], axis=1).to(device)
        num_correct = (preds==labels).sum()
        num_samples = preds.size(0)
        return num_correct, num_samples

    
class YelpDataset(Dataset):
    def __init__(self, ds, dataset_config):
        data_path = f'{PATHS["root"]}/data/yelp'
        
        if not os.path.exists(os.path.join(data_path, 'italian.json')):
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

            american = []
            japanese = []
            chinese = []
            italian = []

            for idx, example in enumerate(review):
                if example['business_id'] in american_business_ids:
                    american.append(example)
                if example['business_id'] in japanese_business_ids:
                    japanese.append(example)
                if example['business_id'] in chinese_business_ids:
                    chinese.append(example)
                if example['business_id'] in italian_business_ids:
                    italian.append(example)

            with open('american.json', 'w') as f3:
                json.dump(american, f3)
            with open('japanese.json', 'w') as f4:
                json.dump(japanese, f4)
            with open('chinese.json', 'w') as f5:
                json.dump(chinese, f5)
            with open('italian.json', 'w') as f6:
                json.dump(italian, f6)
            
        lang_file = open(os.path.join(data_path, f'{ds}.json'))
        language = [json.loads(line) for line in lang_file] 
        lang_file.close()
        #language = language[0]
        na = []
        for i in language:
            na.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})
        self.tokenized_data = na
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)
