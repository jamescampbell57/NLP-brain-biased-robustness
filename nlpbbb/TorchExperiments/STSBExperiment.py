# hf imports
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import get_scheduler

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# nlpbbb imports
import nlpbbb as bbb
from nlpbbb.paths import PATHS

# random imports
import csv
import os

class Experiment():
    
    def __init__(self, config):
        self.train_datasets = [STSBDataset(ds, config["dataset"]) for ds in config["dataset"]["train_datasets"]]
        self.val_datasets = [STSBDataset(ds, config["dataset"]) for ds in config["dataset"]["val_datasets"]]
        
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
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["experiment"]["lr"])
        num_iters = sum([len(dl) for dl in self.train_loaders])
        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_iters)
        self.loss_function = torch.nn.MSELoss()
        self.lr_scheduler = None
        
    def get_model(self, model_config):
        return bbb.networks.STSBBERT(model_config)
        
    def train_forward_pass(self, batch, device):
        
        vec_1 = self.model(batch['sentence_1'])
        vec_2 = self.model(batch['sentence_2'])
        cosine_similarity_times_5 = self.cos(vec_1, vec_2) * 5
        targets = batch['labels'].float().to(device)
        loss = self.loss_function(cosine_similarity_times_5, targets) #replace .loss
        return loss
    
    def val_forward_pass(self, batch, device):
        vec_1 = self.model(batch['sentence_1'])
        vec_2 = self.model(batch['sentence_2'])
        cosine_similarity = self.cos(vec_1, vec_2)
        golds = batch['labels'].float()
        return cosine_similarity, golds

class STSBDataset(Dataset):
    def __init__(self, ds, dataset_config):
        import csv
        data_path = f'{PATHS["root"]}/data/stsb/stsbenchmark'
        if not os.path.exists(data_path):
            dataset_path = f'{PATHS["root"]}/data/stsb'
            os.system('mkdir '+dataset_path)
            os.system('wget https://data.deepai.org/Stsbenchmark.zip -P '+dataset_path)
            os.system(f'unzip {PATHS["root"]}/data/stsb/Stsbenchmark.zip -d {PATHS["root"]}/data/stsb/')
        

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
        
        if ds == "headlines":
            self.tokenized_data = headlines_dataset
        elif ds == "images":
            self.tokenized_data = images_dataset
        elif ds == "MSRpar":
            self.tokenized_data = MSRpar_dataset
        elif ds == "MSRvid":
            self.tokenized_data = MSRvid_dataset
        else:
            raise ValueError
            
        #To be clear, the data is not actually tokenized yet
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)
