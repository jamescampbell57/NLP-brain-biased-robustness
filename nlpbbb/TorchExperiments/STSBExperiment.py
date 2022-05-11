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

class Experiment():
    
    def __init__(self, config):
        self.train_datasets = [AmazonDataset(ds, config["dataset"]) for ds in config["dataset"]["train_datasets"]]
        self.val_datasets = [AmazonDataset(ds, config["dataset"]) for ds in config["dataset"]["val_datasets"]]
        
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
        return bbb.networks.AmazonBERT(model_config)
        
    def train_forward_pass(self, batch, loss_fn, device):
        features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        preds = self.model(features)
        targets = F.one_hot((batch['labels']-1).to(torch.int64), num_classes=5).to(device)
        loss = loss_fn(preds, targets.float()) #replace .loss
        return loss
    
    def val_forward_pass(self, batch, device):
        features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        preds = self.model(features)
        preds = torch.argmax(preds, axis=1)
        labels = F.one_hot((batch['labels']-1).to(torch.int64), num_classes=5).to(device)
        labels = torch.argmax(labels, axis=1)
        num_correct = (preds==labels).sum()
        num_samples = preds.size(0)
        return num_correct, num_samples

class STSBDataset(Dataset):
    def __init__(self, dataset_config):
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
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)