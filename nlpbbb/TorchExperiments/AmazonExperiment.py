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
          
class AmazonDataset(Dataset):
    def __init__(self, ds, dataset_config):
        amazon_large = load_dataset('amazon_us_reviews', ds)
        amazon_small = amazon_large['train'].shuffle(seed=dataset_config["seed"]).select(range(dataset_config["limit"]))
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #tokenize function
        def tokenize_data(examples):
            return tokenizer(examples['review_body'], padding="max_length", truncation=True)
        #pre-tokenize entire dataset
        delete_list = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date']
        
        tokenized_data = amazon_small.map(tokenize_data, batched=True).remove_columns(delete_list)
        self.tokenized_data = tokenized_data.rename_column("star_rating", "labels")
        self.tokenized_data.set_format("torch")
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)