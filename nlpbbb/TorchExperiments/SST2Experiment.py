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
        self.train_datasets = [SST2Datset(ds, "train", config["dataset"]) for ds in config["dataset"]["train_datasets"]]
        assert len(config["dataset"]["val_datasets"]) > 0, "Need val dataset"
        self.val_datasets = [SST2Datset(ds, "val", config["dataset"]) for ds in config["dataset"]["val_datasets"]]
        
        # handels two cases: you want to validate internally or using another experiment object
        self.train_loaders = [DataLoader(ds, batch_size=config["experiment"]["batchsize"], shuffle=True) for ds in self.train_datasets]
        self.val_loaders = [DataLoader(ds, batch_size=config["experiment"]["batchsize"], shuffle=False) for ds in self.val_datasets]
        
        # really you only want to build a model for an experiment object if it is the train experiment
        self.model = self.get_model(config["model"])
                                           
    def get_model(self, model_config):
        return bbb.networks.SST2BERT(model_config)
        
    def train_forward_pass(self, batch, loss_fn, device):
        batch = {k: v.to(device) for k, v in batch.items()}
        features = {k: v for k, v in batch.items() if k != 'labels'}
        preds = model(features)
        loss = loss_function(preds, batch['labels'].float()) #replace .loss
        return loss
    
    def val_forward_pass(self, batch, device):
        features = {k: v for k, v in batch.items() if k != 'labels'}
        preds = model(features)
        preds = torch.where(preds < .5, 0, 1)
        labels = batch['labels'].reshape(preds.shape)
        num_correct = (preds==labels).sum()
        num_samples = preds.size(0)
        return num_correct, num_samples

class SST2Datset(Dataset):
    def __init__(self, ds, split, dataset_config):
        
        if ds == "imdb":
            loaded_dset = load_dataset('imdb')
        elif ds == "sst2":
            loaded_dset = load_dataset('glue','sst2')
        else:
            raise ValueError("Subplit not found.")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #tokenize function
        def tokenize_dset(examples):
            if ds == "imdb":
                data = examples['text']
            else:
                data = examples['sentence']
            return tokenizer(data, padding="max_length", truncation=True)

        #pre-tokenize entire dataset
        tokenized_dset = loaded_dset.map(tokenize_dset, batched=True)
        
        removed_fields = ["text"] if ds == "imdb" else ["sentence","idx"]

        tokenized_dset = tokenized_dset.remove_columns(removed_fields)        
        tokenized_dset = tokenized_dset.rename_column("label", "labels")
        tokenized_dset.set_format("torch")
        self.tokenized_data = tokenized_dset[split].shuffle(seed=dataset_config["seed"]).select(range(dataset_config["limit"]))
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)