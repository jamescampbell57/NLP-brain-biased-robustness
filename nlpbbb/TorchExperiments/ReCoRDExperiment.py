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

class ReCoRDDataset(Dataset):
    def __init__(self, dataset_config):
        
        if dataset_config["dname"] == "imdb":
            loaded_dset = load_dataset('imdb')
        elif dataset_config["dname"] == "sst2":
            loaded_dset = load_dataset('glue','sst2')
        else:
            raise ValueError("Subsplit not found.")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #tokenize function
        def tokenize_dset(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True)

        def tokenize_sst2(examples):
            return tokenizer(examples['sentence'], padding="max_length", truncation=True)

        #pre-tokenize entire dataset
        tokenized_dset = imdb.map(tokenize_imdb, batched=True)
        tokenized_sst2 = sst2.map(tokenize_sst2, batched=True)

        tokenized_imdb = tokenized_imdb.remove_columns(["text"])
        tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
        tokenized_imdb.set_format("torch")

        tokenized_sst2 = tokenized_sst2.remove_columns(["sentence","idx"])
        tokenized_sst2 = tokenized_sst2.rename_column("label", "labels")
        tokenized_sst2.set_format("torch")


        ### Only for practive
        imdb_small_train = tokenized_imdb['train'].shuffle(seed=42).select(range(1000))
        imdb_small_test = tokenized_imdb['test'].shuffle(seed=42).select(range(500))
        ###
        imdb_train_loader = DataLoader(imdb_small_train, shuffle=True, batch_size=8)
        imdb_test_loader = DataLoader(imdb_small_test, shuffle=True, batch_size=8)

        sst2_small_train = tokenized_sst2["train"].shuffle(seed=42).select(range(1000))
        sst2_small_test = tokenized_sst2["validation"].shuffle(seed=42).select(range(500)) #actual test set is fucked up
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)