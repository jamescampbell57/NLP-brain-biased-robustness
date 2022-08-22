import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
import random

class SST2_IMDB(Dataset):
    def __init__(self, split, n_train=40000, n_val=10000):
        if split in ["train","imdb_train","imdb_val"]:
            self.raw_dataset = load_dataset('imdb')
            self.dataset = []
            self._reformat_data('train', 'text', 'label')
            self._reformat_data('test', 'text', 'label')
            del self.raw_dataset
        elif split in ["sst2_val","sst2_train"]:
            self.raw_dataset = load_dataset('glue','sst2')
            self.dataset = []
            self._reformat_data('train', 'sentence', 'label')
            del self.raw_dataset
        else:
            raise ValueError(f'Split "{split}" is unsupported. Acceptable splits include "train", "imdb_val", "sst2_val", "sst2_train", "imdb_train"')
        
        if split in ["imdb_val","sst2_val"]:
            assert n_val <= len(self.dataset), "n_val > size of dataset"
            self._shrink_dataset(0, n_val)
        elif split in ["train","imdb_train","sst2_train"]:
            assert n_val+n_train <= len(self.dataset), "n_val+n_train > size of dataset"
            self._shrink_dataset(n_val, n_val+n_train)
            
    def _shrink_dataset(self, n_one, n_two, seed=42):
        random.seed(seed)
        random.shuffle(self.dataset) #never call this twice on the same instance
        self.dataset = self.dataset[n_one:n_two]
            
    def _reformat_data(self, sub_split, key_one, key_two):
        for i in range(len(self.raw_dataset[sub_split])):
            data_point = self.raw_dataset[sub_split][i]
            self.dataset.append((data_point[key_one], data_point[key_two]))
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)