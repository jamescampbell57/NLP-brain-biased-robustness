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


class Experiment():
    
    def __init__(self, config):
        self.train_dataset = HarryPotterDataset(split="train", dataset_config=config["dataset"])
        self.val_dataset = HarryPotterDataset(split="val", dataset_config=config["dataset"])
        
        # handles two cases: you want to validate internally or using another experiment object (dont do this)
        self.train_loaders = [DataLoader(self.train_dataset, batch_size=config["experiment"]["batchsize"], shuffle=True)]
        self.val_loaders = [DataLoader(self.val_dataset, batch_size=config["experiment"]["batchsize"], shuffle=False)]
        
        # really you only want to build a model for an experiment object if it is the train experiment
        self.model = self.get_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["experiment"]["lr"])
        #num_iters = sum([len(dl) for dl in self.train_loaders])
        #self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_iters)
        self.lr_scheduler = None
        self.loss_function = torch.nn.MSELoss()
        
                                           
    def get_model(self):
        return bbb.networks.BrainBiasedBERT()
        
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
          
class HarryPotterDataset(Dataset):
    def __init__(self, split, dataset_config):
        subs_to_use = dataset_config["subjects"]
        
        dataset = []
        
        for subj in subs_to_use:
            harry_potter = loadmat(f'{PATHS["root"]}/data/harry_potter_brain/subject_{subj}.mat')

            words = []
            word_times = []
            num_words = harry_potter['words'].shape[1]
            for i in range(num_words):
                word_obj = harry_potter['words'][0][i]
                #super weird format
                word = word_obj[0][0][0][0]
                word_time = word_obj[1][0][0]
                words.append(word)
                word_times.append(word_time)

            tr_times = []
            time_length = harry_potter['time'].shape[0]
            for i in range(time_length):
                tr_time = harry_potter['time'][i,0]
                tr_times.append(tr_time)

            # hopefully generalizes across subjects
            dont_include_indices = [i for i in range(15)] + [i for i in range(335,355)] + [i for i in range(687,707)] + [i for i in range(966,986)] + [i for i in range(1346,1351)]

            X_fmri = harry_potter['data']
            useful_X_fmri = np.delete(X_fmri, dont_include_indices,axis=0)
            tr_times_arr = np.asarray(tr_times)
            useful_tr_times = np.delete(tr_times_arr, dont_include_indices)

            sentences = [[]]*1271
            for idx, useful_tr_time in enumerate(useful_tr_times):
                sentence= []
                for word, word_time in zip(words, word_times):
                    if useful_tr_time - 10 <= word_time <= useful_tr_time:
                        sentence.append(word)
                sentences[idx] = sentence   

            actual_sentences = ['']*1271
            for idx, sentence in enumerate(sentences):
                for word in sentence:
                    actual_sentences[idx] = actual_sentences[idx] + word + ' '

            fmri = torch.as_tensor(useful_X_fmri)

            for i in range(1271):
                dataset.append((actual_sentences[i], fmri[i,:]))

        #TRAIN TEST SPLIT HAS OVERLAP IN WORDS AND IN BRAIN STATE
        n_rows = len(dataset)
        if split == "train":
            self.tokenized_data = dataset[:int(.7*n_rows)]
        elif split == "val":
            self.tokenized_data = dataset[int(.8*n_rows):]
        else:
            raise ValueError("Split not recognized")
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)