import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
from datasets import load_dataset

from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import wandb


def single_run(batch_size, learning_rate):
    
    settings = f'bs: {batch_size}, lr: {learning_rate}'
    
    imdb = load_dataset('imdb')
    sst2 = load_dataset('glue','sst2')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    #tokenize function
    def tokenize_imdb(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    def tokenize_sst2(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)

    #pre-tokenize entire dataset
    tokenized_imdb = imdb.map(tokenize_imdb, batched=True)
    tokenized_sst2 = sst2.map(tokenize_sst2, batched=True)

    tokenized_imdb = tokenized_imdb.remove_columns(["text"])
    tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
    tokenized_imdb.set_format("torch")

    tokenized_sst2 = tokenized_sst2.remove_columns(["sentence","idx"])
    tokenized_sst2 = tokenized_sst2.rename_column("label", "labels")
    tokenized_sst2.set_format("torch")



    imdb_small_train = tokenized_imdb['train'].shuffle(seed=42).select(range(10000))
    imdb_small_test = tokenized_imdb['test'].shuffle(seed=42).select(range(10000))

    imdb_train_loader = DataLoader(imdb_small_train, shuffle=True, batch_size=8)
    imdb_test_loader = DataLoader(imdb_small_test, shuffle=True, batch_size=8)

    sst2_small_train = tokenized_sst2["train"].select(range(30000)).shuffle(seed=42).select(range(10000))
    sst2_small_test = tokenized_sst2["train"].select(range(30000,60000)).shuffle(seed=42).select(range(10000))

    sst2_train_loader = DataLoader(sst2_small_train, shuffle=True, batch_size=8)
    sst2_test_loader = DataLoader(sst2_small_test, shuffle=True, batch_size=8)



    def change_all_keys(pre_odict):
        def change_key(odict, old, new):
            for _ in range(len(odict)):
                k, v = odict.popitem(False)
                odict[new if old == k else k] = v
                return odict
        for key in pre_odict.keys():
            if key[:5] == 'bert.':
                post_odict = change_key(pre_odict, key, key[5:])
                return change_all_keys(post_odict)
            if key[:7] == 'linear.':
                del pre_odict[key]
                return change_all_keys(pre_odict)
        return pre_odict

    class PlaceHolderBERT(nn.Module):
        def __init__(self, num_out=1, brain=False):
            super().__init__()
            #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') 
            self.bert = BertModel.from_pretrained('bert-base-cased')
            if brain:
                state_path = '/home/ubuntu/NLP-brain-biased-robustness/state_dicts/cereberto_epoch_5'
                pre_odict = torch.load(state_path)
                filtered_odict = change_all_keys(pre_odict)
                self.bert.load_state_dict(filtered_odict, strict=True)
            self.linear = nn.Linear(768,num_out)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            #embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
            #embeddings.to(self.device)
            representations = self.bert(**x).last_hidden_state
            cls_representation = representations[:,0,:]
            pred = self.linear(cls_representation)
            return self.sigmoid(pred)


    def train(model, dataloader, num_epochs=10):
        run = wandb.init(project="brain biased hyperparameter search", entity="nlp-brain-biased-robustness", reinit=True)
        wandb.run.name = 'sst2/IMDb BERT '+settings
        wandb.config = {
          "learning_rate": learning_rate,
          "epochs": num_epochs,
          "batch_size": batch_size
        }

        #optimizer as usual
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.MSELoss()
        #learning rate scheduler
        num_training_steps = num_epochs * len(dataloader)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        #auto logging; progress bar
        progress_bar = tqdm(range(num_training_steps))

        #training loop
        model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                features = {k: v for k, v in batch.items() if k != 'labels'}
                preds = model(features)
                loss = loss_function(preds, batch['labels'].float())
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                wandb.log({"training loss": loss.item()})
                progress_bar.update(1)
            IID_score = evaluate(model, imdb_test_loader)
            OOD_score = evaluate(model, sst2_test_loader)
            wandb.log({"imdb": IID_score})
            wandb.log({"sst2": OOD_score})
        run.finish()

    def evaluate(model, dataloader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        num_correct = 0
        num_samples = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            features = {k: v for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                preds = model(features)
                preds = torch.where(preds < .5, 0, 1)
                labels = batch['labels'].reshape(preds.shape)
                num_correct += (preds==labels).sum()
                num_samples += preds.size(0)
        return float(num_correct)/float(num_samples)*100 

    model = PlaceHolderBERT(brain=True)
    train(model, imdb_train_loader)


for lr in [.00001]: 
    for bs in [8,16,1]:
        single_run(bs, lr)
