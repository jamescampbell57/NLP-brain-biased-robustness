import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
from datasets import load_dataset

from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
import csv
import sys
import wandb



def single_run(batch_size, learning_rate):

    settings = f'bs: {batch_size}, lr: {learning_rate}'
    
    ################################################################

    dataset_path = '/home/ubuntu/NLP-brain-biased-robustness/data/mnli'
    data_path = dataset_path+'/multinli_1.0'
    if not os.path.exists(data_path):
        os.system('mkdir '+dataset_path)
        os.system('wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip -P '+dataset_path)
        os.system(f'unzip ~/NLP-brain-biased-robustness/data/mnli/multinli_1.0.zip -d ~/NLP-brain-biased-robustness/data/mnli/')

    ##################################################################

    data_path = '/home/ubuntu/NLP-brain-biased-robustness/data/mnli/multinli_1.0/'

    maxInt = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    def load_data(data_file):
        dataset = []
        with open(data_path+data_file) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                dataset.append(line)
        return dataset

    train_set = load_data('multinli_1.0_train.txt')
    dev_matched = load_data('multinli_1.0_dev_matched.txt')
    dev_mismatched = load_data('multinli_1.0_dev_mismatched.txt')

    #####################################################################

    def split_data():
        telephone = []
        letters = []
        facetoface = []

        def extract(dataset):
            for ex in dataset:
                if ex[9] == 'telephone':
                    telephone.append(ex)
                if ex[9] == 'letters':
                    letters.append(ex)
                if ex[9] == 'facetoface':
                    facetoface.append(ex)

        extract(train_set)
        extract(dev_matched)
        extract(dev_mismatched)
        return telephone, letters, facetoface

    telephone, letters, facetoface = split_data()

    def simplify_data(dataset):
        simplified_dataset = []
        for item in dataset:
            i = 0
            example = {}
            example['sentence_1'] = item[5]
            example['sentence_2'] = item[6]
            if item[0] == 'entailment':
                example['labels'] = [0,0,1]
                i = 1
            if item[0] == 'neutral':
                example['labels'] = [0,1,0]
                i = 1
            if item[0] == 'contradiction':
                example['labels'] = [1,0,0]
                i =1
            if i == 1:
                simplified_dataset.append(example)
        return simplified_dataset

    train_set = simplify_data(train_set)[1:]
    dev_matched = simplify_data(dev_matched)[1:]
    dev_mismatched = simplify_data(dev_mismatched)[1:]

    telephone = simplify_data(telephone)
    letters = simplify_data(letters)
    facetoface = simplify_data(facetoface)


    telephone_dataset = []
    for data_point in telephone:
        new_data_point = {}
        new_sentence = data_point['sentence_1']+'. '+data_point['sentence_2']
        new_data_point['sentence'] = new_sentence
        new_data_point['labels'] = data_point['labels']
        telephone_dataset.append(new_data_point)

    letters_dataset = []
    for data_point in letters:
        new_data_point = {}
        new_sentence = data_point['sentence_1']+'. '+data_point['sentence_2']
        new_data_point['sentence'] = new_sentence
        new_data_point['labels'] = data_point['labels']
        letters_dataset.append(new_data_point)

    facetoface_dataset = []
    for data_point in facetoface:
        new_data_point = {}
        new_sentence = data_point['sentence_1']+'. '+data_point['sentence_2']
        new_data_point['sentence'] = new_sentence
        new_data_point['labels'] = data_point['labels']
        facetoface_dataset.append(new_data_point)



    telephone_train_dataloader = DataLoader(telephone_dataset[:15000], shuffle=True, batch_size=batch_size) 
    telephone_val_dataloader = DataLoader(telephone_dataset[19000:20900], shuffle=True, batch_size=batch_size)
    letters_dataloader = DataLoader(letters_dataset[:1900], shuffle=True, batch_size=batch_size) #1977
    facetoface_dataloader = DataLoader(facetoface_dataset[:1900], shuffle=True, batch_size=batch_size) #1974


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
        def __init__(self, brain=False):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            self.bert = BertModel.from_pretrained('bert-base-cased')
            if brain:
                state_path = '/home/ubuntu/NLP-brain-biased-robustness/state_dicts/cereberto_HP_epoch_100'
                pre_odict = torch.load(state_path)
                filtered_odict = change_all_keys(pre_odict)
                self.bert.load_state_dict(filtered_odict, strict=True)
            self.linear = nn.Linear(768, 3)
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x):
            x_embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
            x_embeddings.to(self.device)
            x_representations = self.bert(**x_embeddings).last_hidden_state
            x_cls_representation = x_representations[:,0,:]
            pred = self.linear(x_cls_representation)
            return self.softmax(pred)


    def train(model, dataloader, num_epochs=10):
        run = wandb.init(project="brain biased hyperparameter search 100", entity="nlp-brain-biased-robustness", reinit=True)
        wandb.run.name = 'MNLI BERT '+settings
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
            for batch in dataloader: #tryin unpacking text from 'labels' as in model development
                #batch = {k: v.to(device) for k, v in batch.items()}
                #features = {k: v for k, v in batch.items() if k != 'labels'}
                pred = model(batch['sentence'])
                targets = torch.stack(tuple(batch['labels'])).to(device)
                targets = torch.transpose(targets, 0, 1)
                loss = loss_function(pred, targets.float())
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                wandb.log({"training loss": loss.item()})
                progress_bar.update(1)
            telephone_score = evaluate(model, telephone_val_dataloader)
            wandb.log({"telephone": telephone_score})
            letters_score = evaluate(model, letters_dataloader)
            wandb.log({"letters": letters_score})
            facetoface_score = evaluate(model, facetoface_dataloader)
            wandb.log({"facetoface": facetoface_score})
        run.finish()

    def evaluate(model, dataloader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        num_correct = 0
        num_samples = 0
        for batch in dataloader:
            with torch.no_grad():
                pred = model(batch['sentence'])
                pred = torch.argmax(pred, axis=1)
                targets = torch.stack(tuple(batch['labels'])).to(device)
                targets = torch.transpose(targets, 0, 1)
                labels = torch.argmax(targets, axis=1)
                num_correct += (pred==labels).sum()
                num_samples += pred.size(0)
        return float(num_correct)/float(num_samples)*100 


    model = PlaceHolderBERT(brain=True)
    train(model, telephone_train_dataloader)


for lr in [.0001,.00005, .00001]:
    for bs in [8,16,1]:
        single_run(bs, lr)


    
