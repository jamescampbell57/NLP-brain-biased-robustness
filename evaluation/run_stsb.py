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
import wandb



def single_run(batch_size, learning_rate):
    
    settings = f'bs: {batch_size}, lr: {learning_rate}'

    def read_csv(csv_file):
        file = open(csv_file)
        csvreader = csv.reader(file, delimiter="\t")
        header = next(csvreader)
        rows = []
        for row in csvreader:
            rows.append(row)
        file.close()
        return rows

    
    data_path = '/home/ubuntu/NLP-brain-biased-robustness/data/stsb/stsbenchmark'
    if not os.path.exists(data_path):
        dataset_path = '/home/ubuntu/NLP-brain-biased-robustness/data/stsb'
        os.system('mkdir '+dataset_path)
        os.system('wget https://data.deepai.org/Stsbenchmark.zip -P '+dataset_path)
        os.system(f'unzip ~/NLP-brain-biased-robustness/data/stsb/Stsbenchmark.zip -d ~/NLP-brain-biased-robustness/data/stsb/')

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
    headlines_train_dataset = headlines_dataset[:1500]
    headlines_val_dataset = headlines_dataset[1500:2000]
    images_dataset = create_dataset(images)[:500]
    MSRpar_dataset = create_dataset(MSRpar)[:500]
    MSRvid_dataset = create_dataset(MSRvid)[:500]

    headlines_train_dataloader = DataLoader(headlines_train_dataset, shuffle=True, batch_size=batch_size)
    headlines_val_dataloader = DataLoader(headlines_val_dataset, shuffle=False, batch_size=batch_size)
    images_dataloader = DataLoader(images_dataset, shuffle=False, batch_size=batch_size)
    MSRpar_dataloader = DataLoader(MSRpar_dataset, shuffle=False, batch_size=batch_size)
    MSRvid_dataloader = DataLoader(MSRvid_dataset, shuffle=False, batch_size=batch_size)



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
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        def forward(self, x):
            embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
            embeddings.to(self.device)
            representations = self.bert(**embeddings).last_hidden_state
            cls_representation = representations[:,0,:]
            return cls_representation


    def train(model, dataloader, num_epochs=10):
        run = wandb.init(project="brain biased hyperparameter search 100", entity="nlp-brain-biased-robustness", reinit=True)
        wandb.run.name = 'STS-b BERT '+settings
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

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        #training loop
        model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                #batch = {k: v.to(device) for k, v in batch.items()}
                #features = {k: v for k, v in batch.items() if k != 'labels'}
                vec_1 = model(batch['sentence_1'])
                vec_2 = model(batch['sentence_2'])
                cosine_similarity_times_5 = cos(vec_1, vec_2) * 5
                targets = batch['labels'].float().to(device)
                loss = loss_function(cosine_similarity_times_5, targets) 
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                wandb.log({"training loss": loss.item()})
                progress_bar.update(1)
            headlines_score = evaluate(model, headlines_val_dataloader)
            wandb.log({'headlines': headlines_score})
            images_score = evaluate(model, images_dataloader)
            wandb.log({'images': images_score})
            MSRpar_score = evaluate(model, MSRpar_dataloader)
            wandb.log({'MSRpar': MSRpar_score})
            MSRvid_score = evaluate(model, MSRvid_dataloader)
            wandb.log({'MSRvid': MSRvid_score})
        run.finish()

    def evaluate(model, dataloader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarities = []
        gold = []
        for batch in dataloader:
            with torch.no_grad():
                vec_1 = model(batch['sentence_1'])
                vec_2 = model(batch['sentence_2'])
                cosine_similarity = cos(vec_1, vec_2)
                golds = batch['labels'].float()
                for idx, similarity in enumerate(cosine_similarity):
                    cosine_similarities.append(similarity)
                    gold.append(golds[idx])
        torch_cosines = torch.tensor(cosine_similarities)
        torch_gold = torch.tensor(gold)

        torch_cosines = torch_cosines.reshape((1,torch_cosines.shape[0]))
        torch_gold = torch_gold.reshape((1,torch_gold.shape[0]))

        combined = torch.cat((torch_cosines, torch_gold), axis=0)

        return (torch.corrcoef(combined)[0,1]).item()


    model = PlaceHolderBERT(brain=True)
    train(model, headlines_train_dataloader)
    
    

    
for lr in [.0001, .00005]: 
    for bs in [8,16,1]:
        single_run(bs, lr)
        

