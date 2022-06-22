import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
import datasets
from datasets import load_dataset

from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import torch.nn.functional as F
import wandb


def single_run(batch_size, learning_rate):
    
    settings = f'bs: {batch_size}, lr: {learning_rate}'
    
    amazon_baby = load_dataset('amazon_us_reviews','Baby_v1_00')
    amazon_shoes = load_dataset('amazon_us_reviews','Shoes_v1_00')
    amazon_clothes = load_dataset('amazon_us_reviews','Apparel_v1_00')
    amazon_music = load_dataset('amazon_us_reviews','Music_v1_00')
    amazon_video = load_dataset('amazon_us_reviews','Video_v1_00')

    baby_small = amazon_baby['train'].select(range(200000, len(amazon_baby['train']))).shuffle(seed=42).select(range(10000))
    baby_train = amazon_baby['train'].select(range(200000)).shuffle(seed=42).select(range(10000))
    shoes_small = amazon_shoes['train'].shuffle(seed=42).select(range(10000))
    clothes_small = amazon_clothes['train'].shuffle(seed=42).select(range(10000))
    music_small = amazon_music['train'].shuffle(seed=42).select(range(10000))
    video_small = amazon_video['train'].shuffle(seed=42).select(range(10000))

    ###############################################################

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_data(examples):
        return tokenizer(examples['review_body'], padding="max_length", truncation=True) #CAN PLAY WITH TOKENIZER

    tokenized_baby = baby_small.map(tokenize_data, batched=True) #WHAT IS BATCHED
    tokenized_baby_train = baby_train.map(tokenize_data, batched=True)
    tokenized_shoes = shoes_small.map(tokenize_data, batched=True)
    tokenized_clothes = clothes_small.map(tokenize_data, batched=True)
    tokenized_music = music_small.map(tokenize_data, batched=True)
    tokenized_video = video_small.map(tokenize_data, batched=True)

    delete_list = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date']
    tokenized_baby = tokenized_baby.remove_columns(delete_list)
    tokenized_baby = tokenized_baby.rename_column("star_rating", "labels")
    tokenized_baby.set_format("torch")

    tokenized_baby_train = tokenized_baby_train.remove_columns(delete_list)
    tokenized_baby_train = tokenized_baby_train.rename_column("star_rating", "labels")
    tokenized_baby_train.set_format("torch")

    tokenized_shoes = tokenized_shoes.remove_columns(delete_list)
    tokenized_shoes = tokenized_shoes.rename_column("star_rating", "labels")
    tokenized_shoes.set_format("torch")

    tokenized_clothes = tokenized_clothes.remove_columns(delete_list)
    tokenized_clothes = tokenized_clothes.rename_column("star_rating", "labels")
    tokenized_clothes.set_format("torch")

    tokenized_music = tokenized_music.remove_columns(delete_list)
    tokenized_music = tokenized_music.rename_column("star_rating", "labels")
    tokenized_music.set_format("torch")

    tokenized_video = tokenized_video.remove_columns(delete_list)
    tokenized_video = tokenized_video.rename_column("star_rating", "labels")
    tokenized_video.set_format("torch")


    baby_dataloader = DataLoader(tokenized_baby, shuffle=False, batch_size=batch_size)
    baby_train_dataloader = DataLoader(tokenized_baby_train, shuffle=True, batch_size=batch_size)
    shoes_dataloader = DataLoader(tokenized_shoes, shuffle=False, batch_size=batch_size)
    clothes_dataloader = DataLoader(tokenized_clothes, shuffle=False, batch_size=batch_size)
    music_dataloader = DataLoader(tokenized_music, shuffle=False, batch_size=batch_size)
    video_dataloader = DataLoader(tokenized_video, shuffle=False, batch_size=batch_size)

    ########################################################################################***

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
        def __init__(self, num_out=5, sigmoid=False, return_CLS_representation=False, brain=False):
            super().__init__()
            #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            self.bert = BertModel.from_pretrained('bert-base-cased')
            if brain:
                state_path = '/home/ubuntu/NLP-brain-biased-robustness/state_dicts/cereberto_epoch_5'
                pre_odict = torch.load(state_path)
                filtered_odict = change_all_keys(pre_odict)
                self.bert.load_state_dict(filtered_odict, strict=True)
            self.linear = nn.Linear(768,num_out)
            self.return_CLS_representation = return_CLS_representation
            self.sigmoid_bool = sigmoid
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x):
            #embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
            #embeddings.to(device)
            representations = self.bert(**x).last_hidden_state
            cls_representation = representations[:,0,:]
            pred = self.linear(cls_representation)
            if self.return_CLS_representation:
                return cls_representation
            if self.sigmoid_bool:
                return self.sigmoid(pred)
            return self.softmax(pred)


    def train(model, dataloader, num_epochs=10, settings=settings): #can scrap keyword
        run = wandb.init(project="brain biased hyperparameter search", entity="nlp-brain-biased-robustness", reinit=True)
        wandb.run.name = 'Amazon BERT '+settings
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
                features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                preds = model(features)
                targets = F.one_hot((batch['labels']-1).to(torch.int64), num_classes=5).to(device)
                loss = loss_function(preds, targets.float())
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                wandb.log({"training loss": loss.item()})
                progress_bar.update(1)
            baby_score = evaluate(model, baby_dataloader)
            wandb.log({"baby": baby_score})
            shoes_score = evaluate(model, shoes_dataloader)
            wandb.log({"shoes": shoes_score})
            clothes_score = evaluate(model, clothes_dataloader)
            wandb.log({"clothes": clothes_score})
            music_score = evaluate(model, music_dataloader)
            wandb.log({"music": music_score})
            video_score = evaluate(model, video_dataloader)
            wandb.log({"video": video_score})
        run.finish()

    def evaluate(model, dataloader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        num_correct = 0
        num_samples = 0
        for batch in dataloader:
            #batch = {k: v.to(device) for k, v in batch.items()}
            features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                preds = model(features)
                preds = torch.argmax(preds, axis=1)
                labels = F.one_hot((batch['labels']-1).to(torch.int64), num_classes=5).to(device)
                labels = torch.argmax(labels, axis=1)
                num_correct += (preds==labels).sum()
                num_samples += preds.size(0)
        return float(num_correct)/float(num_samples)*100 

    model = PlaceHolderBERT(brain=True)
    train(model, baby_train_dataloader)

    
    
    
for lr in [.00001]: 
    for bs in [8,16,1]:
        single_run(bs, lr)