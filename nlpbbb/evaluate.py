#!/usr/bin/env python
# coding: utf-8


def evaluate(model, dataloader, task):
    if task == 'sst2_imdb':
        score = evaluate_sst2_imdb(model, dataloader)
    if task == 'mnli':
        score = evaluate_sst2_imdb(model, dataloader)
    if task == 'amazon':
        score = evaluate_sst2_imdb(model, dataloader)
    if task == 'yelp':
        score = evaluate_sst2_imdb(model, dataloader)
    if task == 'stsb':
        score = evaluate_sst2_imdb(model, dataloader)
    if task == 'record':
        score = evaluate_record(model, dataloader)
    return score

#def evaluate_record
# TODO
#
#
#
#
#
#
#
#
#
#



def evaluate_sst2_imdb(model, dataloader):
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




def evaluate_mnli(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_samples = 0
    for batch in dataloader:
        with torch.no_grad():
            pred = model(batch['sentence_1'], batch['sentence_2'])
            pred = torch.argmax(pred, axis=1)
            targets = torch.stack(tuple(batch['labels'])).to(device)
            targets = torch.transpose(targets, 0, 1)
            labels = torch.argmax(targets, axis=1)
            num_correct += (pred==labels).sum()
            num_samples += pred.size(0)
    return float(num_correct)/float(num_samples)*100 


def evaluate_amazon(model, dataloader):
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


def evaluate_yelp(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_samples = 0
    for batch in dataloader:
        #batch = {k: v.to(device) for k, v in batch.items()}
        #features = {k: v for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            preds = model(batch['text'])
            preds = torch.argmax(preds, axis=1)
            labels = torch.argmax(batch['labels'], axis=1).to(device)
            num_correct += (preds==labels).sum()
            num_samples += preds.size(0)
    return float(num_correct)/float(num_samples)*100 





def evaluate_stsb(model, dataloader):
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
    
    return torch.corrcoef(combined)[1,1]

