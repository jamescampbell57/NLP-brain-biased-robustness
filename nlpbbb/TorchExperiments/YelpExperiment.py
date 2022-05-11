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

class YelpDataset(Dataset):
    def __init__(self, dataset_config):
        data_path = f'{dataset_config["root"]}/data/yelp/'
        f1 = open(data_path+'yelp_academic_dataset_business.json') #150346
        f2 = open(data_path+'yelp_academic_dataset_review.json') #6990280

        business = []
        for line in f1:
            business.append(json.loads(line))

        review = []
        for line in f2:
            review.append(json.loads(line))

        f1.close()
        f2.close()
        
        american_business_ids = []
        japanese_business_ids = []
        chinese_business_ids = []
        italian_business_ids = []

        for example in business:
            if (not example['categories'] is None) and 'American' in example['categories']:
                american_business_ids.append(example['business_id'])
            if (not example['categories'] is None) and 'Japanese' in example['categories']:
                japanese_business_ids.append(example['business_id'])
            if (not example['categories'] is None) and 'Chinese' in example['categories']:
                chinese_business_ids.append(example['business_id'])
            if (not example['categories'] is None) and 'Italian' in example['categories']:
                italian_business_ids.append(example['business_id'])
        
        import time

        american = []
        japanese = []
        chinese = []
        italian = []

        start = time.time()

        for idx, example in enumerate(review):
            if example['business_id'] in american_business_ids:
                american.append(example)
            if example['business_id'] in japanese_business_ids:
                japanese.append(example)
            if example['business_id'] in chinese_business_ids:
                chinese.append(example)
            if example['business_id'] in italian_business_ids:
                italian.append(example)
            if idx%250000 == 0:
                print("Hello")


        with open('american.json', 'w') as f3:
            json.dump(american, f3)
        with open('japanese.json', 'w') as f4:
            json.dump(japanese, f4)
        with open('chinese.json', 'w') as f5:
            json.dump(chinese, f5)
        with open('italian.json', 'w') as f6:
            json.dump(italian, f6)
            
        import json
        data_path = '/home/ubuntu/NLP-brain-biased-robustness/data/yelp/'
        f1 = open(data_path+'american.json')
        f2 = open(data_path+'italian.json')
        f3 = open(data_path+'japanese.json')
        f4 = open(data_path+'chinese.json')

        american = []
        for line in f1:
            american.append(json.loads(line))

        italian = []
        for line in f2:
            italian.append(json.loads(line))

        japanese = []
        for line in f3:
            japanese.append(json.loads(line))

        chinese = []
        for line in f4:
            chinese.append(json.loads(line))

        f1.close()
        f2.close()
        f3.close()
        f4.close()


        american = american[0]
        italian = italian[0]
        japanese = japanese[0]
        chinese = chinese[0]
        
        na = []
        for i in american:
            na.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})

        ni = []
        for i in italian:
            ni.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})

        nj = []
        for i in japanese:
            nj.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})

        nc = []
        for i in chinese:
            nc.append({'text': i['text'], 'labels': F.one_hot((torch.tensor(i['stars']-1)).to(torch.int64), num_classes=5)})
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)