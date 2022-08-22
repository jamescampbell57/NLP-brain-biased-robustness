import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class AmazonDataset(Dataset):
    def __init__(self, split, seed=42, n_train = 20000, n_val=1000):
        colloquial_to_technical = {
            "train" : "Baby_v1_00",
            "baby_val" : "Baby_v1_00",
            "shoes_val" : "Shoes_v1_00",
            "clothes_val" : "Apparel_v1_00",
            "music_val" : "Music_v1_00",
            "video_val" : "Video_v1_00"
        }
        raw_dataset = load_dataset('amazon_us_reviews', colloquial_to_technical[split])
        if split == "train":
            raw_dataset = raw_dataset['train'].shuffle(seed=seed).select(range(50000,50000+n_train))
        else:
            raw_dataset = raw_dataset['train'].shuffle(seed=seed).select(range(n_val))

        self.dataset = []
        for item in raw_dataset:
            example = (item['review_body'], item['star_rating']-1) #class labels 0..4
            self.dataset.append(example)
        
    def __getitem__(self, idx):
        return self.dataset[idx]
        
    def __len__(self):
        return len(self.dataset)