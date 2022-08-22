from torch.utils.data import Dataset
import os
import csv
import random

class STSB(Dataset):
    def __init__(self, split, n_val=500, n_train=1500, data_dir='/home/ubuntu/NLP-brain-biased-robustness/data', train_split='headlines'):
        
        if not os.path.exists(f'{data_dir}/stsb'):
            os.system(f'mkdir {data_dir}/stsb')
        if not os.path.exists(f'{data_dir}/stsb/stsbenchmark'):
            os.system(f'mkdir {data_dir}/stsb/stsbenchmark')
            os.system(f'wget https://data.deepai.org/Stsbenchmark.zip -P {data_dir}/stsb')
            os.system(f'unzip {data_dir}/stsb/Stsbenchmark.zip -d {data_dir}/stsb/')

        data_path = f'{data_dir}/stsb/stsbenchmark'

        def read_csv(csv_file):
            file = open(csv_file)
            csvreader = csv.reader(file, delimiter="\t")
            header = next(csvreader)
            rows = []
            for row in csvreader:
                rows.append(row)
            file.close()
            return rows
        
        train_set = read_csv(os.path.join(data_path,'sts-train.csv'))
        dev_set = read_csv(os.path.join(data_path,'sts-dev.csv'))
        test_set = read_csv(os.path.join(data_path,'sts-test.csv'))
        
        self.unsimple_dataset = []
        self.split = split if split != 'train' else train_split #split is unchanged, self.split is defined
        
        for raw_dataset in [train_set, dev_set, test_set]:
            for i in range(len(raw_dataset)):
                if raw_dataset[i][1] == self.split:
                    self.unsimple_dataset.append(raw_dataset[i])
                    
        self.dataset = []
        self._simplify_dataset()
        del self.unsimple_dataset
        if split in ["MSRpar","MSRvid","headlines","images"]: #assume 'headlines' split is for ID validation
            assert n_val <= len(self.dataset), "n_val > size of dataset"
            self._shrink_dataset(0, n_val)
        elif split == "train":
            assert n_val+n_train <= len(self.dataset), "n_val+n_train > size of dataset"
            self._shrink_dataset(n_val, n_val+n_train)
        
    def _simplify_dataset(self):
        for example in self.unsimple_dataset:
            if not len(example) < 7:
                data = {}
                data['sentence_1'] = example[5]
                data['sentence_2'] = example[6]
                data['labels'] = float(example[4])
                self.dataset.append(data)
                
    def _shrink_dataset(self, n_one, n_two, seed=42):
        random.seed(seed)
        random.shuffle(self.dataset) #Do NOT call twice for one instance
        self.dataset = self.dataset[n_one:n_two]
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
