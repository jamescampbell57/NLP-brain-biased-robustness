import torch
import csv
import sys

class MNLIDataset(Dataset):
    def __init__(self, dataset_config):
        data_path = f'{dataset_config["root"]}/data/mnli/multinli_1.0/'
        maxInt = sys.maxsize
        #From stackoverflow
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
           
        #examples of dname: multinli_1.0_train.txt, multinli_1.0_dev_matched.txt, multinli_1.0_dev_mismatched.txt
        train_set = load_data('multinli_1.0_train.txt')
        dev_matched = load_data('multinli_1.0_dev_matched.txt')
        dev_mismatched = load_data('multinli_1.0_dev_mismatched.txt')
        #gather subdatasets
        telephone, letters, facetoface = split_data()
        
        if dataset_config["dname"] == "train_set":
            self.dataset = simplify_data(train_set)[1:]
        elif dataset_config["dname"] == "dev_matched":
            self.dataset = simplify_data(dev_matched)[1:]
        elif dataset_config["dname"] == "dev_mismatched":
            self.dataset = simplify_data(dev_mismatched)[1:]
        elif dataset_config["dname"] == "telephone":
            self.dataset = simplify_data(telephone)
        elif dataset_config["dname"] == "letters":
            self.dataset = simplify_data(letters)
        elif dataset_config["dname"] == "facetoface":
            self.dataset = simplify_data(facetoface)
        else:
            raise ValueError("Dataset not implemented")
        
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)
    
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