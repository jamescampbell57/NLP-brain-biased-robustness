import torch
from torch.utils.data import Dataset
import os
import sys
import csv

class MnliDataset(Dataset):
    def __init__(self, split, data_dir='/home/ubuntu/eNTK-robustness/data'):
        #supported splits include: "train", "mismatched", "matched", "telephone", "letters", "facetoface"
        #need to add size specifications
        if not os.path.exists(data_dir):
            os.system(f'mkdir {data_dir}')
        if not os.path.exists(f'{data_dir}/mnli'):
            os.system(f'mkdir {data_dir}/mnli')
        if not os.path.exists(f'{data_dir}/mnli/multinli_1.0'):
            os.system(f'wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip -P {data_dir}/mnli')
            os.system(f'unzip {data_dir}/mnli/multinli_1.0.zip -d {data_dir}/mnli/')

        data_path = f'{data_dir}/mnli/multinli_1.0/'

        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt/10)

        def _load_data(data_file):
            dataset = []
            with open(data_path+data_file) as file:
                tsv_file = csv.reader(file, delimiter="\t")
                for line in tsv_file:
                    dataset.append(line)
            return dataset
            
        if split == "train":
            self.dataset = _load_data(f'multinli_1.0_{split}.txt')
        elif split in ["matched", "mismatched"]:
            self.dataset = _load_data(f'multinli_1.0_dev_{split}.txt')

        def _split_data(sub_split):
            sub_data = []
            def _extract(super_split):
                super_data = _load_data(f'multinli_1.0_{super_split}.txt')
                for ex in super_data:
                    if ex[9] == sub_split:
                        sub_data.append(ex)
            _extract('train')
            _extract('dev_matched') #watch the 'dev'
            _extract('dev_mismatched')
            return sub_data
        
        if split in ["telephone", "letters", "facetoface"]:
            self.dataset = _split_data(split)
        
        self.dataset = self._simplify_data()[1:]
        self.dataset = self._reformat_data()
       
        
    def _simplify_data(self):
        simplified_dataset = []
        for item in self.dataset:
            i = 0
            example = {}
            example['sentence_1'] = item[5]
            example['sentence_2'] = item[6]
            if item[0] == 'entailment':
                example['labels'] = 0
                i = 1
            if item[0] == 'neutral':
                example['labels'] = 1
                i = 1
            if item[0] == 'contradiction':
                example['labels'] = 2
                i = 1
            if i == 1:
                simplified_dataset.append(example)
        return simplified_dataset

    def _reformat_data(self):
        new_dataset = []
        for data_point in self.dataset:
            if data_point['sentence_1'][-1] == '.':
                new_sentence = data_point['sentence_1']+' '+data_point['sentence_2']
            else:
                new_sentence = data_point['sentence_1']+'. '+data_point['sentence_2']
                
            new_data_point = (new_sentence, data_point['labels'])
            new_dataset.append(new_data_point)
        return new_dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]