import torch

class STSBDataset(Dataset):
    def __init__(self, dataset_config):
        import csv
        data_path = '/home/ubuntu/NLP-brain-biased-robustness/data/stsb/stsbenchmark/'

        #wget https://data.deepai.org/Stsbenchmark.zip

        def read_csv(csv_file):
            file = open(csv_file)
            csvreader = csv.reader(file, delimiter="\t")
            header = next(csvreader)
            rows = []
            for row in csvreader:
                rows.append(row)
            file.close()
            return rows
        
        train_set = read_csv(data_path+'sts-train.csv')
        dev_set = read_csv(data_path+'sts-dev.csv')
        test_set = read_csv(data_path+'sts-test.csv')
        
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
        images_dataset = create_dataset(images)
        MSRpar_dataset = create_dataset(MSRpar)
        MSRvid_dataset = create_dataset(MSRvid)
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)