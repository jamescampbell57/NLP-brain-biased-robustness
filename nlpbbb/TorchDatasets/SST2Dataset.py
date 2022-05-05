import torch

class SST2Datset(Dataset):
    def __init__(self, split, dataset_config):
        
        if dataset_config["dname"] == "imdb":
            loaded_dset = load_dataset('imdb')
        elif dataset_config["dname"] == "sst2":
            loaded_dset = load_dataset('glue','sst2')
        else:
            raise ValueError("Subplit not found.")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #tokenize function
        def tokenize_dset(examples):
            if dataset_config["dname"] == "imdb":
                data = examples['text']
            else:
                data = examples['sentence']
            return tokenizer(data, padding="max_length", truncation=True)

        #pre-tokenize entire dataset
        tokenized_dset = loaded_dset.map(tokenize_dset, batched=True)
        
        removed_fields = ["text"] if dataset_config["dname"] == "imdb" else ["sentence","idx"]

        tokenized_dset = tokenized_dset.remove_columns(removed_fields)        
        tokenized_dset = tokenized_dset.rename_column("label", "labels")
        tokenized_dset.set_format("torch")

        self.tokenized_data = tokenized_dset[split].shuffle(seed=dataset_config["seed"]).select(range(dataset_config["limit"]))
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)