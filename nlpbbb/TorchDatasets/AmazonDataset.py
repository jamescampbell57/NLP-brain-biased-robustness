import torch

class AmazonDataset(Dataset):
    def __init__(self, dataset_config):
        amazon_large = load_dataset('amazon_us_reviews', dataset_config["dname"])
        amazon_small = amazon_large['train'].shuffle(seed=dataset_config["seed"]).select(range(dataset_config["limit"]))
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #tokenize function
        def tokenize_data(examples):
            return tokenizer(examples['review_body'], padding="max_length", truncation=True)
        #pre-tokenize entire dataset
        delete_list = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date']
        
        tokenized_data = amazon_small.map(tokenize_data, batched=True).remove_columns(delete_list)
        self.tokenized_data = tokenized_data.rename_column("star_rating", "labels")
        self.tokenized_data.set_format("torch")
        
    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    
    def __len__(self):
        return len(self.tokenized_data)