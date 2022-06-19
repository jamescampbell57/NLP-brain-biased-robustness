import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
import numpy as np
from scipy.io import loadmat
import os


class HarryPotterDataset(Dataset):
    def __init__(self, subject):

        data_path = '/home/ubuntu/nlp-brain-biased-robustness/data/harry_potter_brain'
        # install dataset
        if not os.path.exists(data_path):
            os.system('mkdir '+data_path)
            for i in range(1,9):
                os.system(f'wget http://www.cs.cmu.edu/~fmri/plosone/files/subject_{i}.mat -P '+data_path)

        harry_potter = loadmat(os.path.join(data_path, f'subject_{subject}.mat'))

        words = []
        for i in range(5176):
            word = harry_potter['words'][0][i][0][0][0][0]
            words.append(word)

        word_times = []
        for i in range(5176):
            word_time = harry_potter['words'][0][i][1][0][0]
            word_times.append(word_time)

        tr_times = []
        for i in range(1351):
            tr_time = harry_potter['time'][i,0]
            tr_times.append(tr_time)

        #dont_include_indices = []
        #for idx, tr_time in enumerate(tr_times):
        #    if not set(np.arange(tr_time - 10, tr_time, .5)).issubset(set(word_times)):
        #        dont_include_indices.append(idx)

        dont_include_indices = [i for i in range(15)] + [i for i in range(335,355)] + [i for i in range(687,707)] + [i for i in range(966,986)] + [i for i in range(1346,1351)]

        X_fmri = harry_potter['data']

        useful_X_fmri = np.delete(X_fmri, dont_include_indices,axis=0)

        #tr_times_arr = np.asarray(tr_times)

        useful_tr_times = np.delete(np.asarray(tr_times), dont_include_indices)

        sentences = [[]]*1271
        for idx, useful_tr_time in enumerate(useful_tr_times):
            sentence= []
            for word, word_time in zip(words,word_times):
                if useful_tr_time - 10 <= word_time <= useful_tr_time:
                    sentence.append(word)
            sentences[idx] = sentence   

        actual_sentences = ['']*1271
        for idx, sentence in enumerate(sentences):
            for word in sentence:
                if word != '+':
                    actual_sentences[idx] = actual_sentences[idx] + word + ' '


        fmri = torch.as_tensor(useful_X_fmri)
        truth_fmri = fmri[:5,:]


        self.fmri_data = []
        for i in range(1271):
            self.fmri_data.append((actual_sentences[i], fmri[i,:]))


    def __getitem__(self, idx):
        return self.fmri_data[idx]

    def __len__(self):
        return len(self.fmri_data)


dataset_1 = HarryPotterDataset(1)
n_rows = len(dataset_1)
train_dataset = dataset_1[:int(.8*n_rows)]
val_dataset = dataset_1[int(.9*n_rows):]

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)


class BrainBiasedBERT(nn.Module):
    def __init__(self, num_voxels=37913):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(768,num_voxels)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True)
        embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        pred_fmri = self.linear(cls_representation)
        return pred_fmri
    
    
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import wandb

run = wandb.init(project="fMRI pretraining", entity="nlp-brain-biased-robustness")
wandb.run.name = 'subject 1 harry potter'
wandb.config = {
  "learning_rate": 1e-5,
  "epochs": 40,
  "batch_size": 8
}

def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_function = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_losses = []
        for batch in dataloader:
            preds = model(list(batch[0]))
            labels = batch[1].to(device)
            test_loss = loss_function(preds, labels.float())
            test_losses.append(test_loss)

    return torch.mean(torch.as_tensor(test_losses)) 

    
def train(model, dataloader, num_epochs=40): 
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_function = torch.nn.MSELoss()
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            preds = model(list(batch[0]))
            labels = batch[1].to(device)
            loss = loss_function(preds, labels.float()) #replace .loss
            loss.backward()            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            wandb.log({"running loss": loss.item()})
            progress_bar.update(1)
        
        val_loss = evaluate(model, test_dataloader)
        wandb.log({"training loss": loss.item()})
        wandb.log({"val loss": val_loss})
        save_dir = '/home/ubuntu/NLP-brain-biased-robustness/state_dicts'
        torch.save(model.state_dict(), os.path.join(save_dir, f'cereberto_epoch_{epoch}'))
        
model = BrainBiasedBERT()
train(model, train_dataloader)
