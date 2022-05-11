# torch imports
import torch
import torch.nn.functional as F

# hf imports
from transformers import get_scheduler

# nlpbbb imports
import nlpbbb as bbb

# random imports 
from tqdm import tqdm
import wandb
from datetime import date as dt


def run_training_config(config):
    #get the date for run saving
    date = dt.today().strftime("%m.%d.%y")
    #First define an experiment object
    if config["misc"]["save"]:
        assert not (config["experiment"]["name"] is None), "Must have a name for the run."
        wandb.init(project="BrainFinetuning", entity="nlp-brain-biased-robustness")
        wandb.run.name = config["experiment"]["name"]
    
    exp = bbb.setup.get_experiment(config)
    model = exp.model
    
    num_epochs = config["experiment"]["epochs"]
    #Then set your optimizer/scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_iters = sum([len(dl) for dl in exp.train_loaders])
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * num_iters)
    
    #Extra details
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    #Training loops
    for epoch in range(num_epochs):
        #Run validation every so often, good to do before training
        if epoch % config["experiment"]["val_frequency"] == 0:
            val_loss = val_loop(config["experiment"], exp, epoch, exp.val_loaders, device)
            
        train_loss = train_loop(config["experiment"], exp, epoch, exp.train_loaders, optimizer, lr_scheduler, loss_fn, device)
        
        #save whenever you validate
        if epoch % config["experiment"]["val_frequency"] == 0:
            if config["misc"]["save"]:
                wandb.log({"train_loss": train_loss,
                           "val_loss": val_loss})
                bbb.utils.save_model(exp, optimizer, val_loss, config, date, epoch)

        
def train_loop(train_config, exp, epoch, dataloaders, optimizer, lr_scheduler, loss_fn, device):
    exp.model.train()
    #training loop
    total_loss = 0
    num_iters = sum([len(dl) for dl in dataloaders])
    for dataloader in dataloaders:
        with tqdm(total=len(dataloader) * train_config["batchsize"], desc=f'Training Epoch {epoch + 1}/{train_config["epochs"]}', unit='batch') as pbar:
            for batch in dataloader: #tryin unpacking text from 'labels' as in model development
                loss = exp.train_forward_pass(batch, loss_fn, device)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(train_config["batchsize"])
    return total_loss/num_iters
            
            
def val_loop(train_config, exp, epoch, dataloaders, device):
    exp.model.eval()
    total_num_correct = 0
    total_num_samples = 0
    for dataloader in dataloaders:
        with tqdm(total=len(dataloader) * train_config["batchsize"], desc=f'Validation Epoch {epoch + 1}/{train_config["epochs"]}', unit='batch') as pbar:
            for batch in dataloader:
                with torch.no_grad():
                    num_correct, num_samples = exp.val_forward_pass(batch, device)
                    total_num_correct += num_correct
                    total_num_samples += num_samples
                pbar.update(train_config["batchsize"])
    return float(num_correct)/float(num_samples)*100 
    