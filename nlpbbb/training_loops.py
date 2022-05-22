# torch imports
import torch
import torch.nn.functional as F

# nlpbbb imports
import nlpbbb as bbb

# random imports 
from tqdm import tqdm
import wandb
from datetime import date as dt
import numpy as np
import pprint

def run_training_config(config):
    #get the date for run saving
    date = dt.today().strftime("%m.%d.%y")
    #First define an experiment object
    if config["misc"]["save"]:
        assert not (config["experiment"]["name"] is None), "Must have a name for the run."
        wandb.init(project=config["experiment"]["model_and_task"], entity="nlp-brain-biased-robustness")
        wandb.run.name = config["experiment"]["name"]
    
    #setup your experiment
    exp = bbb.setup.get_experiment(config)
    
    #Then set your model/optimizer/scheduler
    model = exp.model
    optimizer = exp.optimizer
    lr_scheduler = exp.lr_scheduler
    
    #Extra details
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    num_epochs = config["experiment"]["epochs"]
    #Training loops
    for epoch in range(num_epochs):
        #Run validation every so often, good to do before training
        if epoch % config["experiment"]["val_frequency"] == 0:
            val_losses = []
            if config["experiment"]["experiment_type"] == "HarryPotter" or config["experiment"]["experiment_type"] == "NSD":
                val_losses.append(val_loop(config["experiment"], exp, epoch, exp.val_loaders[0], device))
            else:
                for val_loader in exp.val_loaders:
                    val_losses.append(val_loop(config["experiment"], exp, epoch, val_loader, device))
            
        train_loss = train_loop(config["experiment"], exp, epoch, device)
        #save whenever you validate
        if epoch % config["experiment"]["val_frequency"] == 0:
            if config["misc"]["save"]:
                wandb.log({"train_loss": train_loss})
                if config["experiment"]["experiment_type"] == "HarryPotter" or config["experiment"]["experiment_type"] == "NSD":
                    wandb.log({"val_loss": val_losses[0]})
                else:
                    for i, val_name in enumerate(config["dataset"]["val_datasets"]):
                        wandb.log({val_name: val_losses[i]})
                #bbb.utils.save_model(exp, np.mean(val_losses), config, date, epoch)

        
def train_loop(train_config, exp, epoch, device):
    exp.model.train()
    #training loop
    total_loss = 0
    iter_num = 0
    num_iters = sum([len(dl) for dl in exp.train_loaders])
    for dataloader in exp.train_loaders:
        with tqdm(total=len(dataloader) * train_config["batchsize"], desc=f'Training Epoch {epoch + 1}/{train_config["epochs"]}', unit='batch') as pbar:
            for batch in dataloader: #tryin unpacking text from 'labels' as in model development
                exp.optimizer.zero_grad()
                loss = exp.train_forward_pass(batch, device)
                # standard pytorch backprop
                loss.backward()
                exp.optimizer.step()
                if exp.lr_scheduler is not None:
                    exp.lr_scheduler.step()
                total_loss += loss.item()
                iter_num += 1
                pbar.update(train_config["batchsize"])
                pbar.set_postfix(**{'train loss (batch)': loss.item()})
            pbar.set_postfix(**{'train loss avg': total_loss/iter_num})
    return total_loss/num_iters
            
            
def val_loop(train_config, exp, epoch, dataloader, device):
    exp.model.eval()
    
    #need some flexibility to accomodate STSB/Finetuning
    
    #for Amazon, primary is num correct and secondary is num values
    primary_values = []
    secondary_values = []
    
    with tqdm(total=len(dataloader) * train_config["batchsize"], desc=f'Validation Epoch {epoch + 1}/{train_config["epochs"]}', unit='batch') as pbar:
        for batch in dataloader:
            with torch.no_grad():
                prim_val, seco_val = exp.val_forward_pass(batch, device)
                if train_config["experiment_type"] == "STSB":
                    for idx, similarity in enumerate(seco_val):
                        primary_values.append(prim_val[idx].cpu())
                        secondary_values.append(similarity.cpu())
                else:
                    primary_values.append(prim_val.cpu())
                    secondary_values.append(seco_val)
            pbar.update(train_config["batchsize"])
            if train_config["experiment_type"] != "STSB":
                pbar.set_postfix(**{'val acc (batch)': float(prim_val.cpu()/seco_val)*100})
    
        if train_config["experiment_type"] == "STSB":
            torch_cosines = torch.tensor(primary_values)
            torch_gold = torch.tensor(secondary_values)

            torch_cosines = torch_cosines.reshape((1,torch_cosines.shape[0]))
            torch_gold = torch_gold.reshape((1,torch_gold.shape[0]))
            
            print("Cosines:",torch_cosines)
            print("Gold:",torch_gold)

            combined = torch.cat((torch_cosines, torch_gold), axis=0)
            pbar.set_postfix(**{'val corref avg': torch.corrcoef(combined)[0,1]})
            return torch.corrcoef(combined)[0,1]
        else:
            pbar.set_postfix(**{'val acc avg': float(sum(primary_values)/sum(secondary_values))*100})
            return float(sum(primary_values)/sum(secondary_values))*100 
    
