import torch

def run_training_config(config):
    args = None
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_function = torch.nn.MSELoss()
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        train_loop(args, epoch, model, optimizer, lr_scheduler, loss_fn, device)
        val_loop(args, epoch, model, dataloader, device)


def train_loop(args, epoch, model, dataloader, optimizer, lr_scheduler, loss_fn, device):
    model.train()
    #training loop
    with tqdm(total=len(dataloader), desc=f'Training Epoch {epoch + 1}/{args["training"]["epochs"]}', unit='batch') as pbar:
        for batch in dataloader: #tryin unpacking text from 'labels' as in model development
            #batch = {k: v.to(device) for k, v in batch.items()}
            features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            preds = model(features)
            targets = F.one_hot((batch['labels']-1).to(torch.int64), num_classes=5).to(device)
            loss = loss_function(preds, targets.float()) #replace .loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
def val_loop(args, epoch, model, dataloader, device):
    model.eval()
    num_correct = 0
    num_samples = 0
    with tqdm(total=len(dataloader), desc=f'Validation Epoch {epoch + 1}/{args["training"]["epochs"]}', unit='batch') as pbar:
        for batch in dataloader:
            #batch = {k: v.to(device) for k, v in batch.items()}
            features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                preds = model(features)
                preds = torch.argmax(preds, axis=1)
                labels = F.one_hot((batch['labels']-1).to(torch.int64), num_classes=5).to(device)
                labels = torch.argmax(labels, axis=1)
                num_correct += (preds==labels).sum()
                num_samples += preds.size(0)
    return float(num_correct)/float(num_samples)*100 
    