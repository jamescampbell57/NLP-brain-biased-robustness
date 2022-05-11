import os
import pickle
import torch

from nlpbbb.paths import PATHS

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
    
def save_model(exp, optimizer, loss, config, date, epoch):
    root_dir = PATHS["root"]
    save_dir = f"{root_dir}/results/models"
    models_root = os.path.join(save_dir, date, config["experiment"]["name"])

    if not os.path.exists(models_root):
        os.makedirs(models_root)

    if epoch == 0:
        save_obj(config, os.path.join(models_root, "config"))

    m_name = f"epoch:{epoch}"
    epoch_root = os.path.join(models_root, m_name)
    if not os.path.exists(epoch_root):
        os.mkdir(epoch_root)

    torch.save({
            'epoch': epoch,
            'model_state_dict': exp.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(epoch_root, "train_info"))