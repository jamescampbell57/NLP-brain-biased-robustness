import os

os.system('pip install torch')
os.system('pip install wandb')
os.system('pip install argparse')
os.system('pip install DateTime')
os.system('pip install numpy')
os.system('pip install pyyaml')


# Claims imports
import nlpbbb as bbb

# torch imports
import torch

# misc imports
import wandb
import argparse
from datetime import date
import numpy as np
#import os
import yaml
import sys

if __name__ == "__main__":
    root = "/home/ubuntu/NLP-brain-biased-robustness"
    with open(f"{root}/nlpbbb/configs/Amazon_BABY.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    #if len(sys.argv) > 1:
    #    new_config = f"{root}/nlpbbb/configs/{sys.argv[1]}.yaml"
    #    with open(new_config, 'r') as stream:
    #        new_config = yaml.safe_load(stream)
    #    config = bbb.setup.merge_dicts(config, new_config)
    bbb.training_loops.run_training_config(config)
    
    with open(f"{root}/nlpbbb/configs/Amazon_MUSIC.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    bbb.training_loops.run_training_config(config)
    
    with open(f"{root}/nlpbbb/configs/Amazon_SHOES.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    bbb.training_loops.run_training_config(config)
    
    with open(f"{root}/nlpbbb/configs/Amazon_VIDEO.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    bbb.training_loops.run_training_config(config)
    
    with open(f"{root}/nlpbbb/configs/Amazon_CLOTHES.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    bbb.training_loops.run_training_config(config)
    