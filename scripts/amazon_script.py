import os

os.system('chmod +x install_relevant_packages.sh')
os.system('./install_relevant_packages.sh')

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
    root = "/home/ubuntu/nlp-brain-biased-robustness"
    
    with open(f"{root}/nlpbbb/configs/Amazon_BABY.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
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
    