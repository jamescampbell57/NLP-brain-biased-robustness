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
    
    with open(f"{root}/nlpbbb/configs/SST2_DEFAULT1.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    bbb.training_loops.run_training_config(config)
    
    with open(f"{root}/nlpbbb/configs/SST2_DEFAULT2.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    bbb.training_loops.run_training_config(config)