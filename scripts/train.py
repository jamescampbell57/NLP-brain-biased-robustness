# Claims imports
import nlpbbb as bbb

# torch imports
import torch

# misc imports
import wandb
import argparse
from datetime import date
import numpy as np
import os
import yaml
import sys

if __name__ == "__main__":
    with open("/home/vib9/src/UniverSeg/universeg/torch/configs/DEFAULT.yaml", 'r') as stream:
        args = yaml.safe_load(stream)
    if len(sys.argv) > 1:
        new_config = f"/home/vib9/src/UniverSeg/universeg/torch/configs/{sys.argv[1]}.yaml"
        with open(new_config, 'r') as stream:
            new_args = yaml.safe_load(stream)
        args = bbb.setup.merge_dicts(args, new_args)
    bbb.training_funcs.train_net(args)
