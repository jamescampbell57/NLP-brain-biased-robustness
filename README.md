# NLP-brain-biased-robustness
*Code for CS 6740 term project: "CereBERTo: Improving Distributional Robustness with Brain-Like Language Representations"*

This repository contains all code needed to replicate and extend on our paper, from handling of fMRI datasets to OOD evaluation of the brain-biased models.

To install all the needed packages, run 
```
scripts/install_relevant_packages.sh
```
Then, to conduct an experiment, one must either choose an existing config file in nlpbbb/configs or create one's own, specifying one's desired choices with respect to model, dataset, etc.

From there, all training is handled by running
```
python train.py <config file>
```
from the command line. Results are logged using wandb, which is a web API that can track PyTorch training in real-time, and requires one to create an account.

This pipeline handles both the fMRI pre-training and downstream evaluation. To save model weights after training on fMRI, have `<config file>["misc"]["save"] = True` and to load such a brain-biased model, use `<config file>["model"]["brain_biased"] = True` and `<config file>["model"]["state_path"] = path/to/desired/model/weights`
