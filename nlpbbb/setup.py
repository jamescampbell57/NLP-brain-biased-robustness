# claims imports
import nlpbbb as bbb
import submitit
import yaml
import itertools
import copy

from nlpbbb.paths import PATHS


def get_experiment(config):
    if config["experiment"]["experiment_type"] == "Amazon":
        exp = bbb.TorchExperiments.AmazonExperiment.Experiment
    elif config["experiment"]["experiment_type"] == "MNLI":
        exp = bbb.TorchExperiments.MNLIExperiment.Experiment
    elif config["experiment"]["experiment_type"] == "SST2":
        exp = bbb.TorchExperiments.SST2Experiment.Experiment
    elif config["experiment"]["experiment_type"] == "STSB":
        exp = bbb.TorchExperiments.STSBExperiment.Experiment
    elif config["experiment"]["experiment_type"] == "ReCoRD":
        exp = bbb.TorchExperiments.ReCoRDExperiment.Experiment
    elif config["experiment"]["experiment_type"] == "Yelp":
        exp = bbb.TorchExperiments.YelpExperiment.Experiment
    elif config["experiment"]["experiment_type"] == "HarryPotter":
        exp = bbb.TorchExperiments.HarryPotterExperiment.Experiment
    elif config["experiment"]["experiment_type"] == "NSD":
        exp = bbb.TorchExperiments.NSDExperiment.Experiment
    else:
        raise ValueError("Experiment not implemented yet!")
        
    return exp(config)

def run_submitit_job_array(config_dicts, timeout, mem, num_gpus=1):
    jobs = []
    executor = submitit.AutoExecutor(folder=f'{PATHS["root"]}/bash/submitit')
    executor.update_parameters(timeout_min=timeout, mem_gb=mem, gpus_per_node=num_gpus, slurm_partition="sablab", slurm_wckey="")
    for config in config_dicts:
        job = executor.submit(bbb.training_loops.run_training_config, config)
        jobs.append(job)
    return jobs

def return_empty_dict_copy(original_dict):
    new_dict = {}
    for key in original_dict.keys():
        new_dict[key] = original_dict[key]
        for sub_key in new_dict[key].keys():
            new_dict[key][sub_key] = 0
    return new_dict

def get_num_options(original_dict):
    num_options = 0
    for key in original_dict.keys():
        for sub_key in original_dict[key].keys():
            num_options = len(original_dict[key][sub_key])
    return num_options

def gen_training_name(default_name, run_dict, params):
    names = []
    for key in params.keys():
        if key != "misc":
            subkeys = params[key].keys()
            for sk in subkeys:
                if sk != "state_path":
                    names.append(f"{sk}:{run_dict[key][sk]}")
                else:
                    names.append(f'loaded-epoch:{run_dict[key][sk].split("/")[-2]}')
    return_name = f'config:{default_name}'
    for field in names:
        return_name += f"~{field}"
    print(return_name)
    return return_name

def create_gridsearch(params, default_name=None, merge_default=False):
    if merge_default:
        assert default_name is not None, "Must specify default config."
        with open(f'{PATHS["root"]}/nlpbbb/configs/{default_name}.yaml','r') as stream:
            default = yaml.safe_load(stream)
        
    new_dicts = []
    first_dicts = True
    
    #go through all options you want to set
    for key in params.keys():
        #lower levels like num_feat, model_type, num levels, etc.
        for sub_key in params[key].keys():
            prepared_new_dicts = []
            for option in params[key][sub_key]:
                nd = {
                    key: {
                        sub_key: option
                    }
                }
                prepared_new_dicts.append(nd)
            if first_dicts:
                for nd in prepared_new_dicts:
                    new_dicts.append(nd)
            else:
                old_dicts = new_dicts
                merged_dicts = []
                for od in old_dicts:
                    for nd in prepared_new_dicts:
                        merged_dicts.append(merge_dicts(od, nd))
                new_dicts = merged_dicts
            first_dicts = False
    
    if merge_default:
        for n in range(len(new_dicts)):
            new_dicts[n] = merge_dicts(default, new_dicts[n])
            new_dicts[n]["experiment"]["name"] = gen_training_name(default_name, new_dicts[n], params)

    return new_dicts


def merge_dicts(od, nd):
    merged_dict = copy.deepcopy(od)
    original_dict = copy.deepcopy(nd)
    
    for key in list(original_dict.keys()):
        if key in merged_dict and isinstance(merged_dict[key], dict):
            for subkey in original_dict[key]:
                merged_dict[key][subkey] = original_dict[key][subkey]
        else:
            merged_dict[key] = original_dict[key]
    return merged_dict
