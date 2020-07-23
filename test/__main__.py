from .test_0720 import main


import argparse
import torch.optim as optim
import learners.radam.radam_optim as radam

from . import run_N_times

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('optimizer', type=str, help="Optimizer type", default="")
    parser.add_argument('--store_defaults', dest='defaults', action='store_true')
    parser.set_defaults(defaults=False)

    mapping_dict = {
        "": ({}, "default"),
        "Radam":({'optimizer': radam.RAdam}, "Radam"),
        "PlainRadam":({'optimizer': radam.PlainRAdam}, "PlainRadam"),
        "Adagrad": ({'optimizer': optim.Adagrad}, "Adagrad"),
        "AdamW": ({'optimizer': radam.AdamW}, "AdamW"),
        "SGD": ({'optimizer': optim.SGD}, "SGD"),
        "LBFGS": ({'optimizer': optim.LBFGS}, "LBFGS"),
        "Rprop": ({'optimizer': optim.Rprop}, "Rprop"),
    }

    args = parser.parse_args()

    agent_kwargs, end_name = mapping_dict[args.optimizer]
    if args.defaults:
        default_name = "_data/0723/_"
    else:
        default_name = None
    full_name = "_data/0723/" + end_name
    run_N_times(agent_kwargs=agent_kwargs, base_name=full_name+"0", default_name=default_name)
    run_N_times(agent_kwargs=agent_kwargs, base_name=full_name+"1", default_name=default_name)