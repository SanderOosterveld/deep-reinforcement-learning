import argparse
import torch.optim as optim
import learners.radam.radam_optim as radam

from . import run_N_times

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('optimizer', type=str, help="Optimizer type", default="")

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

    print(agent_kwargs)
    print(end_name)