import argparse
import torch.optim as optim
import torch.nn as nn
import learners.radam.radam_optim as radam

from . import run_N_times

optimizer_mapping = {
    "Radam": ({'optimizer': radam.RAdam}, "Radam"),
    "PlainRadam": ({'optimizer': radam.PlainRAdam}, "PlainRadam"),
    "Adagrad": ({'optimizer': optim.Adagrad}, "Adagrad"),
    "AdamW": ({'optimizer': radam.AdamW}, "AdamW"),
    "SGD": ({'optimizer': optim.SGD}, "SGD"),
    "LBFGS": ({'optimizer': optim.LBFGS}, "LBFGS"),
    "Rprop": ({'optimizer': optim.Rprop}, "Rprop"),
}

loss_function_mapping = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
    "SmoothL1": nn.SmoothL1Loss()
}
def parse_args(args: dict):
    learner_kwargs = {}
    agent_kwargs = {}
    environment_kwargs = {}
    file_name = ""

    for argument in args.keys():
        value = args[argument]
        if value is not None:
            if argument == 'optimizer':
                agent_kwargs = optimizer_mapping[value][0]
                file_name = file_name + optimizer_mapping[value][1] + "__"
            if argument == 'lr':
                agent_kwargs['learning_rate'] = value
                file_name = file_name + "lr" + str(value).replace(".","_") + "__"
            if argument == 'reward_scale':
                learner_kwargs['reward_scale'] = value
                file_name = file_name + "reward_scale" + str(value).replace(".", "_") + "__"
            if argument == 'soft_update':
                agent_kwargs['soft_update_speed'] = value
                file_name = file_name + "soft_update" + str(value).replace(".","_") + "__"
            if argument == 'replay_capacity':
                agent_kwargs['replay_capacity'] = value
                file_name = file_name + "replay_capacity" + str(value) + "__"
            if argument == 'loss_function':
                agent_kwargs['loss_function'] = loss_function_mapping[value]
                file_name = file_name + value + "Loss" + "__"
            if argument == 'gamma':
                agent_kwargs['gamma'] = value
                file_name = file_name + "gamma" + str(value).replace(".","_") + "__"

    if file_name=="":
        file_name = "default"
    else:
        file_name = file_name[0:-1]
    return agent_kwargs, learner_kwargs, environment_kwargs, file_name


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, default = "_")
    parser.add_argument('--optimizer', type=str, choices=optimizer_mapping.keys(), help="Optimizer type: default = Adam")
    parser.add_argument('--store_defaults', action='store_true', help ="Use to store the defaults in 'directory/_'")
    parser.add_argument('--lr', type=float, help="Learning rate: default = 0.0005")
    parser.add_argument('--reward_scale', type=float, help="Scale the reward during learning, eval gives the true rewards: default =  0")
    parser.add_argument('--soft_update', type=float, help="Soft update speed: default = 0.005")
    parser.add_argument('--replay_capacity', type=int, help="Size of the replay memory: default = 100000")
    parser.add_argument('--loss_function', type=str, choices=loss_function_mapping.keys(), help="Loss function used for both the critic and actor: default = MSE")
    parser.add_argument('--gamma', type=float, help="Gamma value: default = 0.99")

    args = parser.parse_args()
    args_dict = vars(args)
    directory_name = args.directory
    if args.store_defaults:
        default_name = "_data/" + directory_name + "/_"
    else:
        default_name = None

    agent_kwargs, learner_kwargs, environment_kwargs, file_name = parse_args(args_dict)
    full_name = "_data/"+directory_name+"/" + file_name
    run_N_times(agent_kwargs=agent_kwargs, base_name=full_name+"0", default_name=default_name)
    run_N_times(agent_kwargs=agent_kwargs, base_name=full_name+"1", default_name=default_name)