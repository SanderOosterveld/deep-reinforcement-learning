import argparse
import pprint
import torch
import torch.optim as optim
import torch.nn as nn
import learners.radam.radam_optim as radam

from environments import ContinuousUpswingPendulum, HumanoidEnvironment

from . import run_N_times

import learners.epsilon as eps

activation_function_mapping = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': torch.tanh,
    'softmax': nn.Softmax
}

epsilon_options = {
    "constant": (eps.ConstantEpsilon, ["value"]),
    "linear": (eps.LinearEpsilon, ["start", "end", "steps"]),
    "exponential": (eps.ExponentialEpsilon, ["start", "decay"])
}

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
            if argument == 'n_runs':
                learner_kwargs['n_runs'] = value
                file_name = file_name + "NRuns" + str(value) + "__"
            if argument == 'epsilon_type':
                epsilon_args = args['epsilon_args']
                learner_kwargs['epsilon'] = epsilon_options[value][0](*epsilon_args)
                file_name = file_name + value + str(epsilon_args) + "__"
            if argument == 'batch_size':
                agent_kwargs['batch_size'] = value
                file_name = file_name + "batch_size" + str(value) + "__"
            if argument == 'activation_function':
                agent_kwargs['activation_function'] = activation_function_mapping[value]
                file_name = file_name + value + "__"
            if argument == 'control_scale':
                if args['mujoco']:
                    environment_kwargs['max_torque'] = value
                else:
                    environment_kwargs['torque_angle_limit'] = value
                file_name = file_name + "control_scale" + str(value) + "__"

    if file_name=="":
        file_name = "default_"
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
    parser.add_argument('--n_runs', type=int, help="Number of epochs before calling it quits")
    parser.add_argument('--mujoco', action='store_true', help ="Switch to using the mujoco environment")
    parser.add_argument('--batch_size', type=int, help="Change batch size; default = 150")
    parser.add_argument('--ddpg', action='store_true', help ="Switch to using DDPG-agent")
    parser.add_argument('--epsilon_options', action='store_true', help="Show-epsilon-options")
    parser.add_argument('--epsilon_type', type=str, choices=epsilon_options.keys(), help="Optimizer type: default = linear")
    parser.add_argument('--epsilon_args', type=float, nargs="+", help="Epsilon args, as shown using --epsilon_options")
    parser.add_argument('--activation_function', type=str, choices=activation_function_mapping.keys(), help="Activation function: default = relu")
    parser.add_argument('--control_scale', type=int, help="Max torque scaler, for humanoid max torque in Nm, for pendulum max angle in deg: default = 15 & default = 30 respectively")

    args = parser.parse_args()
    args_dict = vars(args)
    directory_name = args.directory
    if args.store_defaults:
        default_name = "_data/" + directory_name + "/_"
    else:
        default_name = None
    agent_kwargs, learner_kwargs, environment_kwargs, file_name = parse_args(args_dict)
    if args.mujoco:
        environment_class = HumanoidEnvironment
        file_name = file_name + "__mujoco_"
    else:
        environment_class = ContinuousUpswingPendulum
    full_name = "_data/"+directory_name+"/" + file_name

    if args.epsilon_options:
        pprint.pprint(epsilon_options)
        exit()

    run_N_times(agent_kwargs=agent_kwargs, learner_kwargs=learner_kwargs, env_kwargs=environment_kwargs, environment_class = environment_class, base_name=full_name+"0", default_name=default_name)
    run_N_times(agent_kwargs=agent_kwargs, learner_kwargs=learner_kwargs, env_kwargs=environment_kwargs, environment_class = environment_class, base_name=full_name+"1", default_name=default_name)