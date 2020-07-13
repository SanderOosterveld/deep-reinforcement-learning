import torch
import math
import matplotlib.pyplot as plt
import numpy as np

from agents.networks import FullyConnectedNetwork, DelayedInputNetwork
def get_policy_matrix(policy_model: FullyConnectedNetwork, samples = 300, range_angle = (-math.pi, math.pi), range_vel = (-3*math.pi, 3*math.pi), device =torch.device("cuda") ):
    angles = torch.linspace(range_angle[0], range_angle[1], steps=samples, device=device).unsqueeze(1)
    velocities = torch.linspace(range_vel[0], range_vel[1], steps=samples, device=device)
    state_array = torch.tensor([], device=device)
    for velocity in velocities:
        vel_array = velocity*torch.ones(angles.shape, device=device)
        sub_state_array = torch.cat((angles, vel_array), 1)
        state_array = torch.cat((state_array, sub_state_array))

    with torch.no_grad():
        policy = policy_model(state_array)
    policy = policy.reshape((samples, samples))

    return np.flip(np.rot90(policy.cpu().numpy()), axis=0)

def plot_policy_matrix(policy_model: FullyConnectedNetwork, samples = 100, range_angle = (-math.pi, math.pi), range_vel = (-3*math.pi, 3*math.pi), device =torch.device("cuda") ):
    policy = get_policy_matrix(policy_model, samples, range_angle, range_vel, device)

    plot_matrix(policy)
    plt.show()

def plot_matrix(matrix: np.ndarray):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect='auto')
    fig.colorbar(im, ax=ax)
   # plt.show()

def get_value_matrix(policy_model: FullyConnectedNetwork, q_model: DelayedInputNetwork, samples = 300, range_angle = (-math.pi, math.pi), range_vel = (-3*math.pi, 3*math.pi), device =torch.device("cuda") ):
    angles = torch.linspace(range_angle[0], range_angle[1], steps=samples, device=device).unsqueeze(1)
    velocities = torch.linspace(range_vel[0], range_vel[1], steps=samples, device=device)
    state_array = torch.tensor([], device=device)
    for velocity in velocities:
        vel_array = velocity * torch.ones(angles.shape, device=device)
        sub_state_array = torch.cat((angles, vel_array), 1)
        state_array = torch.cat((state_array, sub_state_array))

    with torch.no_grad():
        actions = policy_model(state_array)
        values = q_model((state_array, actions))

    values = values.reshape((samples, samples))

    return np.flip(np.rot90(values.cpu().numpy()), axis=0)