import random
import numpy as np
from typing import Type
import os
import warnings

import utils
import torch

import agents.networks.network as net
from collections import namedtuple

REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 128
SOFT_UPDATE_SPEED = 0.001

class Agent:

    def __init__(self, n_sensors, n_actuators):
        self._inputs = n_sensors
        self._outputs = n_actuators
        pass

    def greedy_action(self, state: np.ndarray):
        """
        Find the action which is the best according to the policy network for the current state

        :param state: numpy array with the size equal to the
        :return:
        """
        if state.shape == (self._inputs,):
            return self._greedy_action(state)
        else:
            raise ValueError("State not of correct shape. Needs to be: %s, now is %s" % (state.shape, (self._inputs,)))

    def _greedy_action(self, state: np.ndarray):
        raise NotImplementedError

    def random_action(self):
        return self._random_action()

    def _random_action(self):
        raise NotImplementedError

    def epsilon_greedy_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = self.random_action()
        else:
            action = self.greedy_action(state)
        return action

    def learning_action(self, state, *args):
        self.epsilon_greedy_action(state, *args)

    def learn(self, old_state, action, new_state, reward):
        raise NotImplementedError

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """ Class which holds the replay memory.

        len() can be used to find the amount of elements in the replay memory, after pushing enough values this will
        be equal ot the capacity.
    """

    def __init__(self, capacity):
        """
        Initializeds the replay memory
        :param capacity: set the size of the replay memory
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Add an argument to the memory, if the memory is size of capacity replaces the first entry
        :param args: Values used in the transition object: state, action, new_state, reward
        :return:
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns a sample from the replay memory, already zips the sample to hold the seperate parts together in the sample
        eg: reward = (reward1, reward2, reward3), state = (state1, state2, state3)
        :param batch_size: Number of values returned in sample
        :return: Transition object with the zipped transition parts, state, action, new_state, reward
        """
        sample = random.sample(self.memory, batch_size)
        return Transition(*zip(*sample))

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return str(self.sample(len(self)))


def find_in_networks(networks, key, value):
    result = []
    for (network, kwargs) in networks:
        if key in kwargs:
            if kwargs[key] == value:
                result.append((network, kwargs))

    return result


class AgentWithNetworks(Agent):

    def __init__(self, n_sensors, n_actuators,
                 replay_memory_size=REPLAY_MEMORY_SIZE,
                 batch_size=BATCH_SIZE,
                 soft_update_speed=SOFT_UPDATE_SPEED):
        super(AgentWithNetworks, self).__init__(n_sensors, n_actuators)

        self.networks = []

        self.replay_memory = ReplayMemory(replay_memory_size)
        self.batch_size = batch_size

        self.soft_update_speed = soft_update_speed

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _add_network(self, network: net.Network, **kwargs):
        """
        You can add the networks using this function the possible kwargs are:
        network_store_name: name under which the networks will be stored (str)
        type: used to split different networks form each other (str)
        target: If this is set to true the network will be copied (or attempted to be copied) on hard/soft update (bool)
        :param network:
        :param kwargs:
        :return:
        """
        self.networks.append((network, kwargs))
        return network

    def load(self, file_path):
        """
        Load the policy networks from the file names.
        :param file_path: Path name from current directory
        :return: None
        """

        for (network, kwargs) in self.networks:
            if 'network_store_name' in kwargs.keys():
                network_file_name = os.path.join(os.getcwd(), file_path + "_" + kwargs['network_store_name'])
                if utils.check_open_name(network_file_name):
                    network.load_state_dict(torch.load(network_file_name))

    def store(self, file_path, **store_kwargs):
        """
        Load the policy networks from the file names.
        :param file_path: Path name from current directory
        :return: None
        """

        for (network, kwargs) in self.networks:
            if 'network_store_name' in kwargs.keys():
                network_store_name = os.path.join(os.getcwd(), file_path + "_" + kwargs['network_store_name'])
                network_store_name = utils.check_store_name(network_store_name, **store_kwargs)
                torch.save(network.state_dict(), network_store_name)

    def soft_update(self):
        did_something = False
        for (network, kwargs) in self.networks:
            if 'target_network' in kwargs.keys() and kwargs['target_network']:
                if 'type' in kwargs.keys():
                    same_type_networks = find_in_networks(self.networks, 'type', kwargs['type'])
                else:
                    same_type_networks = self.networks
                non_target = find_in_networks(same_type_networks, 'target_network', False)
                assert len(non_target) == 1, "Something wrong with the initialization of the networks since there are" \
                                             " %i networks to copy the target from. \n This: %s,\n\t Non Target: %s" % \
                                             (len(non_target), (network, kwargs), non_target)
                network.soft_update(non_target[0][0], self.soft_update_speed)
                did_something = True
        assert did_something, "Tried to do a soft-update but did not do anything, make sure to define the" \
                                      "'target_network' parameter for the networks"

    def hard_update(self):
        did_something = False
        for (network, kwargs) in self.networks:
            if 'target_network' in kwargs.keys() and kwargs['target_network']:
                if 'type' in kwargs.keys():
                    same_type_networks = find_in_networks(self.networks, 'type', kwargs['type'])
                else:
                    same_type_networks = self.networks
                non_target = find_in_networks(same_type_networks, 'target_network', False)
                assert len(non_target) == 1, "Something wrong with the initialization of the networks since there are" \
                                             " %i networks to copy the target from. \n This: %s,\n\t Non Target: %s" % \
                                             (len(non_target), (network, kwargs), non_target)
                network.hard_update(non_target[0][0])
                did_something = True

        assert did_something, "Tried to do a hard-update but did not do anything, make sure to define the" \
                                      "'target_network' parameter for the networks"

    def add_to_memory(self, state, action, next_state, reward):
        """

        :param state:
        :param action:
        :param next_state:
        :param reward:
        :return:
        """

        # Change the
        tensor_state = torch.from_numpy(state.astype(np.float32)).view((1, -1)).to(self.device)
        tensor_action = torch.tensor([action], dtype=torch.float32, device=self.device).view((1, -1))
        tensor_reward = torch.tensor([reward], dtype=torch.float32, device=self.device).view((1, -1))

        # if next_state is a real state than we need to convert it to a tensor, else we just put None in it
        # (next state is final)
        if next_state is not None:
            assert len(state) == len(next_state) and len(state) == self._inputs, \
                "States must have the size of the input nodes, next_state can be None for final state"
            tensor_next_state = torch.from_numpy(next_state.astype(np.float32)).view((1, -1)).to(self.device)
        else:
            tensor_next_state = None

        self.replay_memory.push(tensor_state, tensor_action, tensor_next_state, tensor_reward)

    def _sample_from_memory(self):
        return self.replay_memory.sample(self.batch_size)

    @property
    def loss(self):
        raise NotImplementedError