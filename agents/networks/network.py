import torch.nn as nn
from typing import Type


class Network(nn.Module):

    def __init__(self, input_nodes, output_nodes, hidden_layers=None):
        super(Network, self).__init__()
        if hidden_layers is not None:
            if not isinstance(hidden_layers, tuple):
                raise TypeError("hidden layers should be a tuple when it is not equal to none, for one element use "
                                "hidden_layers='(layer_size, )'")

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

    def _build_network(self):
        raise NotImplementedError

    def rebuild(self):
        self._build_network()

    def soft_update(self, other_network: type(nn.Module), update_speed):
        this_dict = self.state_dict()
        other_dict = other_network.state_dict()

        assert this_dict.keys() == other_dict.keys(), \
            "Policy and target model are not the same therefore cannot update state values"
        for param in this_dict.keys():
            if param.split('.')[1] != 'num_batches_tracked':
                this_dict[param] += update_speed * (other_dict[param] - this_dict[param])

    def hard_update(self, other_network: type(nn.Module)):
        self.load_state_dict(other_network.state_dict)

    # def __str__(self):
    #     return str(self.__class__)