import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Network


class FullyConnectedNetwork(Network):

    def __init__(self, input_nodes, output_nodes, hidden_layers=None, activation_function=F.relu, clamped=False):
        super(FullyConnectedNetwork, self).__init__(input_nodes, output_nodes, hidden_layers)
        self.layers = []
        self.activation_function = activation_function
        self.clamped = clamped
        self.hidden_layers = hidden_layers
        self.tanh = torch.tanh


        self._build_network()
        i = 1
        for fc in self.layers:
            self.add_module("fc%i" % i, fc)
            i += 1

    def _build_network(self):
        if self.hidden_layers is None:
            self._build_network_no_hidden()
        else:
            self._build_network_with_hidden()

    def _build_network_no_hidden(self):
        self.layers = []
        self.layers.append(nn.Linear(self.input_nodes, self.output_nodes))

    def _build_network_with_hidden(self):
        self.layers = []
        self.layers.append(nn.Linear(self.input_nodes, self.hidden_layers[0]))

        # Add the remaining hidden layers using a for loop
        hidden_layers_popped = list(self.hidden_layers)
        hidden_layers_popped.pop(0)
        previous_index = 0
        for layer_size in hidden_layers_popped:
            previous_layer_size = self.hidden_layers[previous_index]
            self.layers.append(nn.Linear(previous_layer_size, layer_size))
            previous_index += 1

        self.layers.append(nn.Linear(self.hidden_layers[previous_index], self.output_nodes))

    def forward(self, input):
        layers_without_final = self.layers.copy()
        final_layer = layers_without_final.pop()
        out = input
        for layer in layers_without_final:
            out = self.activation_function(layer(out))
        out = final_layer(out)
        if self.clamped:
            out = self.tanh(out)
        return out


def _check_input_layer_valid(input_layer, hidden_layers):
    if hidden_layers is None:
        assert input_layer == 0, "Invalid input layer: %i, maximum value is 0" % input_layer
    else:
        assert input_layer <= len(hidden_layers), "Invalid input layer: %i, maximum value is %i" % (
            input_layer, len(hidden_layers))


class DelayedInputNetwork(FullyConnectedNetwork):

    def __init__(self, input_nodes, delayed_input_nodes, output_nodes, hidden_layers=None, activation_function=F.relu,
                 delayed_input_layer=0, clamped=False):
        _check_input_layer_valid(delayed_input_layer, hidden_layers)
        self.delayed_input_layer = delayed_input_layer
        if delayed_input_layer == 0:
            input_nodes += delayed_input_nodes
            super(DelayedInputNetwork, self).__init__(input_nodes, output_nodes, hidden_layers, activation_function,
                                                      clamped)
        else:
            # make the hidden layers mutable and add the delayed input nodes, if necessary
            super(DelayedInputNetwork, self).__init__(input_nodes, output_nodes, hidden_layers, activation_function,
                                                      clamped)

            new_layer_inputs = self.hidden_layers[delayed_input_layer-1]+delayed_input_nodes
            new_layer_outputs = self.hidden_layers[delayed_input_layer]

            self.layers[delayed_input_layer] = nn.Linear(new_layer_inputs, new_layer_outputs)
            i=1
            for fc in self.layers:
                self.add_module("fc%i" % i, fc)
                i += 1

    def forward(self, input):
        """

        :param input: tuple with both the delayed and the regular inputs, where the first dimension (the batch size) must
        be the same
        :return:
        """
        inputs, delayed_inputs = input

        if self.delayed_input_layer == 0:
            out = torch.cat((inputs, delayed_inputs), -1)
        else:
            out = inputs

        layers_without_final = self.layers.copy()
        final_layer = layers_without_final.pop()
        for i in range(len(layers_without_final)):
            out = self.activation_function(layers_without_final[i](out))
            if i + 1 == self.delayed_input_layer:
                out = torch.cat((out, delayed_inputs), -1)

        out = final_layer(out)
        if self.clamped:
            out = self.tanh(out)
        return out
