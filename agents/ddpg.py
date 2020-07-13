from agents.agent import AgentWithNetworks

from .networks import FullyConnectedNetwork, DelayedInputNetwork
from utils import check_store_name
from utils.random_processes import OrnsteinUhlenbeckProcess

import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


DDPG_REPLAY_CAPACITY = 100000
DDPG_BATCH_SIZE = 150
DDPG_ACTIVATION_FUNCTION = F.leaky_relu
DDPG_GAMMA = 0.99
DDPG_CLAMPED_GRADIENT = False
DDPG_LOSS_FUNCTION = nn.MSELoss()
DDPG_OPTIMIZER = optim.Adam
DDPG_LEARNING_RATE = 0.0005
DDPG_SOFT_UPDATE_SPEED = 0.005
DDPG_CRITIC_LR_MULTIPLIER = 1
DDPG_ACTION_INSERTION = 1
DDPG_HIDDEN_LAYERS = (100, 200)


class DDPGAgent(AgentWithNetworks):

    def __init__(self, states,
                 actions,
                 hidden_layers=DDPG_HIDDEN_LAYERS,
                 replay_capacity=DDPG_REPLAY_CAPACITY,
                 batch_size=DDPG_BATCH_SIZE,
                 activation_function=DDPG_ACTIVATION_FUNCTION,
                 gamma=DDPG_GAMMA,
                 clamped_gradient=DDPG_CLAMPED_GRADIENT,
                 loss_function=DDPG_LOSS_FUNCTION,
                 optimizer=DDPG_OPTIMIZER,
                 learning_rate=DDPG_LEARNING_RATE,
                 soft_update_speed=DDPG_SOFT_UPDATE_SPEED,
                 critic_lr_multiplier=DDPG_CRITIC_LR_MULTIPLIER,
                 action_insertion=DDPG_ACTION_INSERTION,
                 **ornstein_kwargs):

        super(DDPGAgent, self).__init__(states, actions, replay_capacity, batch_size)

        self.actor_lr = learning_rate
        self.critic_lr = learning_rate * critic_lr_multiplier

        self.clamped_gradient = clamped_gradient
        self.loss_function = loss_function
        self.soft_update_speed = soft_update_speed
        self.gamma = gamma

        self.actor = FullyConnectedNetwork(states, actions, hidden_layers, activation_function, True).to(self.device)
        self.actor_target = FullyConnectedNetwork(states, actions, hidden_layers, activation_function, True).to(self.device)
        self.actor_target.eval()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.actor_lr)

        self._add_network(self.actor, target_network=False, network_store_name="actor.pth", type="actor")
        self._add_network(self.actor_target, target_network=True, type="actor")

        self.critic = DelayedInputNetwork(states, actions, 1, hidden_layers, activation_function, action_insertion).to(self.device)
        self.critic_target = DelayedInputNetwork(states, actions, 1, hidden_layers, activation_function,
                                                 action_insertion).to(self.device)
        self.critic_target.eval()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.critic_lr)

        self._add_network(self.critic, target_network=False, network_store_name="critic.pth", type="critic")
        self._add_network(self.critic_target, target_network=True, type="critic")

        self._actions = actions
        self._states = states
        self._actor_loss = torch.tensor([0])
        self._critic_loss = torch.tensor([0])

        self.randomizer = OrnsteinUhlenbeckProcess(size=self._actions, **ornstein_kwargs)

    def _random_action(self):
        return np.random.uniform(-1, 1, self._actions)

    def _greedy_action(self, state: np.ndarray):
        return self.select_ddpg_actions(state, 0)

    def select_ddpg_actions(self, state, epsilon):
        state_tensor = torch.from_numpy(state.astype(np.float32)).view((1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().reshape(-1)
            action += max(epsilon, 0) * self.randomizer.sample()
            action = np.clip(action, -1, 1)
            return action

    def epsilon_greedy_action(self, state, epsilon):
        if len(self.replay_memory) >= self.batch_size:
            return self.select_ddpg_actions(state, epsilon)
        else:
            return self.random_action()

    def learn(self, old_state, action, new_state, reward):
        if old_state is not None:
            self.add_to_memory(old_state, action, new_state, reward)

        if len(self.replay_memory) > self.batch_size:
            batch = self._sample_from_memory()
            # Compute and act on the Bellman's loss in the critic (DQN) network
            current_q_values = self.compute_q_value(batch.state, batch.action)
            expected_q_values = self.compute_expected_value(batch.next_state, batch.reward)
            self._critic_loss = self.compute_critic_loss(current_q_values, expected_q_values)
            self.update_critic_network(self._critic_loss)

            self._actor_loss = self.compute_actor_loss(batch.state)
            self.update_actor_network(self._actor_loss)

            if self.soft_update_speed is not None:
                self.soft_update()

    def compute_q_value(self, states, actions):
        """
        Computes the current (batch of) state action pairs. Takes a tuple of states and the tuple of corresponding
        actions. Then using the critic network finds the q-values.
        :param states: tuple of states tensors
        :param actions: tuple of action tensors
        :return: a torch tensor with all the q-values for the state-action pairs shape is (N,1) where N is the number of
        elements in states and actions
        """
        state_batch = torch.cat(states)
        action_batch = torch.cat(actions)

        return self.critic((state_batch, action_batch))

    def compute_expected_value(self, next_states, rewards):
        """
        Computes the batch of expected values. When the next state is None the value of that state is set to
        0. Note that we talk about the 'value' not the 'q-value', it returns the q-value for given next_state when the
        'expected' best action is taken.
        :param next_state_values: tuple of next_state_values
        :param reward: tuple of rewards corresponding to moving to this next state.
        :return: Torch tensor with the expected q-values for the current state for the next_state and rewards as
        given. Returns an (N,1) torch tensor where N is the number of elements in next_state_value and reward
        """

        # First we have to find the non_final next_states for we only need to find the state for these, the value
        # for the others is 0 anyhow.

        non_final_mask = torch.tensor(list(map(lambda s: s is not None, next_states)),
                                      device=self.device, dtype=torch.bool)
        if len(non_final_mask) != self.batch_size:
            print(len(non_final_mask))
        non_final_next_states = torch.cat([s for s in next_states if s is not None])

        expected_best_actions = self.actor_target(non_final_next_states)

        next_state_values = torch.zeros((self.batch_size, 1), device=self.device)
        next_state_values[non_final_mask] = self.critic_target((non_final_next_states, expected_best_actions)).detach()

        reward_batch = torch.cat(rewards)
        expected_value = self.gamma * next_state_values + reward_batch
        return expected_value

    def compute_critic_loss(self, q_values, expected_q_values):
        """
        Simply apply the loss function to the q-value and what the q-value should be. This can be overwritten to make
        a regularized model where an additional loss is added to based on the size of the nodes.
        :param q_values:
        :param expected_q_values:
        :return:
        """
        return self.loss_function(q_values, expected_q_values)

    def update_critic_network(self, critic_loss):
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clamped_gradient:
            for param in self.critic.parameters():
                param.grad.data.clamp(-1, 1)

        self.critic_optimizer.step()

    def compute_actor_loss(self, states):
        """
        The loss is equal to negative q-value based on the action of the current actor network. Based on the
        current critic network. Since the loss needs to be always become more negative in the gradient descent
        method it will take the step which will make this value more negative e.g. the policy which action which
        would have resulted in a higher value.

        :param states: Batch with the current states
        :return: loss for the actor network in batch shape
        """
        state_batch = torch.cat(states)
        expected_best_actions = self.actor(state_batch)
        negative_q_value = -self.critic((state_batch, expected_best_actions))
        actor_loss = negative_q_value.mean()

        return actor_loss

    def update_actor_network(self, actor_loss):
        self.actor.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clamped_gradient:
            for param in self.actor.parameters():
                param.grad.data.clamp(-1, 1)

        self.actor_optimizer.step()

    @property
    def loss(self):
        return np.array([float(self._critic_loss.item()), float(self._actor_loss.item())])

    def store_defaults(self, file_name):
        store_ddpg_defaults(file_name)


def store_ddpg_defaults(file_name):
    file_name += "_defaults_ddpg.txt"
    file_name = os.path.join(os.getcwd(), file_name)
    file_name = check_store_name(file_name, overwrite=True)
    f = open(file_name, 'w')
    for name, value in globals().items():
        if name.split('_')[0] == 'DDPG':
            line = str(name)+' ---> ' + str(value) + "\r\n"
            f.write(line)
    print("Done writing to " + file_name)
    f.close()
