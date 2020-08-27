import typing

import numpy as np
from itertools import count

from environments.environment import _Environment
from agents.agent import AgentWithNetworks
from .epsilon import LinearEpsilon, ConstantEpsilon, BaseEpsilonFunction

from datetime import datetime

import utils

import os

LEARNER_N_RUNS = 1000
LEARNER_EPSILON = LinearEpsilon(0.8, 0.2, LEARNER_N_RUNS)
LEARNER_SAVE_FREQUENCY = 10
LEARNER_EVAL_FREQUENCY = 5
LEARNER_RANDOM_INIT_DECAY_RATE = 1
LEARNER_REWARD_SCALE = 1
LEARNER_TARGET_UPDATE = None


class Learner:

    def __init__(self, environment: _Environment, agent: AgentWithNetworks,
                 file_name=None,
                 n_runs=LEARNER_N_RUNS,
                 epsilon=LEARNER_EPSILON,
                 save_frequency=LEARNER_SAVE_FREQUENCY,
                 eval_frequency=LEARNER_EVAL_FREQUENCY,
                 reward_scale=LEARNER_REWARD_SCALE,
                 target_update=LEARNER_TARGET_UPDATE):

        self.env = environment
        self.agent = agent
        self._n_runs = n_runs
        self._eval_freq = eval_frequency
        self._save_freq = save_frequency
        self._reward_scale = reward_scale
        self._target_update = target_update

        self._random_init_range = self.env.get_random_range()

        if isinstance(epsilon, float):
            self.epsilon = ConstantEpsilon(epsilon)
        else:
            assert issubclass(epsilon.__class__,
                              BaseEpsilonFunction), "epsilon must be either a float constant value or a epsilon function"
            self.epsilon = epsilon

        self.file_name = file_name

        self.accumulated_loss = []
        self.accumulated_reward = []
        self.evaluated_reward = []

        self._check_env_and_agent_compatible()
        self.start_time = datetime.now()
        if file_name is not None:
            self.file_name = utils.check_store_name(self.file_name)
            f = open(utils.check_store_name(self.file_name), 'w')
            f.write("Running Learner at: " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\r\n")
            self.total_loss_name = self.file_name + "_total_loss.csv"
            self.evaluated_reward_name = self.file_name + "_evaluated_reward.csv"
            self.total_reward_name = self.file_name + "_total_reward.csv"

        self._first_store = True

    def _check_env_and_agent_compatible(self):
        assert self.env.nb_sensors == self.agent.inputs, "Mismatch in agent inputs (%i) and number of sensors (%i)" % \
                                                         (self.agent.inputs, self.env.nb_sensors)

        assert self.env.nb_actuators == self.agent.outputs, "Mismatch in agent outputs (%i) and number of actuators (" \
                                                            "%i)" % (self.agent.outputs, self.env.nb_actuators)

        assert self.agent.soft_update_speed is None or self._target_update is None, "Cannot both do target update and" \
                                                                                    "soft update, set one to None"

    def evaluate(self):
        self.env.evaluated_init()
        evaluated_reward = self.env.get_reward()
        for _ in count():
            state = self.env.state
            action = self.agent.greedy_action(state)
            self.env.step(action)
            evaluated_reward += self.env.get_reward()
            if self.env.done:
                break
        self.evaluated_reward.append(evaluated_reward)

    def run(self):
        try:
            total_counter = 0

            for epoch in range(self._n_runs):
                total_loss = 0
                total_reward = self.env.get_reward()
                epsilon = self.epsilon()
                if epoch % self._eval_freq == self._eval_freq-1:
                    self.evaluate()
                    self.env.evaluated_init()
                else:
                    self.env.random_init(self._random_init_range)

                if epoch % self._save_freq == self._save_freq-1:
                    self.save()

                print("Init state: %s"%self.env.state)
                for i in count():
                    total_counter += 1
                    old_state = self.env.state

                    controls = self.agent.epsilon_greedy_action(old_state, epsilon)

                    self.env.step(controls)

                    if self.env.success:
                        new_state = None
                    else:
                        new_state = self.env.state

                    reward = self.env.get_reward()*self._reward_scale
                    total_reward += reward
                    loss = self.agent.loss
                    if loss is not None:
                        total_loss += loss

                    self.agent.learn(old_state, controls, new_state, reward)
                    if self.agent.soft_update_speed is None and total_counter % self._target_update == 0:
                        self.agent.hard_update()

                    if self.env.done:
                        self.accumulated_loss.append(total_loss/i)
                        self.accumulated_reward.append(total_reward)
                        break
                print("%s: %.1f%% Total Reward: %f" % (self.file_name.split('/')[-1], (epoch / self._n_runs * 100), total_reward))

        except KeyboardInterrupt:
            self.save()

    def restart(self):
        self.agent.load(self.file_name)
        self.agent.hard_update()
        self.load_lists()
        return self

    def save(self):
        if self.file_name is not None:
            overwrite = self._first_store
            time_diff = datetime.now()-self.start_time
            with open(self.file_name, 'a') as f:
                f.write("Saved Learner after: " + str(time_diff) + "\r\n")
            self.store_lists(overwrite)
            self.agent.store(self.file_name, overwrite=True)

            self._first_store = False

    def store_lists(self, overwrite):
        # If this is the first time for this class to store the variables it should check if the file name already exists
        # and then not override the existing file.
        np.savetxt(self.total_loss_name, self.accumulated_loss, delimiter=',', fmt='%.10f')
        np.savetxt(self.evaluated_reward_name, self.evaluated_reward, delimiter=',', fmt='%.10f')
        np.savetxt(self.total_reward_name, self.accumulated_reward, delimiter=',', fmt='%.10f')

    def load_lists(self):
        total_loss_name = self.file_name + "_average_loss.csv"
        evaluated_reward_name = self.file_name + "_evaluated_reward.csv"
        total_reward_name = self.file_name + "_total_reward.csv"

        if utils.check_open_name(evaluated_reward_name):
            self.evaluated_reward = list(np.loadtxt(evaluated_reward_name))
        if utils.check_open_name(total_loss_name):
            self.accumulated_loss = list(np.loadtxt(total_loss_name))
        if utils.check_open_name(total_reward_name):
            self.accumulated_reward = list(np.loadtxt(total_reward_name))

    def store_defaults(self, file_name):
        store_learner_defaults(file_name)

def store_learner_defaults(file_name):
    file_name += "_defaults_learner.txt"
    file_name = os.path.join(os.getcwd(), file_name)
    file_name = utils.check_store_name(file_name, overwrite=True)
    f = open(file_name, 'w')
    for name, value in globals().items():
        if name.split('_')[0] == 'LEARNER':
            line = str(name)+' ---> ' + str(value) + "\r\n"
            f.write(line)
    f.close()
