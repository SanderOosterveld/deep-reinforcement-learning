import numpy as np
import typing

import warnings

from .simulators.simulator import _Simulator


class _Environment:
    """
    Base of the environment. An environment is defined an object/space where something is simulated, this can be
    either a cart, human, pendulum or space station. The goal of this class is to combine the needs of all these
    different environments into one base. This base can than be used inside the learning is such a way that one learner
    can be used to combine them all.

    It is hereby assumed that environment (like agent) will be an input to the learner.

    Therefore the needs for the environment are:
    - an general function to get the state (required by the agent)
    - an general function to make a step in the simulation
    - an general function to receive the next-state, maybe through just calling the state again or as an return of the
        step function
    - a reward function
    - the size of the action space needed to build the agents (would be nice to know)
    - the size/limits of the action space.
    - a way to (re-)initialize the environment
    - a way to know if the simulation is done/out of bound such that we can return a done.

    Now let's think a bit on noisy environments, (for which a noise in state measurement is assumed) for them we will
    add the function to know the real state and the 'read' state. It is therefore nice to get the state using a function
    name observe...

    Finally we need to determine what we already know/what is general for all the environments.
    First off there is always a set number of states and actions. When the environment is discrete (inherit from this)
    there must be another option (discrete learning) to show the action space.

    To keep the nomenclature nice we use sensors and actuators as an input.

    It makes the very basic assumption that the number of sensors is always equal to the number of states, but this is
    not true
    """

    def __init__(self, nb_states, nb_actuators, control_scale_factor=1):
        self.nb_states = nb_states
        self.nb_sensors = nb_states
        self.nb_actuators = nb_actuators

        self._state = np.zeros(nb_states)
        self._controls = np.zeros(nb_actuators)
        self._sensor_data = np.zeros(self.nb_sensors)

        self._control_scale_factor = control_scale_factor

        if not hasattr(self, 'simulator'):
            self.simulator = _Simulator()
            warnings.warn("No simulator defined, using basic _simulator however this does nothing so likely code"
                          "will give exception soon")

        if not hasattr(self, 'reward_args'):
            self.reward_args = {}

    def step(self, controls) -> np.ndarray:
        if type(controls) == int or type(controls) == float:
            controls = [controls, ]
        self._check_valid_control(controls)
        controls = controls.copy()
        for i in range(len(controls)):
            controls[i] = self._control_scale_factor * controls[i]
        self._controls[:] = np.asarray(controls)

        self._state = self._step(self._controls)
        return self._state

    def _check_valid_control(self, controls):
        assert len(controls) == self.nb_actuators, "Given control values is: %i, but number of actuators is: %i" % \
                                                   (len(controls), self.nb_actuators)

    def _step(self, controls) -> np.ndarray:
        return self.simulator.do_simulation(controls)

    def set_state(self, state):
        self._state[:] = state[:]
        self.simulator.set_state(state)

    def get_state(self):
        return self._state

    def random_init(self, random_range: typing.Union[list, tuple, float, np.ndarray] = (-0.1, 0.1)):
        """
        Randomly initializes the environment with the given range. Always applies this random range to all the
        states which define the problem. E.g with the five-linked-biped the x,y coordinate and velocities are the first
        four states of the environment so maybe you want to not take these into account when randomly initializing
        :param random_range: size of the uniform range in which the random_init initializes. If a float is given this
        is assumed to be a symmetric range i.e. 0.1 -> random_range = (-0.1,0.1). If one random range is given the range
        is used for all the states, if exactly the amount of ranges is given for the amount of states each states
        gets the given range
        :return: The new state
        """
        self.reset()
        if type(random_range) == float:
            random_range = (-random_range, random_range)

        if type(random_range) is not np.ndarray:
            random_range = np.asarray(list(random_range))

        # if random_range.shape == (self.nb_states,):
        #     random_range = np.stack((-random_range, random_range), axis=1)

        if random_range.shape == (2,):
            random_states = np.random.uniform(random_range[0], random_range[1], self.nb_states)
        elif random_range.shape == (self.nb_states, 2):
            random_states = np.zeros(self.nb_states)
            for i in range(self.nb_states):
                state_range = random_range[i]
                random_states[i] = np.random.uniform(state_range[0], state_range[1])
        else:
            raise TypeError("Range of shape %s, it can only be of shape %s, %s or %s" %
                            (random_range.shape, (2,), (self.nb_states, 2), (self.nb_states, 1)))
        self.state = random_states

        return self.state

    def get_random_range(self):
        """
        The easy way to get the standard random range for the problem, returns a random range which can be used for
        random init. Ofcourse the learner is free to modify this range
        :return: list of tuples for the given environment which can optionally be used as a random init.
        """
        raise NotImplementedError

    def evaluated_init(self):
        self.reset()

    def get_reward(self, **reward_adaptions):
        reward_args = self.reward_args
        if reward_adaptions:
            for key in reward_adaptions.keys():
                if key in reward_args.keys():
                    reward_args[key] = reward_adaptions[key]

        return self._get_reward(**reward_args)

    def _get_reward(self, **kwargs):
        raise NotImplementedError

    @property
    def done(self):
        raise NotImplementedError

    @property
    def success(self):
        return False

    def reset(self):
        self.simulator.reset()

    def measure(self):
        self._sensor_data[:] = self.state
        return self._sensor_data

    @property
    def state(self) -> np.ndarray:
        return self.get_state()

    @state.setter
    def state(self, state: np.ndarray):
        self.set_state(state)

    @property
    def controls(self):
        return self._controls


class _DiscreteEnvironment(_Environment):

    def __init__(self, nb_states, nb_inputs, discrete_space, control_scale_factor=1):
        self.discrete_space = np.asarray(list(discrete_space))
        # check if the first element of the discrete_space is a float, in that case we have a one dimensional tuple
        # tuple/list input and the dimension needs to be expanded to have the same format as that of multiple
        # state input
        if type(self.discrete_space[0]) == float or int:
            self.discrete_space = np.expand_dims(self.discrete_space, axis=0)
        assert self.discrete_space.shape[0] == nb_inputs, "discrete space not of correct shape is %s: should be: %s" % \
                                                     (self.discrete_space.shape, "(" + str(nb_inputs) + ", N)")

        super(_DiscreteEnvironment, self).__init__(nb_states, nb_inputs, control_scale_factor)

    def _check_valid_control(self, controls):
        assert len(controls) == self.nb_actuators, "Given control values is: %i, but number of actuators is: %i" % \
                                                   (len(controls), self.nb_actuators)
        index = 0
        for control in controls:
            if control not in self.discrete_space[index]:
                raise ValueError("Control with index %i not in discrete space %s. Value of control is %s") % \
                      (index, self.discrete_space[index], control)
            index += 1

    def control_indices(self, controls):
        if type(controls) == int or float:
            controls = (controls,)
        indices = []
        self._check_valid_control(controls)
        index = 0
        for control in controls:
            control_space = list(self.discrete_space[index])
            indices.append(control_space.index(control))
            index += 1
        if len(indices) == 1:
            return indices[0]
        else:
            return indices

    def _step(self, controls) -> np.ndarray:
        return self.simulator.do_simulation(controls)

    @property
    def done(self):
        raise NotImplementedError

    def _get_reward(self, *args):
        raise NotImplementedError




