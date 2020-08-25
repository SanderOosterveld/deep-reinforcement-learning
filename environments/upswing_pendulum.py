import numpy as np
import math
import os

from .simulators.pendulum import PendulumSimulator, PendulumParameters
from .environment import _Environment, _DiscreteEnvironment
from utils.file_handling import check_store_name

UPSWPEND_PARAMETERS = PendulumParameters()
UPSWPEND_DISCRETE_SPACE = (-1, 0, 1)
UPSWPEND_MAX_TORQUE_ANGLE = 30
UPSWPEND_DT = 0.025
UPSWPEND_ANGLE_REQUIRED = 0.02
UPSWPEND_ANG_VEL_REQUIRED = 0.02
UPSWPEND_ANG_VEL_LIMIT = 6 * math.pi
UPSWPEND_TIME_LIMIT = 8
UPSWPEND_REWARD_KWARGS = {'angle_factor': 0.1,
                          'velocity_factor': 0.001,
                          'control_factor': 0.01}
UPSWPEND_VEL_INIT_RANGE = (-1, 1)
UPSWPEND_ANG_INIT_RANGE = (-math.pi/2, math.pi/2)


def store_upswing_pendulum_defaults(file_name):
    file_name += "_defaults_ups_pend.txt"
    file_name = os.path.join(os.getcwd(), file_name)
    file_name = check_store_name(file_name, overwrite=True)
    f = open(file_name, 'w')
    for name, value in globals().items():
        if name.split('_')[0] == 'UPSWPEND':
            line = str(name) + ' ---> ' + str(value) + "\r\n"
            f.write(line)
    f.close()


class DiscreteUpSwingPendulum(_DiscreteEnvironment):

    def __init__(self, discrete_space=UPSWPEND_DISCRETE_SPACE,
                 torque_angle_limit=UPSWPEND_MAX_TORQUE_ANGLE,
                 angle_required=UPSWPEND_ANGLE_REQUIRED,
                 angular_velocity_required=UPSWPEND_ANG_VEL_REQUIRED,
                 angular_velocity_limit=UPSWPEND_ANG_VEL_LIMIT,
                 time_limit=UPSWPEND_TIME_LIMIT):

        self.simulator = PendulumSimulator(UPSWPEND_PARAMETERS, UPSWPEND_DT)
        self.max_torque = self.simulator.compute_max_torque(torque_angle_limit)

        self.reward_args = UPSWPEND_REWARD_KWARGS
        super(self.__class__, self).__init__(2, 1, discrete_space, control_scale_factor=self.max_torque)

        self.time_limit = time_limit
        self.angular_velocity_limit = angular_velocity_limit

        self.angle_required = angle_required
        self.angular_velocity_required = angular_velocity_required

    def _get_reward(self, angle_factor=0.1, velocity_factor=0.001, control_factor=0):
        """
        Return the reward of the current state of the environment.

        :param angle_factor: Scaling factor for reward due to angle
        :param velocity_factor: Scaling factor for reward due to angular momentum
        :param standardized: the reward can be standardized to give an mean value of 0 along the domain angle in [0,2pi],
        omega in [-3pi, 3pi]
        :return: reward
        """
        angle = self.simulator.get_jasper_angle()
        Ra = angle_factor * ((angle - math.pi) ** 2 - math.pi ** 2)
        Rv = -velocity_factor * self._state.angular_velocity ** 2
        Rc = -control_factor * np.linalg.norm(self._controls/self.max_torque)
        return Ra + Rv + Rc

    @property
    def success(self):
        if abs(self.simulator.get_real_angle() - math.pi) < self.angle_required and abs(
                self.state[1]) < self.angular_velocity_required:
            return True
        else:
            return False

    @property
    def failure(self):
        if self.simulator.time > self.time_limit or abs(self.state[1]) > self.angular_velocity_limit:
            return True
        else:
            return False

    @property
    def done(self):
        return self.failure or self.success

    def reset(self):
        self.state = np.zeros(self.nb_states)
        self.simulator.reset()

    def store_defaults(self, file_name):
        store_upswing_pendulum_defaults(file_name)

    def get_random_range(self):
        return list((UPSWPEND_ANG_INIT_RANGE, UPSWPEND_VEL_INIT_RANGE))


class ContinuousUpswingPendulum(_Environment):

    def __init__(self,
                 torque_angle_limit=UPSWPEND_MAX_TORQUE_ANGLE,
                 angle_required=UPSWPEND_ANGLE_REQUIRED,
                 angular_velocity_required=UPSWPEND_ANG_VEL_REQUIRED,
                 angular_velocity_limit=UPSWPEND_ANG_VEL_LIMIT,
                 time_limit=UPSWPEND_TIME_LIMIT):

        self.simulator = PendulumSimulator(UPSWPEND_PARAMETERS, UPSWPEND_DT)
        self.max_torque = self.simulator.compute_max_torque(torque_angle_limit)

        self.reward_args = UPSWPEND_REWARD_KWARGS

        super(self.__class__, self).__init__(2, 1, control_scale_factor=self.max_torque)

        self.time_limit = time_limit
        self.angular_velocity_limit = angular_velocity_limit

        self.angle_required = angle_required
        self.angular_velocity_required = angular_velocity_required

    def _get_reward(self, angle_factor=0.1, velocity_factor=0.001, control_factor=0.001):
        """
        Return the reward of the current state of the environment.

        :param angle_factor: Scaling factor for reward due to angle
        :param velocity_factor: Scaling factor for reward due to angular momentum
        :param standardized: the reward can be standardized to give an mean value of 0 along the domain angle in [0,2pi],
        omega in [-3pi, 3pi]
        :return: reward
        """
        angle = self.simulator.get_jasper_angle()
        Ra = angle_factor * ((angle - math.pi) ** 2 - math.pi ** 2)
        Rv = -velocity_factor * self._state[1] ** 2
        Rc = -control_factor * np.linalg.norm(self._controls/self.max_torque)
        return Ra + Rv + Rc

    def store_defaults(self, file_name):
        store_upswing_pendulum_defaults(file_name)

    @property
    def success(self):
        if abs(self.simulator.get_real_angle() - math.pi) < self.angle_required and abs(
                self.state[1]) < self.angular_velocity_required:
            return True
        else:
            return False

    @property
    def failure(self):
        if self.simulator.time > self.time_limit or abs(self.state[1]) > self.angular_velocity_limit:
            return True
        else:
            return False

    @property
    def done(self):
        return self.failure or self.success

    def reset(self):
        self.state = np.zeros(self.nb_states)
        self.simulator.reset()

    def get_random_range(self):
        return list((UPSWPEND_ANG_INIT_RANGE, UPSWPEND_VEL_INIT_RANGE))
