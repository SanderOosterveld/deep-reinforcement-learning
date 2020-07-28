import numpy as np

from .environment import _Environment
from .simulators import FiveLinkedBiped
from utils.file_handling import check_store_name

import os

_HUMANOID_DOF = 14
_N_MOTORS = 4

HUMANOID_MAX_TORQUE = 15
HUMANOID_DT = 0.01
HUMANOID_TIME_LIMIT = 4
HUMANOID_HIP_INIT_RANGE = (-0.1, 0.5)
HUMANOID_KNEE_INIT_RANGE = (0.1, 0.3)
HUMANOID_HIP_VEL_INIT_RANGE = (0, 0)
HUMANOID_KNEE_VEL_INIT_RANGE = (0, 0)
HUMANOID_REWARD_KWARGS = {'Cth':1.0,
                          'Clv':0.1,
                          'required_linvel':1.0,
                          'Cc':0.05,
                          'alive_bonus':1.0,
                          'torso_min':0.7,
                          'power':0.33,
                          'torso_rectified_loss':1.0}

def get_joint_angles():
    def _tuple_to_2x2_np(range):
        np_range = np.zeros((2, 2))
        np_range[:, 0] = range[0]
        np_range[:, 1] = range[1]
        return np_range

    joint_random_range = np.zeros(_HUMANOID_DOF * 2).reshape((_HUMANOID_DOF, 2))

    knee_pos_mask = np.zeros(_HUMANOID_DOF, dtype=np.bool)
    knee_pos_mask[8::4] = True
    knee_vel_mask = np.zeros(_HUMANOID_DOF, dtype=np.bool)
    knee_vel_mask[9::4] = True
    hip_pos_mask = np.zeros(_HUMANOID_DOF, dtype=np.bool)
    hip_pos_mask[6::4] = True
    hip_vel_mask = np.zeros(_HUMANOID_DOF, dtype=np.bool)
    hip_vel_mask[7::4] = True

    _tuple_to_2x2_np(HUMANOID_HIP_INIT_RANGE)
    joint_random_range[knee_pos_mask] = _tuple_to_2x2_np(HUMANOID_KNEE_INIT_RANGE)
    joint_random_range[knee_vel_mask] = _tuple_to_2x2_np(HUMANOID_KNEE_VEL_INIT_RANGE)

    joint_random_range[hip_pos_mask] = _tuple_to_2x2_np(HUMANOID_HIP_INIT_RANGE)
    joint_random_range[hip_vel_mask] = _tuple_to_2x2_np(HUMANOID_HIP_VEL_INIT_RANGE)

    return joint_random_range


_HUMANOID_INIT_RANGE = get_joint_angles()


class HumanoidEnvironment(_Environment):

    def __init__(self,
                 dt=HUMANOID_DT,
                 time_limit=HUMANOID_TIME_LIMIT,
                 max_torque=HUMANOID_MAX_TORQUE):
        self.simulator = FiveLinkedBiped().set_dt(dt)
        self.reward_args = HUMANOID_REWARD_KWARGS
        self.max_torque = max_torque
        super(self.__class__, self).__init__(_HUMANOID_DOF, _N_MOTORS, control_scale_factor=max_torque)

        self.time_limit = time_limit

    @property
    def sim(self) -> FiveLinkedBiped:
        return self.simulator.sim

    def _get_reward(self, Cth, Clv, required_linvel, Cc, alive_bonus, torso_min, power, torso_rectified_loss):
        """
        Total reward equals:
        Loss due to torso-height = -Cth(1/sqrt(torso_height) - 1/sqrt(torso_height_null))
        Loss due to lin-vel =       -Clv|required_linvel - com_linvel|
        Loss due to controls = -Cc*norm(controls)
        Alive_bonus = alive_bonus
        :return:
        """
        torso_height = self.simulator.torso_height
        torso_loss = -Cth * (1 / (max(torso_height, torso_min + 0.01) - torso_min) ** power - 1 / (
                    self.simulator.torso_height_null - torso_min) ** power)

        if self.simulator.torso_height < torso_min:
            torso_loss -= Cth*torso_rectified_loss*(torso_min-torso_height)

        lin_vel_loss = -Clv * (required_linvel - self.simulator.linvel[0]) ** 2
        control_loss = -Cc * np.linalg.norm(np.asarray(self.controls)) / self.max_torque

        if True in self.simulator.joint_constraints():
            control_loss -= 1000 * Cc
        return torso_loss + lin_vel_loss + control_loss + alive_bonus

    @property
    def done(self):
        return self.simulator.time > self.time_limit

    def get_random_range(self):
        return list(_HUMANOID_INIT_RANGE)


def store_humanoid_defaults(file_name):
    file_name += "_defaults_humanoid.txt"
    file_name = os.path.join(os.getcwd(), file_name)
    file_name = check_store_name(file_name)
    f = open(file_name, 'w')
    for name, value in globals().items():
        if name.split('_')[0] == 'HUMANOID':
            line = str(name) + ' ---> ' + str(value) + "\r\n"
            f.write(line)
    f.close()
