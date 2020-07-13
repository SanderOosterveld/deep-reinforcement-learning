from environments.simulators.mujoco import MuJoCoSimulator


import numpy as np

import os

HIP_RANGE = (-0.53, 1.92)


def get_xml_path(xml_file = "walker_v2.xml"):
    dir_name = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_name, xml_file)


class FiveLinkedBiped(MuJoCoSimulator):

    def __init__(self):
        super(self.__class__, self).__init__(get_xml_path())

        self.torso_height_null = self.torso_height

        self.left_hip_constrained = 0

    def _get_body_positions(self):
        all_position = self.data.xipos[1:]

        return all_position[:,[0,2]]


    def _check_done(self):
        qpos = self.get_qpos_qvel()[0]
        qvel = self.get_qpos_qvel()[1]

        print(qpos)
        print(qvel)

    def _compute_energy(self):
        return self.sim.data.energy[0], self.sim.data.energy[1]

    def get_joint_angles(self):
        return self.get_qpos_qvel()[3:, 0]

    def set_joint_angles(self, angles):
        qpos = self.get_qpos_qvel()[:, 0]
        qvel = self.get_qpos_qvel()[:, 1]
        qpos[3:] = angles
        self.set_qpos_qvel(qpos, qvel)


    @property
    def com(self):
        return self.sim.data.sensordata[[0, 2]]

    @property
    def linvel(self):
        """
        Linear velocity of the center of mass of the 5 linked biped. First value is the xvelocity and the second
        value is the y-velocity
        :return:
        """
        return self.sim.data.sensordata[[3,5]]

    @property
    def torso_height(self):
        return self.sensordata[8]

    def joint_constraints(self):
        """
        Order of the constraints is: right knee, right hip, left hip, left knee
        :return:
        """
        right_knee_constrained = self.sensordata[9]<0
        right_hip_constrained = self.sensordata[10]<0
        left_knee_constrained = self.sensordata[11]<0
        left_hip_constrained = self.sensordata[12]<0

        return right_knee_constrained, right_hip_constrained, left_hip_constrained, left_knee_constrained

    def negative_bending_knees(self):
        qpos = self.get_joint_angles()
        right_knee_negative_bending = max(0, -qpos[1])
        left_knee_negative_bending = max(0, -qpos[3])
        return right_knee_negative_bending, left_knee_negative_bending

    def constrains_value(self, knee_power_factor=0.33):
        """
        Helper class which uses the fact that the knee can become a slight negative angle without directly reaching
        the constrained motion as that would cause the knee to never be straight.

        It then takes the max value of all the constraints to see if something is constrained. If something is
        constrained returns a 1. Next has a function going from 0 to max negative knee angle which maps to 0-1 based
        on how negative the knee is bent and the power given in the input.
        :return:
        """
        knee_power = knee_power_factor
        knee_negative_max = -self.knee_limits()[0]+0.01
        joint_constraints_reached = np.amax(np.asarray(self.joint_constraints(), dtype=np.int))
        print(joint_constraints_reached)
        knee_constraints_value = np.amax((np.asarray(self.negative_bending_knees())/knee_negative_max)**knee_power)

        print(max(joint_constraints_reached, knee_constraints_value))


    def knee_limits(self):
        return self.joint_limits[4]

if __name__ == '__main__':
    import time
    import mujoco_py
    env = FiveLinkedBiped().set_dt(0.002)
    print(env.model.opt.gravity)
    # env.model.opt.gravity[2]=0
    print(env.n_bodies)
    viewer = mujoco_py.MjViewer(env.sim)
    viewer._paused = True
    env.negative_bending_knees()
    print(env.joint_limits)
    print(env.sensordata)
    for i in range(10000000):
        env.do_simulation([0, 0.4, 0, -0.4])
        viewer.render()
        #print(env.torso_height)
        env.constrains_value()
        print(env.sim.data.energy)
        # print("COM: %s \t LinVel: %s\t torso_height: %s"%(env.com, env.linvel, env.torso_height))
