import numpy as np

from .simulator import _Simulator
from utils.functions import approx_zero


import mujoco_py

import typing
class MuJoCoSimulator(_Simulator):

    def __init__(self, xml_path):
        super(MuJoCoSimulator, self).__init__()
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.sim.forward()
        self.skip_frames = 1
        self.data = self.sim.data

        self.pre_sim_functions = []
        self.post_sim_functions = []

    def set_dt(self, dt):
        assert approx_zero(dt%0.002), "dt must be a fraction of the %f(default simulation dt)"%self.model.opt.timestep
        self.skip_frames = int(dt/self.model.opt.timestep)
        return self

    def _step(self):
        for _ in range(self.skip_frames):
            self.sim.step()

    def _actuate(self, control: typing.Sequence):
        assert len(control)==self.n_actuators, "Control must have dim of actuators : %i, is %i"%(self.n_actuators, len(control))

        self.data.ctrl[:] = control[:]

    def do_simulation(self, controls:  typing.Sequence):
        self.pre_sim()

        self._actuate(controls)
        self._step()
        self.post_sim()
        return self.state

    def pre_sim(self):
        pass

    def post_sim(self):
        pass

    def get_qpos_qvel(self):
        states = np.zeros((self.sim.data.qpos.shape[0], 2))
        states[:, 0] = self.sim.data.qpos
        states[:, 1] = self.sim.data.qvel
        return states

    def set_state(self, value):
        qpos = value[::2]
        qvel = value[1::2]
        self.set_qpos_qvel(qpos, qvel)

    def get_state(self):
        return self.get_qpos_qvel().reshape(-1)

    def set_qpos_qvel(self, qpos, qvel):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset(self):
        self.sim.reset()
        self.sim.forward()

    @property
    def n_bodies(self):
        return self.sim.data.body_xpos.shape[0]-1

    @property
    def dt(self):
        return self.skip_frames*self.model.opt.timestep

    @dt.setter
    def dt(self, dt):
        self.set_dt(dt)

    @property
    def n_actuators(self):
        return self.model.nu

    @property
    def sensordata(self):
        return self.sim.data.sensordata

    @property
    def controls(self):
        return self.data.ctrl

    @property
    def time(self):
        return self.data.time

    def disable_gravity(self):
        self.model.opt.gravity[2] = 0

    @property
    def joint_limits(self):
        return self.model.jnt_range