from .simulator import _Simulator


import numpy as np
import math

from scipy.integrate import odeint


def get_angle_in_range(angle, range):
    total_range = range[1] - range[0]
    temp_angle = angle - range[0]
    temp_angle = temp_angle % total_range
    final_angle = temp_angle + range[0]
    return final_angle


class PendulumParameters:

    def __init__(self, mass=1, length=1, damping=0.05, g=9.81):
        self.m = mass
        self.L = length
        self.b = damping
        self.g = g
        self.J = self.m * self.L ** 2

    def __str__(self):
        return str(self.__dict__)


class PendulumSimulator(_Simulator):

    def __init__(self,parameters=PendulumParameters(), dt=0.05):
        super(self.__class__, self).__init__(dt=dt)
        self.par = parameters
        self._controls = 0
        self._state = [0, 0]

        self._time = 0

    def _model(self, z, t):
        theta = z[0]
        omega = z[1]
        dthetadt = omega
        domegadt = 1 / self.par.J * (
                -self.par.b * omega - (math.sin(theta)) * self.par.g * self.par.L * self.par.m + self._controls)
        dzdt = [dthetadt, domegadt]
        return dzdt

    def compute_max_torque(self, max_angle):
        max_torque = self.par.m * self.par.g * self.par.L * math.sin(max_angle / 180 * math.pi)
        return max_torque

    def _step(self, state: np.ndarray, action):
        self._controls = action
        z0 = state
        t = np.asarray([0, self._dt])
        z = odeint(self._model, z0, t)
        self._time += self._dt
        return z[1, :]

    def do_simulation(self, controls):
        self._state = self._step(np.asarray(self._state), controls)
        return self.state

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def get_real_angle(self):
        return get_angle_in_range(self.state[0], (0, 2 * math.pi))

    def get_jasper_angle(self):
        old_angle = self._state[0]
        self._state[0] += math.pi
        jasper_angle = self.get_real_angle()
        self._state[0] = old_angle
        return jasper_angle

    def _compute_energy(self):
        kinetic_energy = 0.5*self.par.J*self.state[1]**2
        height = self.par.L*(1-math.cos(self.state[0]))
        potential_energy = height*self.par.g*self.par.m
        return potential_energy, kinetic_energy

    def reset(self):
        self._time = 0
        self._state = [0, 0]

    @property
    def time(self):
        return self._time

#
# class PendulumViewer:











