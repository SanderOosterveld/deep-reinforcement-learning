class _Simulator:

    def __init__(self, dt=0.05):
        self._dt = dt
        self._controls = []
        self._state = []

    def step(self, state, controls):
        """
        Almost the same as do_simulation however also takes the current state as an input. Simply runs do_simulation
        after setting the current state to the values as given as the input
        :param state:
        :param controls:
        :return:
        """
        self._controls = controls
        self._state = state
        self._state = self.do_simulation(controls)
        return self._state


    def do_simulation(self, controls):
        """
        Do the simulation this is based on the current state.
        :return:
        """
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def _compute_energy(self):
        """
        Energy of the system. Note that the potential energy is with respect to the 0-state so when all the states are
        zero.
        :return: Potential energy w.r.t. 0, kinetic energy
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def state(self):
        return self.get_state()

    @state.setter
    def state(self, value):
        self.set_state(value)

    @property
    def energy(self):
        return self._compute_energy()

    @property
    def time(self):
        raise NotImplementedError




