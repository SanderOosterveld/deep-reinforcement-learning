class BaseEpsilonFunction:

    def __init__(self):
        self.counter = 0

    @property
    def epsilon(self):
        self.counter += 1
        return self._get_epsilon()

    def __call__(self):
        return self.epsilon

    def _get_epsilon(self):
        raise NotImplementedError

    def reset(self):
        self.counter = 0


class ConstantEpsilon(BaseEpsilonFunction):

    def __init__(self, value):
        super(ConstantEpsilon, self).__init__()

        self._value = value

    def _get_epsilon(self):
        return self._value

    def __str__(self):
        return "<ConstantEpsilon(%f)>" % self._value


class LinearEpsilon(BaseEpsilonFunction):

    def __init__(self, start, end, steps):
        super(LinearEpsilon, self).__init__()

        # Use a function in the form Ax + b. Where x is the step number
        self.A = (end - start) / steps
        self.B = start

    def _get_epsilon(self):
        return self.A * self.counter + self.B

    def __str__(self):
        return "<LinearEpsilon() with %f x + %f>" % (self.A, self.B)


class ExponentialEpsilon(BaseEpsilonFunction):

    def __init__(self, initial, decay_rate):
        super(ExponentialEpsilon, self).__init__()
        assert decay_rate < 1, "Decay rate must be below 1 else epsilon explodes"
        self._decay_rate = decay_rate
        self._initial = initial

    def _get_epsilon(self):
        return self._decay_rate ** self.counter * self._initial

    def __str__(self):
        return "ExponentialEpsilon() with %f * %f^x" % (self._initial, self._decay_rate)
