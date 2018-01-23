import math


# ==================================================
# Scheduled Parameter Model
# ------------------------------
# Models a number that can decrease with each step.
# There are three different models available:
#  1) Constant
#     If value is given as a number, it will be assumed constant
#  2) Exponential Decay
#     Value is given as 'ExponentialDecay'
#  3) Power Decay
#     Value is given as 'PowerDecay'
#
# In the decaying models, we decay from an initial value to a final value.
#   The 'Initial{name}' parameter controls where the parameter starts.
#   The 'Final{name}' parameter controls where the parameter decays towards.
#
# The decay rate can be specified in one of two ways:
#   a) '{name}Decay' will use the given value as the rate
#   b) '{name}Stop' will calculate the rate to decrease the parameter most of the way
#      to the final value at the given step. Most of the way means 1/100 of the initial
#      value remains, or the amount given by '{name}Margin'
# ==================================================

class Parameter(object):
    def __init__(self, label, config):
        pass

    def step(self, t):
        pass

    @property
    def value(self):
        return self._value


class ConstantParameter(Parameter):
    @classmethod
    def matches(cls, label, config):
        try:
            config.getfloat(label)
            return True
        except ValueError:
            return False

    def __init__(self, label, config):
        self._value = config.getfloat(label)


class ExponentialDecayParameter(Parameter):
    @classmethod
    def matches(cls, label, config):
        return config.get(label).lower() == 'exponentialdecay'

    def __init__(self, label, config):
        initial = config.getfloat('Initial' + label)
        final = config.getfloat('Final' + label, 0.)
        self.a = final
        self.b = initial - final
        if label + 'Decay' in config:
            self.rate = config.getfloat(label + 'Decay')
        elif label + 'Stop' in config:
            stop = config.getint(label + 'Stop')
            margin = config.getfloat(label + 'Margin', 0.01)
            self.rate = -math.log(margin) / float(stop)
        else:
            msg = "Must specifiy '{}Decay' or '{}Stop'".format(label, label)
            raise ValueError(msg)

    def step(self, t):
        self._value = self.a + self.b * math.exp(-self.rate * t)


class ConstExponentialDecayParameter(Parameter):
    @classmethod
    def matches(cls, label, config):
        return config.get(label).lower() == 'constexponentialdecay'

    def __init__(self, label, config):
        self.initial = config.getfloat('Initial' + label)
        self.final = config.getfloat('Final' + label, 0.)

        self.a = self.final
        self.b = self.initial - self.final

        self.start = config.getint(label + 'Start')
        self.stop = config.getint(label + 'Stop')
        margin = config.getfloat(label + 'Margin', 0.01)

        self.rate = -math.log(margin) / float(self.stop - self.start)

    def step(self, t):
        if t < self.start:
            self._value = self.initial
        elif t > self.stop:
            self._value = self.final
        else:
            self._value = self.a + self.b * math.exp(-self.rate * (t - self.start))


class PowerDecayParameter(Parameter):
    @classmethod
    def matches(cls, label, config):
        return config.get(label).lower() == 'powerdecay'

    def __init__(self, label, config):
        initial = config.getfloat('Initial' + label)
        final = config.getfloat('Final' + label, 0.)
        self.a = final
        self.b = initial - final
        if label + 'Decay' in config:
            self.rate = config.getfloat(label + 'Decay')
        elif label + 'Stop' in config:
            stop = config.getint(label + 'Stop')
            margin = config.getfloat(label + 'Margin', 0.01)
            self.rate = -math.log(margin) / math.log(float(stop) + 1)
        else:
            msg = "Must specifiy '{}Decay' or '{}Stop'".format(label, label)
            raise ValueError(msg)

    def step(self, t):
        self._value = self.a + self.b * (1. / (t + 1) ** self.rate)


class ConstPowerDecayParameter(Parameter):
    @classmethod
    def matches(cls, label, config):
        return config.get(label).lower() == 'constpowerdecay'

    def __init__(self, label, config):
        self.initial = config.getfloat('Initial' + label)
        self.final = config.getfloat('Final' + label, 0.)
        self.a = self.final
        self.b = self.initial - self.final

        self.start = config.getint(label + 'Start')
        self.stop = config.getint(label + 'Stop')
        margin = config.getfloat(label + 'Margin', 0.01)

        self.rate = -math.log(margin) / math.log(float(self.stop - self.start) + 1)

    def step(self, t):
        if t < self.start:
            self._value = self.initial
        elif t > self.stop:
            self._value = self.final
        else:
            self._value = self.a + self.b * (1. / (t - self.start + 1) ** self.rate)


class LinearParameter(Parameter):
    @classmethod
    def matches(cls, label, config):
        return config.get(label).lower() == 'linear'

    def __init__(self, label, config):
        self.initial = config.getfloat('Initial' + label)
        self.final = config.getfloat('Final' + label, 0.)
        self.a = self.final
        self.b = self.initial - self.final

        self.start = config.getint(label + 'Start')
        self.stop = config.getint(label + 'Stop')

        self.rate = (self.b) / float(self.stop - self.start)

    def step(self, t):
        if t < self.start:
            self._value = self.initial
        elif t > self.stop:
            self._value = self.final
        else:
            self._value = self.a + self.b * (1 - (t - self.start) * self.rate)


# ==================================================
# Factory function for Parameters
# ==================================================
def ScheduledParameter(label, config):
    # if label not in config:
    #    raise ValueError("'{}' not specifed".format(label))
    for cls in Parameter.__subclasses__():
        if cls.matches(label, config):
            return cls(label, config)
    raise ValueError("Invalid value for '{}': {}".format(label, config.get(label)))
