import random

import numpy as np

from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation


# ==================================================
# A POLK SARSA Model

class KSARSAModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount, actionCount, config)
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        # Running estimate of our expected TD-loss
        self.y = 0.

    def bellman_error(self, s, a, r, s_, a_):
        if s_ is None:
            return r - self.predictOne(s)[a]
        else:
            return r + self.gamma * self.predictOne(s_)[a_] - self.predictOne(s)[a]

    def model_error(self):
        return 0.5 * self.lossL * self.Q.normsq()

    def predict(self, s):
        "Predict the Q function values for a batch of states."
        return self.Q(s)

    def predictOne(self, s):
        "Predict the Q function values for a single state."
        return self.Q(s.reshape(1, len(s))).flatten()

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')


class KSARSAModelTD(KSARSAModel):
    def train(self, step, sample):
        self.eta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
        # Compute error
        delta = self.bellman_error(s, a, r, s_, a_)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)
        W = np.zeros((1, self.Q.W.shape[1]))
        W[0, a] = -1.
        self.Q.append(s, -self.eta.value * delta * W)
        # Prune
        modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value ** 2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_, a_) ** 2 + self.model_error()
        return (float(loss), float(modelOrder_))


class KSARSAModelSCGD(KSARSAModel):
    def __init__(self, stateCount, actionCount, config):
        super(KSARSAModelSCGD, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.

    def train(self, step, sample):
        self.eta.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
        # Compute error
        delta = self.bellman_error(s, a, r, s_, a_)
        # Running average
        self.y += self.beta.value * (delta - self.y)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)
        if s_ is None:
            W = np.zeros((1, self.Q.W.shape[1]))
            W[0, a] = -1.
            self.Q.append(s, -self.eta.value * self.y * W)
        else:
            W = np.zeros((2, self.Q.W.shape[1]))
            W[0, a] = -1.
            W[1, a_] = self.gamma
            self.Q.append(np.vstack((s, s_)), -self.eta.value * self.y * W)
        # Prune
        # modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value ** 2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_, a_) ** 2 + self.model_error()
        return (float(loss), float(modelOrder_))


# ==================================================
# An agent using SARSA

class KSARSAAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'TD')
        if algorithm.lower() == 'scgd':
            self.model = KSARSAModelSCGD(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = KSARSAModelTD(self.stateCount, self.actionCount, config)
        else:
            raise ValueError('Unknown algorithm: {}'.format(algorithm))
        # How many steps we have observed
        self.steps = 0
        # ---- Configure exploration
        self.epsilon = ScheduledParameter('ExplorationRate', config)
        self.epsilon.step(0)
        # ---- Configure rewards
        self.gamma = config.getfloat('RewardDiscount')

    def act(self, s, stochastic=True):
        "Decide what action to take in state s."
        if stochastic and (random.random() < self.epsilon.value):
            return random.randint(0, self.actionCount - 1)
        else:
            return self.model.predictOne(s).argmax()

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
        return self.model.train(self.steps, self.lastSample)

    def bellman_error(self, s, a, r, s_, a_):
        return self.model.bellman_error(s, a, r, s_, a_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
