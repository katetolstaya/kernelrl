import json
import random

import numpy as np

from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation


# ==================================================
# A POLK Q-Learning Model

class KQLearningContModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount + actionCount, 1, config)

        # Learning rates
        self.eta = ScheduledParameter('LearningRate', config)
        self.beta = ScheduledParameter('ExpectationRate', config)
        self.algorithm = config.get('Algorithm', 'td').lower()  # gtd, td or hybrid

        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)

        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

        # Multiplier (see Baird paper)
        # self.phi = config.getfloat('Phi', 1.0)

        # Running estimate of our expected TD-loss
        self.y = 0.

        self.rand_act = True

    def bellman_error(self, s, a, r, s_):
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            a_ = self.Q.argmax(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            return r + self.gamma * self.Q(x_) - self.Q(x)

    def bellman_error_helper(self, x, r, x_):
        if x_ is None:
            return r - self.Q(x)
        else:
            return r + self.gamma * self.Q(x_) - self.Q(x)

    def model_error(self):
        return 0.5 * self.lossL * self.Q.normsq()

    def predict(self, s):
        pass

    def predictOne(self, s):
        pass

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')

    def train(self, step, sample):
        self.eta.step(step)
        self.beta.step(step)

        # Unpack sample and compute error
        s, a, r, s_ = sample
        x = np.concatenate((np.reshape(np.array(s), (1, -1)), np.reshape(np.array(a), (1, -1))), axis=1)
        if s_ is None:
            x_ = None
        else:
            a_ = self.Q.argmax(s_)
            x_ = np.concatenate((np.reshape(np.array(s_), (1, -1)), np.reshape(np.array(a_), (1, -1))), axis=1)
        delta = self.bellman_error_helper(x, r, x_)

        # Running average of TD-error
        self.y += self.beta.value * (delta - self.y)

        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)

        if self.algorithm == 'td' or (self.algorithm == 'hybrid' and not self.rand_act):
            self.Q.append(x, self.eta.value * self.y)
        elif self.algorithm == 'gtd' or (self.algorithm == 'hybrid' and self.rand_act):
            if s_ is None:
                self.Q.append(x, self.eta.value * self.y)
            else:
                W = np.zeros((2, 1))
                W[0] = -1.
                W[1] = self.gamma  # * self.phi
                self.Q.append(np.vstack((x, x_)), -self.eta.value * self.y * W)
        else:
            raise ValueError('Unknown algorithm: {}'.format(self.algorithm))

        # Prune
        self.Q.prune(self.eps ** 2 * self.eta.value ** 2 / self.beta.value)
        modelOrder_ = self.Q.model_order()

        # Compute new error
        loss = 0.5 * self.bellman_error_helper(x, r, x_) ** 2 + self.model_error()  # TODO should we have model error here?

        return (float(loss), float(modelOrder_))


# ==================================================
# An agent using Q-Learning
class KQLearningContAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount

        # ---- Configure exploration
        self.epsilon = ScheduledParameter('ExplorationRate', config)
        self.epsilon.step(0)
        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        # How many steps we have observed
        self.steps = 0
        self.lastSample = None

        # Initialize model
        self.model = KQLearningContModel(self.stateCount, self.actionCount, config)

    def act(self, s, stochastic=True):
        # "Decide what action to take in state s."
        if stochastic and (random.random() < self.epsilon.value):
            a = np.random.uniform(self.min_act, self.max_act)
            self.model.rand_act = True
        else:
            a = self.model.Q.argmax(s)
            self.model.rand_act = False

        return np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
        return self.model.train(self.steps, self.lastSample)

    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s, a, r, s_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
