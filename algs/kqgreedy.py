import json
import random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation


# ==================================================
# A POLK Q-Learning Model

class KGreedyQModel(object):
    def __init__(self, stateCount, actionCount, config):

        self.Q = KernelRepresentation(stateCount + actionCount, 2, config)
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.

    def get_q(self,x):
        return self.Q(x)[0][0]

    def get_g(self, x):
        return self.Q(x)[0][1]

    def bellman_error(self, s, a, r, s_):
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r - self.get_q(x)
        else:
            a_ = self.Q.argmax(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            return r + self.gamma * self.get_q(x_) - self.get_q(x)

    def bellman_error2(self, x, r, x_):
        if x_ is None:
            return r - self.get_q(x)
        else:
            return r + self.gamma * self.get_q(x_) - self.get_q(x)

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
            x_ = np.concatenate((np.reshape(np.array(s_), (1, -1)), np.reshape(np.array(a_), (1, -1))),axis=1)

        delta = self.bellman_error2(x, r, x_)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)

        if s_ is None:
            W = np.zeros((1, 2))
            W[0,0] = self.eta.value * delta
            W[0,1] = self.beta.value * (delta - self.get_g(x))
            self.Q.append(x, W)
        else:
            W = np.zeros((2, 2))
            W[0,0] = self.eta.value * delta
            W[1,0] = - self.eta.value * self.gamma * self.get_g(x)
            W[0,1] = self.beta.value * (delta - self.get_g(x))
            self.Q.append(np.vstack((x, x_)), W)

        # Prune
        self.Q.prune(self.eps ** 2 * self.eta.value ** 2 / self.beta.value)
        modelOrder_ = self.Q.model_order()
        #print modelOrder_
        # Compute new error
        loss = 0.5 * self.bellman_error2(x, r, x_) ** 2 + self.model_error() # TODO should we have model error here?
        # print modelOrder_
        return (float(loss), float(modelOrder_))


# ==================================================
# An agent using Q-Learning

class KGreedyQAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.model = KGreedyQModel(self.stateCount, self.actionCount, config)

        # How many steps we have observed
        self.steps = 0

        # ---- Configure exploration
        self.epsilon = ScheduledParameter('ExplorationRate', config)
        self.epsilon.step(0)
        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))
        self.max_model_order = config.getfloat('MaxModelOrder', 10000)
        self.act_mult = config.getfloat('ActMultiplier', 1)

        self.lastSample = None

    def act(self, s, stochastic=True):
        # "Decide what action to take in state s."
        if stochastic and (random.random() < self.epsilon.value):
            a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
        else:
            a = self.model.Q.argmax(s)
        a_temp = np.reshape(np.clip(a, self.min_act, self.max_act),(-1,))
        return a_temp

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
