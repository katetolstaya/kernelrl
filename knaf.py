from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter

# ==================================================
# A POLK Q-Learning Model

class KNAFModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.dim_v = 1
        self.dim_p = actionCount
        self.dim_l = (1+actionCount)*actionCount/2
        self.dim_a = 1 + actionCount + (1+actionCount)*actionCount/2

        self.vpl = KernelRepresentation(stateCount, self.dim_a, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        self.phi = config.getfloat('Phi', 0.0)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.

    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.predictOne(s)[a]
        else:
            return r + self.gamma*self.predictOne(s_).max() - self.predictOne(s)[a]


    def model_error(self):
        return 0.5*self.lossL*self.Q.normsq()

    def predict(self, s):
        "Predict the Q function values for a batch of states."
        return self.vpl(s)

    def predictOne(self, s):
        "Predict the Q function values for a single state."
        return self.vpl(s.reshape(1, len(s))).flatten()
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order')

    def train(self, step, sample):
        self.eta.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_ = sample
        # Compute error
        delta = self.bellman_error(s,a,r,s_)
        # Running average of TD-error
        self.y += self.beta.value * (delta - self.y)
        # Gradient step
        self.vpl.shrink(1. - self.eta.value * self.lossL)
        if s_ is None:
            W = np.zeros((1, self.vpl.W.shape[1]))
            W[0,a] = -1.
            self.vpl.append(s, -self.eta.value * self.y * W)
        else:
            a_ = self.predictOne(s_).argmax()
            W = np.zeros((2, self.Q.W.shape[1]))
            W[0,a]  = -1.
            W[1,a_] = 0 #self.gamma
            self.Q.append(np.vstack((s,s_)), -self.eta.value * self.y * W)
        # Prune
        modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value**2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

# ==================================================
# An agent using Q-Learning

class KNAFAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount

        self.min_act = np.reshape(self.min_act, (-1, 1))
        self.max_act = np.reshape(self.max_act, (-1, 1))

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'TD')

        self.model = KNAFModel(self.stateCount, self.actionCount, config)

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
            return random.randint(0, self.actionCount-1)
        else:
            return self.model.predictOne(s).argmax()
    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)
    def improve(self):
        return self.model.train(self.steps, self.lastSample)
    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s,a,r,s_)
    def model_error(self):
        return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names


