from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter

# ==================================================
# A POLK Q-Learning Model

class KQLearningModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount, actionCount, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.predictOne(s)[a]
        else:
            return r + self.gamma*self.predictOne(s_).max() - self.predictOne(s)[a]
    def model_error(self):
        return 0.5*self.lossL*self.Q.normsq()
    def train(self, step, sample):
        pass
    def predict(self, s):
        "Predict the Q function values for a batch of states."
        return self.Q(s)
    def predictOne(self, s):
        "Predict the Q function values for a single state."
        return self.Q(s.reshape(1, len(s))).flatten()
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order')

class KQLearningModelTD(KQLearningModel):
    def train(self, step, sample):
        self.eta.step(step)
        # Unpack sample
        s, a, r, s_ = sample
        # Compute error
        delta = self.bellman_error(s,a,r,s_)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)

        x = np.concatenate((s, a))
        W = np.zeros((2, 1))
        W[0] = -1
        W[1] = self.gamma
        self.Q.append(np.vstack((x, x_)), -self.eta.value * delta * W)
        self.Q.append(x, -self.eta.value * delta * W)


        W = np.zeros((1, self.Q.W.shape[1]))
        W[0,a] = -1.

        self.Q.append(s, -self.eta.value * delta * W)
        # Prune
        modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value**2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

class KQLearningModelSCGD(KQLearningModel):
    def __init__(self, stateCount, actionCount, config):
        super(KQLearningModelSCGD, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.
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
        self.Q.shrink(1. - self.eta.value * self.lossL)
        if s_ is None:
            W = np.zeros((1, self.Q.W.shape[1]))
            W[0,a] = -1.
            self.Q.append(s, -self.eta.value * self.y * W)
        else:
            a_ = self.predictOne(s_).argmax()
            W = np.zeros((2, self.Q.W.shape[1]))
            W[0,a]  = -1.
            W[1,a_] = self.gamma
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

class KQLearningAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'TD')
        if algorithm.lower() == 'scgd':
            self.model = KQLearningModelSCGD(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = KQLearningModelTD(self.stateCount, self.actionCount, config)
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


