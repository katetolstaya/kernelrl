from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import copy
import pickle


# ==================================================
# A POLK Policy improvements Model

class KPolicyModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount + actionCount, 1, config)
        self.Q2 = KernelRepresentation(stateCount + actionCount, 1, config)
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
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            return r + self.Q(x_) - self.Q(x)

    def model_error(self):
        return 0.5 * self.lossL * self.Q.normsq()

    def predict(self, s):
        pass

    # "Predict the Q function values for a batch of states."
    # return self.Q(s)
    def predictOne(self, s):
        pass

    # "Predict the Q function values for a single state."
    # return self.Q(s.reshape(1, len(s))).flatten()
    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')


class KPolicyModelTD(KPolicyModel):
    def train(self, step, sample):
        self.eta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample

        # Compute error
        delta = self.bellman_error(s, a, r, s_, a_)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        self.Q.append(x, self.eta.value * delta)
        # Prune
        modelOrder = len(self.Q.D)
        # self.Q.prune(self.eps * self.eta.value**2)

        # if self.steps % self.prune_steps == 0:
        #	self.Q.prune(self.eps * self.eta.value**2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_, a_) ** 2 + self.model_error()
        return (float(loss), float(modelOrder_))


class KPolicyModelSCGD(KPolicyModel):
    def __init__(self, stateCount, actionCount, config):
        super(KPolicyModelSCGD, self).__init__(stateCount, actionCount, config)
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

        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            self.Q.append(x, self.eta.value * self.y)
        else:
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            W = np.zeros((2, 1))
            W[0] = -1.
            W[1] = self.gamma
            self.Q.append(np.vstack((x, x_)), -self.eta.value * self.y * W)
        # Prune
        modelOrder = len(self.Q.D)
        # if self.steps % self.prune_steps == 0:
        self.Q.prune(self.eps * self.eta.value ** 2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_, a_) ** 2 + self.model_error()
        return (float(loss), float(modelOrder_))


# ==================================================
# An agent using policy improvement

class KPolicyAgent(object):
    def __init__(self, env, config):
        self.config = config
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.max_act = 2
        self.avg_steps = 1000  # config.get('ErrorAveragingSteps', 1000)
        self.sarsa_steps = config.getint('SARSASteps', 100000)
        self.folder = config.get('Folder', 'exp')
        self.prune_steps = 100

        self.last_avg_error = 0
        self.avg_error = 0

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'SCGD')
        if algorithm.lower() == 'scgd':
            self.model = KPolicyModelSCGD(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = KPolicyModelTD(self.stateCount, self.actionCount, config)
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
            return np.random.uniform(-self.max_act, self.max_act, (self.actionCount, 1))
        # return random.randint(0, self.actionCount-1)
        else:
            action = self.model.Q2.argmax(s)
            return action
            # return self.model.predictOne(s).argmax()

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        # self.epsilon.step(self.steps)

    def improve(self):
        loss, modelOrder = self.model.train(self.steps, self.lastSample)
        # self.avg_error = abs(self.avg_error) + loss

        # if self.steps % self.avg_steps == 0:
        #   if self.avg_error != 0 and abs((self.last_avg_error - self.avg_error)) <= self.last_avg_error * 0.1:
        #	self.epsilon.step(self.steps)
        #	self.model.Q2 = copy.deepcopy(self.model.Q)
        #	self.model.Q = KernelRepresentation(self.stateCount+self.actionCount,1, self.config)
        #   	self.avg_error = 0

        #   self.last_avg_error = self.avg_error
        #   self.avg_error = 0
        # if self.steps % self.prune_steps == 0:
        #	self.model.Q.prune(self.model.eps * self.model.eta.value**2)


        if self.steps % self.sarsa_steps == 0:
            self.epsilon.step(self.steps)
            with open(self.folder + '/kpolicy_model_' + str(int(self.steps / self.sarsa_steps)) + '.pkl', 'wb') as f:
                pickle.dump(self.model.Q, f)

            self.model.Q2 = copy.deepcopy(self.model.Q)
            self.model.Q = KernelRepresentation(self.stateCount + self.actionCount, 1, self.config)

        return loss, modelOrder

    def bellman_error(self, s, a, r, s_, a_):
        return self.model.bellman_error(s, a, r, s_, a_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
