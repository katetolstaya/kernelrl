from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import json
import copy


# ==================================================
# A POLK Q-Learning Model

class KNAF2Model(object):
    def __init__(self, stateCount, actionCount, config):
        self.dim_v = 1
        self.dim_p = actionCount
        self.dim_l = (1 + actionCount) * actionCount / 2
        self.dim_a = 1 + actionCount + (1 + actionCount) * actionCount / 2

        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))
        self.init_l = np.reshape(json.loads(config.get('InitL')), (-1, 1))

        self.v = KernelRepresentation(stateCount, self.dim_v, config)
        self.p = KernelRepresentation(stateCount, self.dim_p, config)
        self.l = KernelRepresentation(stateCount, self.dim_l, config)
        # Learning rate
        self.eta_v = ScheduledParameter('LearningRateV', config)
        self.eta_p = ScheduledParameter('LearningRateP', config)
        self.eta_l = ScheduledParameter('LearningRateL', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-6)
        # Representation error budget
        self.epsV = config.getfloat('RepresentationError', 1.0)
        self.epsP = config.getfloat('RepresentationError', 1.0)
        self.epsL = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.

        self.last_val = None
        self.last_s = None
        self.changed = True

    def get_q(self, s, a):
        lmat = self.get_lmat(s)
        pi = self.get_pi(s)
        return self.get_v(s) - 0.5 * ((a-pi).T).dot(lmat).dot(lmat.T).dot(a-pi)

    def get_v(self, s):
        return self.v(s.reshape(1, len(s))).flatten()

    def get_pi(self, s):
        pi = self.p(s.reshape(1, len(s))).flatten()
        return np.reshape(np.clip(pi, self.min_act, self.max_act), (-1,))

    def get_lmat(self, s):
        lmat = np.zeros((self.dim_p, self.dim_p))
        lmat[np.tril_indices(self.dim_p)] = self.l(s.reshape(1, len(s))).flatten()
        return lmat + self.init_l

    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.get_q(s, a)
        else:
            return r + self.gamma * self.get_v(s_) - self.get_q(s, a)

    def model_error(self):
        return 0.5 * self.lossL * self.v.normsq()

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')


    def train(self, step, sample):
        self.eta_v.step(step)
        self.eta_p.step(step)
        self.eta_l.step(step)
        self.beta.step(step)

        #min_eta = np.min([self.eta_v.value,self.eta_p.value,self.eta_l.value])
        # Unpack sample
        s, a, r, s_ = sample
        # Compute error
        delta = self.bellman_error(s, a, r, s_)
        # Gradient step

        self.v.shrink(1. - self.lossL)
        self.p.shrink(1. - self.lossL)
        self.l.shrink(1. - self.lossL)

        lmat = self.get_lmat(s)
        pi = self.get_pi(s)

        self.v.append(s, delta * self.eta_v.value)
        self.p.append(s, delta * self.eta_p.value * np.matmul(np.matmul(lmat, np.transpose(lmat)), a - pi))
        print delta * self.eta_p.value * np.matmul(np.matmul(lmat, np.transpose(lmat)), a - pi)

        # lmat grad
        lgrad_temp = np.matmul(np.matmul(np.transpose(lmat), a - pi), np.transpose(a - pi))
        if self.dim_p > 1:
            W = np.reshape(lgrad_temp[np.tril_indices(self.dim_p)], (-1, 1)) * self.eta_l.value
        else:
            W = lgrad_temp  * self.eta_l.value

        self.l.append(s, - delta * np.reshape(W, (1, -1)))

        # Prune
        self.v.prune(self.epsV*self.eta_v.value**2)
        self.p.prune(self.epsP*self.eta_p.value**2)
        self.l.prune(self.epsL*self.eta_l.value**2)

        print len(self.v.D)
        print len(self.p.D)
        print len(self.l.D)
        modelOrder_ = len(self.v.D) + len(self.p.D) + len(self.l.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_) ** 2  # + self.model_error()
        return (float(loss), float(modelOrder_))  # ==================================================

# An agent using Q-Learning

class KNAF2Agent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount

        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))
        self.model = KNAF2Model(self.stateCount, self.actionCount, config)
        self.act_mult = config.getfloat('ActMultiplier', 1)

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

    def act(self, s, stochastic=True):
        # "Decide what action to take in state s."

        if stochastic and (random.random() < self.epsilon.value):
            a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
        else:

            a = self.model.get_pi(s)
            #print a
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
