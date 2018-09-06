import random

import numpy as np

from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
import json

# ==================================================
# A POLK Q-Learning Model\

def assemble(s, a):
    return np.concatenate((s.reshape((1, -1)), a.reshape((1, -1))), axis=1).flatten()

class KNAFIIDModel(object):
    def __init__(self, stateCount, actionCount, config):

        # Get dimensions of V, pi and L
        self.dim_v = 1
        self.dim_p = actionCount
        self.dim_l = 1 #(1 + actionCount) * actionCount / 2 #TODO
        self.dim_a = self.dim_v + self.dim_p + self.dim_l

        # Get action space
        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        # Initialize L
        self.init_l = config.getfloat('InitL', 0.01)

        # Represent V, pi, L in one RKHS
        self.vpl = KernelRepresentation(stateCount, self.dim_a, config)

        # Learning rates
        self.eta_v = ScheduledParameter('LearningRateV', config)
        self.eta_p = ScheduledParameter('LearningRateP', config)
        self.eta_l = ScheduledParameter('LearningRateL', config)
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        #self.phi = config.getfloat('Phi', 1)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

    def get_q(self, s, a):
        lmat = self.get_lmat(s)
        pi = self.get_pi(s)
        return np.array([self.get_v(s) - 0.5 * (a - pi)*lmat*lmat*(a - pi)])

    def get_v(self, s):
        return np.array([self.predictOne(s)[0,0]])

    def get_pi(self, s):
        pi = self.predictOne(s)[0,1:self.dim_p + 1]
        return np.reshape(np.clip(pi, self.min_act, self.max_act), (-1,))

    def get_lmat(self, s):
        lmat = np.zeros((self.dim_p, self.dim_p))
        temp = self.predictOne(s)
        if self.dim_p > 1:
            lmat[np.tril_indices(self.dim_p)] = temp[self.dim_p + 1:]
            return lmat + self.init_l * np.eye(self.dim_p)
        else:
            return np.array([temp[0,2] + self.init_l])

    def train(self, step, sample):
        self.eta_v.step(step)
        self.eta_p.step(step)
        self.eta_l.step(step)

        s,a,r,s_ = sample[0][1][0],sample[0][1][1],sample[0][1][2],sample[0][1][3]
        delta = self.bellman_error(s, a, r, s_)

        # Gradient step
        self.vpl.shrink(1. - self.lossL)

        W = np.zeros((self.dim_a,))
        W[0] = -1 * self.eta_v.value
        lmat = self.get_lmat(s)
        pi = self.get_pi(s)

        lgrad_temp = lmat * (a - pi) * (a - pi)
        W[1] = -self.eta_p.value * lmat * lmat * (a - pi)
        W[-1] = lgrad_temp * self.eta_l.value
        self.vpl.append(np.array(s), - delta * np.reshape(W, (1, -1)))
        self.vpl.prune(self.eps)
        modelOrder_ = len(self.vpl.D)
        loss = 0.5 * self.bellman_error(s, a, r, s_) ** 2  # + self.model_error()
        return (float(loss), float(modelOrder_))

    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.get_q(s, a)
        else:
            return r + self.gamma * self.get_v(s_) - self.get_q(s, a)

    def predict(self, s):  # Predict the Q function values for a batch of states.
        return self.vpl(s)

    def predictOne(self, s):  # Predict the Q function values for a single state.
        return self.vpl(np.reshape(s,(1,-1)))

    def model_error(self):
        return 0.5 * self.lossL * self.vpl.normsq()

# ==================================================
# An agent using Q-Learning

class KNAFIIDAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high

        self.memory = None

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

        # ---- Configure batch size
        self.batchSize = config.getint('MinibatchSize', 1)
        self.model = KNAFIIDModel(self.stateCount, self.actionCount, config)

        # How many steps we have observed
        self.steps = 0
        # ---- Configure exploration
        self.epsilon = ScheduledParameter('ExplorationRate', config)
        self.epsilon.step(0)
        # ---- Configure rewards
        self.gamma = config.getfloat('RewardDiscount')
        # ---- Configure priority experience replay

        self.eps = config.getfloat('ExperiencePriorityMinimum', 0.01)
        self.alpha = config.getfloat('ExperiencePriorityExponent', 1.)
        self.noise_var = ScheduledParameter('NoiseVariance', config)
        self.noise_var.step(0)

    def act(self, s, stochastic=True):
        # "Decide what action to take in state s."
        a = self.model.get_pi(s)
        if stochastic: # if exploration, add noise
            a = a + np.random.normal(0,self.noise_var.value,self.actionCount)
        a = np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))
        return a

    def observe(self, sample):
        error = self.model.bellman_error(sample[0],sample[1],sample[2],sample[3])
        self.memory.add(sample, np.abs((error[0] + self.eps) ** self.alpha))
        self.steps += 1
        self.epsilon.step(self.steps)
        self.noise_var.step(self.steps)

    def improve(self):


        sample = self.memory.sample(1)

        s,a,r,s_ = sample[0][1][0],sample[0][1][1],sample[0][1][2],sample[0][1][3]

        # compute bellman error
        #error = self.bellman_error(s,a,r,s_)

        # update model
        self.model.train(self.steps, sample)

        # compute updated error
        error = self.model.bellman_error(s, a, r, s_)

        # compute our average minibatch loss
        loss = 0.5 * np.mean(error ** 2) + self.model.model_error()
        # compute our model order
        modelOrder = len(self.model.vpl.D)
        # report metrics
        return (float(loss), float(modelOrder))

    def model_error(self):
        return self.model.model_error()
    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s, a, r, s_)

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')