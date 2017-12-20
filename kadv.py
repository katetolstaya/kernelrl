from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import pickle
import json


# ==================================================
# A POLK Q-Learning Model

class KAdvModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.V = KernelRepresentation(stateCount, 1, config)
        self.A = KernelRepresentation(stateCount + actionCount, 1, config)
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        self.phi = config.getfloat('Phi', 0.0)
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.

    def eval_q(self,s,a):

        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        #print "A"
        #print self.A(x)
        #print self.V(s)
        return self.A(x) + self.V(s)

    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.eval_q(s,a)
        else:
            a_ = self.A.argmax(s_)
            return r + self.gamma * self.eval_q(s_,a_) - self.eval_q(s,a)

    def bellman_error2(self, s, a, r, s_,a_):
        if s_ is None:
            return r - self.eval_q(s,a)
        else:
            return r + self.gamma * self.eval_q(s_,a_) - self.eval_q(s,a)

    def model_error(self):
        return 0.5 * self.lossL * self.A.normsq()

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
            a_ = None
        else:
            a_ = self.A.argmax(s_)
        delta = self.bellman_error2(s,a, r, s_, a_)

        # Running average of TD-error
        self.y += self.beta.value * (delta - self.y)

        # Gradient step
        self.A.shrink(1. - self.eta.value * self.lossL)
        self.V.shrink(1. - self.eta.value * self.lossL)

        if s_ is None:
            self.A.append(x, self.eta.value * self.y)
        else:
            self.A.append(x, self.eta.value * self.y)
            self.V.append(s_, - self.eta.value * self.y * self.gamma)

        # Prune
        self.A.prune(self.eps ** 2 * self.eta.value ** 2 / self.beta.value)
        self.V.prune(self.eps ** 2 * self.eta.value ** 2 / self.beta.value)

        modelOrder_ = self.A.model_order() + self.V.model_order()

        #print self.A.model_order()
        #print self.V.model_order()

        # Compute new error
        loss = 0.5 * self.bellman_error2(s, a, r, s_,a_) ** 2 + self.model_error() # TODO should we have model error here?
        # print modelOrder_
        return (float(loss), float(modelOrder_))


# ==================================================
# An agent using Q-Learning

class KAdvAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount

        self.model = KAdvModel(self.stateCount, self.actionCount, config)

        # self.save_steps = config.getint('SaveInterval', 1000000000)
        # self.folder = config.get('Folder', 'exp')

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
            # return self.action_space.sample()
        else:
            a = self.model.A.argmax(s)

        a_temp = np.reshape(np.clip(a, self.min_act, self.max_act),(-1,))
        return a_temp

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
        #if len(self.model.Q.D) > self.max_model_order:
        #    self.model.eps = self.model.eps * 2
        #    # self.model.eps = self.model.eps * 1
        #if self.steps % self.save_steps == 0:
        #    with open(self.folder + '/kpolicy_model_' + str(int(self.steps / self.save_steps)) + '.pkl', 'wb') as f:
        #        pickle.dump(self.model.Q, f)

        return self.model.train(self.steps, self.lastSample)

    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s, a, r, s_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
