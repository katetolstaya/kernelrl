import random

import numpy as np

from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation


# ==================================================
# A POLK Q-Learning Model\

def assemble(s, a):
    return np.concatenate((s.reshape((1, -1)), a.reshape((1, -1))), axis=1).flatten()

class KQLearningModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.algorithm = config.get('Algorithm', 'td').lower() # gtd, td or hybrid
        self.Q = KernelRepresentation(stateCount + actionCount, 1, config)
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        #self.phi = config.getfloat('Phi', 1)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0

    def train(self, step, x, x_, nonterminal, delta, gamma, rand):
        self.eta.step(step)
        self.beta.step(step)

        # Following 2 lines are under debate - what's the right way to do batch SCGD?
        yy = self.y + self.beta.value * (delta - self.y)
        self.y = np.mean(yy)  # Running average of TAD error
        ######


        N = float(len(delta))

        if self.algorithm == 'td' or self.algorithm == 'td0':
            if self.algorithm == 'td0': # No steps for exploratory actions
                yy[rand] = 0

            if np.flatnonzero(yy).size > 0:
                self.Q.shrink(1. - self.eta.value * self.lossL)
                self.Q.append(x, self.eta.value / N * yy)
        elif self.algorithm == 'hybrid' or self.algorithm == 'gtd' or self.algorithm == 'gtdrandom':
            # Stack sample points
            X = np.vstack((x, x_[nonterminal]))
            W = np.zeros((len(X), 1))

            if self.algorithm == 'gtd': # No steps for greedy actions...
                yy[np.logical_not(rand)] = 0

            W[:len(x)] = self.eta.value / N * yy

            if self.algorithm == 'hybrid': # TD steps for the greedy actions
                yy[np.logical_not(rand)] = 0

            W[len(x):] = -self.eta.value / N * gamma * yy[nonterminal]

            if np.flatnonzero(W).size > 0:
                self.Q.shrink(1. - self.eta.value * self.lossL)
                X = X[np.flatnonzero(W), :]
                W = W[np.flatnonzero(W)]
                if self.algorithm == 'gtdrandom':
                    randX = np.reshape(np.random.uniform([-1,-1,-8,-2],[1,1,8,2]),(1,-1))
                    randW = np.reshape(np.random.normal(0, 5*np.sqrt(2*self.eta.value)),(1,-1))
                    X = np.append(X,randX,axis=0)
                    W = np.append(W,randW,axis=0)
                self.Q.append(X[np.flatnonzero(W),:], W[np.flatnonzero(W)])

        else:
            raise ValueError('Unknown algorithm: {}'.format(self.algorithm))

        # Prune
        self.Q.prune((self.eps / N) ** 2 * self.eta.value ** 2 / self.beta.value)

    def evaluate(self, xs):
        "Evaluate the Q function for a list of (s,a) pairs."
        return self.Q(np.array(xs))

    def evaluateOne(self, x):
        "Evaluate the Q function for a single (s,a) pair."
        return self.Q(x)

    def maximize(self, ss):
        "Find the maximizing action for a batch of states."
        return [self.Q.argmax(s) for s in ss]

    def maximizeOne(self, s):
        "Find the maximizing action for a single state."
        return self.Q.argmax(s)


    def model_error(self):
        return 0.5 * self.lossL * self.Q.normsq()

# ==================================================
# An agent using Q-Learning

class KQLearningAgentIID(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

        # ---- Configure batch size
        self.batchSize = config.getint('MinibatchSize', 1)
        self.model = KQLearningModel(self.stateCount, self.actionCount, config)

        # How many steps we have observed
        self.steps = 0
        # ---- Configure exploration
        self.epsilon = ScheduledParameter('ExplorationRate', config)
        self.epsilon.step(0)
        # ---- Configure rewards
        self.gamma = config.getfloat('RewardDiscount')
        # ---- Configure priority experience replay
        self.memory = None
        self.eps = config.getfloat('ExperiencePriorityMinimum', 0.01)
        self.alpha = config.getfloat('ExperiencePriorityExponent', 1.)

    def _getStates(self, batch):
        #no_state = np.zeros(self.stateCount + self.actionCount)

        x = np.array([assemble(e[0], e[1]) for (_, e) in batch])
        rand = np.array([e[4] for (_, e) in batch])
        x_ = np.zeros((len(batch), self.stateCount + self.actionCount))

        nonterminal = []
        for i, (_, e) in enumerate(batch):
            if e[3] is not None:
                a_ = self.model.maximizeOne(e[3])
                x_[i] = assemble(e[3], a_)

                nonterminal.append(i)

        r = np.array([e[2] for (_, e) in batch])
        return x, x_, nonterminal, r, rand

    def _computeError(self, x, x_, nonterminal, r):
        error = r.reshape((-1, 1)) - self.model.evaluate(x)
        error[nonterminal] += self.gamma * self.model.evaluate(x_[nonterminal])
        return error

    def bellman_error(self, s, a, r, s_):
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r - self.model.evaluateOne(x)
        else:
            a_ = self.model.maximizeOne(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            return r + self.gamma * self.model.evaluateOne(x_) - self.model.evaluateOne(x)

    def act(self, s, stochastic=True):
        "Decide what action to take in state s."

        if stochastic and (random.random() < self.epsilon.value):
            a = np.random.uniform(self.min_act, self.max_act)
            rand = True
        else:
            a = self.model.Q.argmax(s)
            
            rand = False

        a_temp = np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))
        return a_temp, rand

    def observe(self, sample):
        error = self.bellman_error(sample[0],sample[1],sample[2],sample[3])
        self.memory.add(sample, np.abs((error[0] + self.eps) ** self.alpha))
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):

        batch = self.memory.sample(self.batchSize)

        x, x_, nt, r, rand = self._getStates(batch)

        # compute bellman error
        error = self._computeError(x, x_, nt, r)

        # update model
        self.model.train(self.steps, x, x_, nt, error, self.gamma, rand)

        # compute updated error
        error = self._computeError(x, x_, nt, r)

        # compute our average minibatch loss
        loss = 0.5 * np.mean(error ** 2) + self.model.model_error()
        # compute our model order
        modelOrder = len(self.model.Q.D)
        # report metrics
        return (float(loss), float(modelOrder))

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')