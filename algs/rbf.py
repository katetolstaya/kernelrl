import random

import numpy as np

from corerl.core import ScheduledParameter
from scipy import *
from scipy.linalg import norm, pinv
import json


# ==================================================
# A POLK Q-Learning Model\

def assemble(s, a):
    return np.concatenate((s.reshape((1, -1)), a.reshape((1, -1))), axis=1).flatten()

class RBFModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.algorithm = config.get('Algorithm', 'td').lower() # gtd, td or hybrid or gtdrandom

        #self.Q = KernelRepresentation(stateCount + actionCount, 1, config)
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

        sigma = json.loads(config.get('GaussianBandwidth'))
        self.sig = np.array(sigma)
        self.s = -1. / (2 * self.sig ** 2)

        self.indim = stateCount + actionCount
        self.outdim = 1
        self.numCenters = config.getint('NumberCenters', 1.0)

        # action space boundaries
        self.low_sa = np.reshape(json.loads(config.get('MinSA')), (-1, 1))
        self.high_sa = np.reshape(json.loads(config.get('MaxSA')), (-1, 1))
        # action space boundaries
        self.low_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.high_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        # Initialize model matrices
        # D is a K x N matrix of dictionary elements


        self.D = np.zeros((self.numCenters, self.indim)) #np.random.uniform(self.low_sa, self.high_sa, (self.numCenters))

        x_ = np.linspace(self.low_sa[0], self.high_sa[0], 10)
        y_ = np.linspace(self.low_sa[1], self.high_sa[1], 10)
        z_ = np.linspace(self.low_sa[2], self.high_sa[2], 5)
        w_ = np.linspace(self.low_sa[3], self.high_sa[3], 10)

        x, y, z, w = np.meshgrid(x_, y_, z_,w_, indexing='ij')

        self.D[:, 0] = x.flatten()
        self.D[:, 1] = y.flatten()
        self.D[:, 2] = z.flatten()
        self.D[:, 3] = w.flatten()

        #for i in range(0, self.indim):
        #    self.D[:, i] = np.random.uniform(self.low_sa[i], self.high_sa[i], (self.numCenters,))

        # W is a K x M matrix of weights
        self.W = np.zeros((self.numCenters, self.outdim))

        # gradient ascent parameters for argmax function
        self.grad_step = config.getfloat('GradStep', 0.05)
        self.grad_prec = config.getfloat('GradPrecision',0.005)
        self.n_iters = config.getint('GradIters', 40)
        self.n_points = config.getint('GradPoints', 20)

    def _distEucSq(self, X, Y):
        if len(X.shape) > 1:
            if len(Y.shape) > 1:
                m = X.shape[0]
                n = Y.shape[0]
                # print(Y.shape)
                # print(X.shape)
                XX = np.sum(X * self.s * X, axis=1)
                YY = np.sum(Y * self.s * Y, axis=1)
                # print(XX.shape)
                # print(YY.shape)
                return np.tile(XX.reshape(m, 1), (1, n)) + np.tile(YY, (m, 1)) - 2 * X.dot((self.s * Y).T)
            else:
                m = X.shape[0]
                XX = np.sum(X * self.s * X, axis=1)
                YY = np.sum(Y * self.s * Y)
                return np.tile(XX.reshape(m, 1), (1, 1)) + np.tile(YY, (m, 1)) - 2 * X.dot(self.s * Y).reshape(m, 1)
        else:
            if len(Y.shape) > 1:
                n = Y.shape[0]
                XX = np.sum(X * self.s * X)
                YY = np.sum(Y * self.s * Y, axis=1)
                return np.tile(XX, (1, n)) + np.tile(YY, (1, 1)) - 2 * X.dot((self.s * Y).T)
            else:
                m = 1
                n = 1
                XX = np.sum(X * self.s * X)
                YY = np.sum(Y * self.s * Y)
                return np.tile(XX, (1, 1)) + np.tile(YY, (1, 1)) - 2 * X.dot(Y)

    def fsquare(self, X, Y):
        sx1 = np.shape(X)[0]
        sy1 = np.shape(Y)[0]

        ret = np.zeros((sx1,sy1))
        try:
            ret = np.exp(self._distEucSq(X, Y))
        except AttributeError:
            print ("Attribute Error")
            print (self.s)
            print (X)
            print (Y)
        except ValueError:
            print ("Value Error")
            print (self.s)
            print (np.shape(X))
            print (np.shape(Y))

        return ret

    # ------------------------------
    # Evaluate a set of values
    # ------------------------------
    # X : a L x N matrix where L is number of points to evaluate
    def f(self, X):
        # Evaluate f
        value = self.fsquare(X, self.D).dot(self.W)
        return value

    def train(self, step, x, x_, nonterminal, delta, gamma, rand):
        self.eta.step(step)
        self.beta.step(step)

        # Following 2 lines are under debate - what's the right way to do batch SCGD?
        yy = self.y + self.beta.value * (delta - self.y)
        self.y = np.mean(yy)  # Running average of TAD error
        ######


        N = float(len(delta))

        if self.algorithm == 'td':
            #self.W = (1 - self.eta.value) * self.W + self.eta.value * self.fsquare(x, self.D).T * delta
            self.W = self.W + self.eta.value * self.fsquare(x, self.D).T * delta
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
                self.W = (1 - self.eta.value) * self.W
                X = X[np.flatnonzero(W), :]
                W = W[np.flatnonzero(W)]

                self.W += self.eta.value * (W.T.dot(self.fsquare(X,self.D))).T

        else:
            raise ValueError('Unknown algorithm: {}'.format(self.algorithm))

    def evaluate(self, xs):
        "Evaluate the Q function for a list of (s,a) pairs."
        return self.f(np.array(xs))

    def evaluateOne(self, x):
        "Evaluate the Q function for a single (s,a) pair."
        temp =  self.f(x)
        return temp

    def maximize(self, ss):
        "Find the maximizing action for a batch of states."
        return [self.argmax(s) for s in ss]

    def maximizeOne(self, s):
        "Find the maximizing action for a single state."
        return self.argmax(s)

    def argmax(self, Y):

        Y = np.reshape(Y, (1, -1))
        dim_s2 = np.shape(Y)[1]  # num states

        # Initialize candidate points to random values in the action space, with given state
        N = self.n_points
        acts = np.zeros((N, self.indim))
        for i in range(0, self.indim - dim_s2):
            acts[:, i + dim_s2] = np.random.uniform(self.low_act[i], self.high_act[i], (N,))
        acts[:, 0:dim_s2] = np.tile(Y, (N, 1))

        # # Gradient ascent #TODO
        # iters = 0
        # keep_updating = np.full((N,), True, dtype=bool)
        # while (keep_updating.any()) and iters < self.n_iters:
        #     iters = iters + 1
        #
        #     # compute gradient of Q with respect to (s,a), zero out the s component
        #     df = np.zeros((N,dim_d2))
        #     df[keep_updating, :] = self.df(acts[keep_updating, :])
        #     df[:, 0:dim_s2] = 0
        #
        #     # gradient step
        #     acts = acts + self.grad_step * df
        #
        #     # stop updating points on edge of action space, points where delta is small
        #     temp1 = np.logical_and(np.any(acts[:,dim_s2:] <= self.high_act.T,axis=1), np.any(acts[:,dim_s2:] >= self.low_act.T,axis=1))
        #     temp2 = np.logical_and(temp1, np.linalg.norm(self.grad_step * df, axis=1) > self.grad_prec)
        #     keep_updating = temp2
        #
        # # Clip points to action space
        # for i in range(0, dim_d2 - dim_s2):
        #     acts[:, i + dim_s2] = np.clip(acts[:,i + dim_s2], self.low_act[i], self.high_act[i])

        # Check for point with best Q value
        b = self.f(acts)[:,0]
        amax = np.array([np.argmax(np.random.random(b.shape) * (b == b.max()))])
        action = np.reshape(acts[amax, dim_s2:], (-1, 1))
        return action
    def model_error(self):
        return 0


# ==================================================
# An agent using Q-Learning

class RBFAgentIID(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

        # ---- Configure batch size
        self.batchSize = config.getint('MinibatchSize', 1)
        self.model = RBFModel(self.stateCount, self.actionCount, config)

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
        a = a[0]
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        error = 0
        if s_ is None:
            error =  r - self.model.evaluateOne(x)
        else:
            a_ = self.model.maximizeOne(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            error =  r + self.gamma * self.model.evaluateOne(x_) - self.model.evaluateOne(x)
        #print error
        return error

    def act(self, s, stochastic=True):
        "Decide what action to take in state s."

        if stochastic and (random.random() < self.epsilon.value):

            a = np.random.uniform(self.min_act, self.max_act)
            rand = True
        else:

            a = self.model.argmax(s)

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
        loss = 0.5 * np.mean(error ** 2)
        # compute our model order
        modelOrder = self.model.numCenters
        # report metrics
        return (float(loss), float(modelOrder))

    def model_error(self):
        return 0

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')