import random

import numpy as np

from corerl.core import ScheduledParameter
import json
from corerl.kernel import _distEucSq, GaussianKernel


# ==================================================
# A POLK Q-Learning Model\

def assemble(s, a):
    return np.concatenate((s.reshape((1, -1)), a.reshape((1, -1))), axis=1).flatten()


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

class RBFModel(object):
    def __init__(self, state_count, action_count, config):
        self.algorithm = config.get('Algorithm', 'td').lower()  # gtd, td or hybrid or gtdrandom

        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # self.phi = config.getfloat('Phi', 1)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0

        sigma = json.loads(config.get('GaussianBandwidth'))
        self.sig = np.array(sigma)
        self.s = -1. / (2 * self.sig ** 2)

        self.kernel = GaussianKernel(self.sig)

        self.indim = state_count + action_count
        self.outdim = 1
        self.num_centers = config.getint('NumberCenters', 1.0)

        # action space boundaries
        self.min_bounds = np.reshape(json.loads(config.get('MinSA')), (-1, 1))
        self.max_bounds = np.reshape(json.loads(config.get('MaxSA')), (-1, 1))
        # action space boundaries
        self.min_action = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_action = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        self.dim_centers = np.reshape(json.loads(config.get('DimCenters')), (-1,))
        self.multipliers = np.insert(np.cumprod(self.dim_centers), 0, [1])[:-1]

        self.res = (self.max_bounds - self.min_bounds) / (self.dim_centers.reshape((-1, 1)) - 1)

        # Initialize model matrices
        # D is a K x N matrix of dictionary elements

        self.D = np.zeros(
            (self.num_centers, self.indim))  # np.random.uniform(self.low_sa, self.high_sa, (self.numCenters))

        axes = [np.linspace(self.min_bounds[i], self.max_bounds[i], self.dim_centers[i]) for i in range(self.indim)]
        grids = np.meshgrid(*axes)
        self.D = np.stack([g.ravel() for g in grids], axis=1)

        # W is a K x M matrix of weights
        self.W = np.zeros((self.num_centers, self.outdim))

        # gradient ascent parameters for argmax function
        self.grad_step = config.getfloat('GradStep', 0.05)
        self.grad_prec = config.getfloat('GradPrecision', 0.005)
        self.n_iters = config.getint('GradIters', 40)
        self.n_points = config.getint('GradPoints', 100)

    def linear_basis(self, X, Y):
        ind = (X - self.min_bounds.reshape((1, -1))) / self.res.reshape((1, -1))
        ind = np.sum(ind.astype(np.int32) * self.multipliers.reshape((1, -1)), axis=1)
        ret = get_one_hot(ind, self.num_centers)
        return ret

    # ------------------------------
    # Evaluate a set of values
    # ------------------------------
    # X : a L x N matrix where L is number of points to evaluate
    def f(self, X):
        # Evaluate f
        value = self.linear_basis(X, self.D).dot(self.W)
        return value

    def train(self, step, x, x_, nonterminal, delta, gamma, rand):
        self.eta.step(step)
        self.beta.step(step)

        if self.algorithm == 'td':
            # self.W = (1 - self.eta.value) * self.W + self.eta.value * self.fsquare(x, self.D).T * delta
            self.W += self.eta.value * np.dot(self.linear_basis(x, self.D).T, delta)

        elif self.algorithm == 'hybrid' or self.algorithm == 'gtd' or self.algorithm == 'gtdrandom':
            # Following 2 lines are under debate - what's the right way to do batch SCGD?
            yy = self.y + self.beta.value * (delta - self.y)
            self.y = np.mean(yy)  # Running average of TAD error
            ######

            N = float(len(delta))

            # Stack sample points
            X = np.vstack((x, x_[nonterminal]))
            weight_gradients = np.zeros((len(X), 1))

            if self.algorithm == 'gtd':  # No steps for greedy actions...
                yy[np.logical_not(rand)] = 0

            weight_gradients[:len(x)] = self.eta.value / N * yy

            if self.algorithm == 'hybrid':  # TD steps for the greedy actions
                yy[np.logical_not(rand)] = 0

            weight_gradients[len(x):] = -self.eta.value / N * gamma * yy[nonterminal]

            if np.flatnonzero(weight_gradients).size > 0:
                self.W = (1 - self.lossL) * self.W
                X = X[np.flatnonzero(weight_gradients), :]
                weight_gradients = weight_gradients[np.flatnonzero(weight_gradients)]
                self.W += self.eta.value * (weight_gradients.T.dot(self.linear_basis(X, self.D))).T

        else:
            raise ValueError('Unknown algorithm: {}'.format(self.algorithm))

    def evaluate(self, xs):
        """Evaluate the Q function for a list of (s,a) pairs."""
        return self.f(np.array(xs))

    def evaluateOne(self, x):
        """Evaluate the Q function for a single (s,a) pair."""
        temp = self.f(x)
        return temp

    def maximize(self, ss):
        """Find the maximizing action for a batch of states."""
        return [self.argmax(s) for s in ss]

    def maximizeOne(self, s):
        """Find the maximizing action for a single state."""
        return self.argmax(s)

    # def df(self, x):
    #     # Handle model divergence
    #     # Evaluate gradient
    #     tempW = np.reshape(self.W[:, 0], (1, 1, -1))  # use only axis 0 for argmax, df
    #     tempdf = self.kernel.df(x, self.D)
    #     return np.reshape(np.dot(tempW, tempdf), np.shape(x))

    # ------------------------------
    def argmax(self, Y):
        # TODO only supports 1 point in query
        Y = np.reshape(Y, (1, -1))
        num_states = np.shape(Y)[1]  # num states
        num_states_and_actions = self.D.shape[1]  # num states + actions

        axes = [np.linspace(self.min_bounds[i], self.max_bounds[i], self.dim_centers[i]) for i in range(num_states, num_states_and_actions)]
        grids = np.meshgrid(*axes)
        actions = np.stack([g.ravel() for g in grids], axis=1)
        actions = np.concatenate([np.tile(Y, (np.shape(actions)[0], 1)), actions], axis=1)

        # Check for point with best Q value
        action_value = self.f(actions)[:, 0]
        argmax = np.array([np.argmax(np.random.random(action_value.shape) * (action_value == action_value.max()))])
        action = np.reshape(actions[argmax, num_states:], (-1, 1))
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
        if s_ is None:
            error = r - self.model.evaluateOne(x)
        else:
            a_ = self.model.maximizeOne(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            error = r + self.gamma * self.model.evaluateOne(x_) - self.model.evaluateOne(x)
        return error

    def act(self, s, stochastic=True):
        """Decide what action to take in state s."""

        if stochastic and (random.random() < self.epsilon.value):
            action = np.random.uniform(self.min_act, self.max_act)
            random_flag = True
        else:
            action = self.model.argmax(s)
            random_flag = False
        clipped_action = np.reshape(np.clip(action, self.min_act, self.max_act), (-1,))
        return clipped_action, random_flag

    def observe(self, sample):
        error = self.bellman_error(sample[0], sample[1], sample[2], sample[3])
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

        return (float(loss), float(self.model.num_centers))

    def model_error(self):
        return 0

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')
