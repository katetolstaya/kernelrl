from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import pickle
from memory import PrioritizedMemory, Memory


# ==================================================
# A POLK Q-Learning Model

class KQGreedyModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount + actionCount, 2, config)
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0  # np.zeros((0,1))

    def train(self, step, x, x_, nonterminal, delta, g, gamma):
        self.eta.step(step)
        self.beta.step(step)

        self.Q.shrink(1. - self.eta.value * self.lossL)

        # Stack sample points
        X = np.vstack((x, x_[nonterminal]))
        W = np.zeros((len(X), 2))
        N = float(len(delta))

        W[:len(x),0] = self.eta.value / N * delta
        W[len(x):,0] = - self.eta.value / N * gamma * g[nonterminal][:]
        W[:len(x),1] = self.beta.value / N * (delta[:] - g[:])
        self.Q.append(X, W)
        # Prune
        self.Q.prune((self.eps / N) ** 2 * self.eta.value ** 2 / self.beta.value)

    def evaluate(self, xs):
        "Evaluate the Q function for a list of (s,a) pairs."
        return np.reshape(self.Q(np.array(xs))[:,0],(-1,1)) #self.Q(np.array(xs))

    def evaluateOne(self, x):
        "Evaluate the Q function for a single (s,a) pair."
        return self.Q(x)[:,0]

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

class KQGreedyAgentPER(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high
        self.max_model_order = config.getfloat('MaxModelOrder', 10000)

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        self.act_mult = config.getfloat('ActMultiplier', 1)

        # ---- Configure batch size
        self.batchSize = config.getint('MinibatchSize', 16)

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'SCGD')
        self.save_steps = config.getint('SaveInterval', 10000000)
        self.folder = config.get('Folder', 'exp')
        self.train_steps = config.getint('TrainInterval', 4)
        if algorithm.lower() == 'scgd':
            self.model = KQGreedyModel(self.stateCount, self.actionCount, config)
        else:
            raise ValueError('Unknown algorithm: {}'.format(algorithm))
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
        no_state = np.zeros(self.stateCount + self.actionCount)

        def assemble(s, a):
            return np.concatenate((s.reshape((1, -1)), a.reshape((1, -1))), axis=1).flatten()

        x = np.array([assemble(e[0], e[1]) for (_, e) in batch])
        x_ = np.zeros((len(batch), self.stateCount + self.actionCount))
        nonterminal = []
        for i, (_, e) in enumerate(batch):
            if e[3] is not None:
                a_ = self.model.maximizeOne(e[3])
                x_[i] = assemble(e[3], a_)

                nonterminal.append(i)

        r = np.array([e[2] for (_, e) in batch])
        return x, x_, nonterminal, r

    def _computeError(self, x, x_, nonterminal, r):
        qgvals = self.model.Q(np.array(x))
        error = r.reshape((-1, 1)) - np.reshape(qgvals[:,0],(-1,1))
        error[nonterminal] += self.gamma * self.model.evaluate(x_[nonterminal])
        return error.flatten(), (qgvals[:,1]).flatten() # return error, g

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
            a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
        else:
            a = self.model.Q.argmax(s)
            #x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)

        a_temp = np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))
        return a_temp

    def observe(self, sample):
        error, _ = self._computeError(*self._getStates([(0, sample)]))
        self.memory.add(sample, np.abs((error[0] + self.eps) ** self.alpha))
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
        batch = self.memory.sample(self.batchSize)
        x, x_, nt, r = self._getStates(batch)

        # compute bellman error
        error, g = self._computeError(x, x_, nt, r)

        # update model
        self.model.train(self.steps, x, x_, nt, error, g, self.gamma)

        # compute updated error
        error, _ = self._computeError(x, x_, nt, r)

        # update errors in memory
        if self.memory is PrioritizedMemory:
            for (idx, _), delta in zip(batch, error):
                self.memory.update(idx, (np.abs(delta) + self.eps) ** self.alpha)

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
