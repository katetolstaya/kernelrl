import random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
from corerl.memory import PrioritizedMemory


# ==================================================
# A POLK Q-Learning Model

class KQLearningModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount + actionCount, 1, config)
        self.algorithm = config.get('Algorithm', 'td').lower()  # gtd, td or hybrid
        # Learning rate
        self.eta = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        self.phi = config.getfloat('Phi', 0.0)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # TD-loss expectation approximation rate
        self.beta = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0  # np.zeros((0,1))

    def train(self, step, x, x_, nonterminal, delta, gamma, rand_act=None):
        self.eta.step(step)
        self.beta.step(step)

        yy = self.y + self.beta.value * (delta - self.y)
        self.Q.shrink(1. - self.eta.value * self.lossL)

        # Stack sample points
        if self.algorithm == 'hybrid':
            nonterminal = list(set(nonterminal) & set(rand_act))

        if self.algorithm == 'gtd' or self.algorithm == 'hybrid':
            X = np.vstack((x, x_[nonterminal]))
            W = np.zeros((len(X), 1))
            N = float(len(delta))

            W[:len(x)] = self.eta.value / N * yy
            W[len(x):] = -self.phi * self.eta.value / N * gamma * yy[nonterminal]
            self.y = np.mean(yy)  # Running average of TAD error
        elif self.algorithm == 'td':
            X = x
            N = float(len(delta))
            W = self.eta.value / N * yy
            self.y = np.mean(yy)  # Running average of TAD error
        else:
            raise ValueError('Unknown algorithm: {}'.format(self.algorithm))

        self.Q.append(X, W)
        # Prune
        # self.Q.prune(self.eps ** 2 * (self.eta.value / N) ** 2 / self.beta.value)
        self.Q.prune((self.eps * self.eta.value ** 2) ** 2)

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

class KQLearningAgentPER(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high
        self.max_model_order = config.getfloat('MaxModelOrder', 10000)

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        self.act_mult = config.getfloat('ActMultiplier', 1)
        self.rand_act = True

        # ---- Configure batch size
        self.batchSize = config.getint('MinibatchSize', 16)

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'gtd').lower()
        if algorithm == 'gtd' or algorithm == 'td' or algorithm == 'hybrid':
            self.model = KQLearningModel(self.stateCount, self.actionCount, config)
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
        # no_state = np.zeros(self.stateCount + self.actionCount)

        def assemble(s, a):
            return np.concatenate((s.reshape((1, -1)), a.reshape((1, -1))), axis=1).flatten()

        x = np.array([assemble(e[0], e[1]) for (_, e) in batch])
        x_ = np.zeros((len(batch), self.stateCount + self.actionCount))
        nonterminal = []
        rand_act = []
        for i, (_, e) in enumerate(batch):
            if e[3] is not None:
                a_ = self.model.maximizeOne(e[3])
                x_[i] = assemble(e[3], a_)

                nonterminal.append(i)

                if self.model.algorithm == 'hybrid':
                    if len(e) == 4 or e[4]:
                        # assumes that samples added into the buffer initially are random
                        rand_act.append(i)

        r = np.array([e[2] for (_, e) in batch])

        if self.model.algorithm == 'hybrid':
            return x, x_, nonterminal, r, rand_act
        else:
            return x, x_, nonterminal, r

    def _computeError(self, x, x_, nonterminal, r, rand_act=None):
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
            a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
            self.rand_act = True
        else:
            a = self.model.Q.argmax(s)
            self.rand_act = False

        a_temp = np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))
        return a_temp

    def observe(self, sample):
        error = self._computeError(*self._getStates([(0, sample)]))

        if self.model.algorithm == 'hybrid':
            s, a, r, s_ = sample
            sample = (s, a, r, s_, self.rand_act)

        self.memory.add(sample, np.abs((error[0] + self.eps) ** self.alpha))
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
        batch = self.memory.sample(self.batchSize)

        if self.model.algorithm == 'hybrid':
            x, x_, nt, r, rand_act = self._getStates(batch)
        else:
            x, x_, nt, r = self._getStates(batch)

        # compute bellman error
        error = self._computeError(x, x_, nt, r)

        # update model
        if self.model.algorithm == 'hybrid':
            self.model.train(self.steps, x, x_, nt, error, self.gamma, rand_act)
        else:
            self.model.train(self.steps, x, x_, nt, error, self.gamma)

        # compute updated error
        error = self._computeError(x, x_, nt, r)

        # update errors in memory
        if type(self.memory) == PrioritizedMemory:
            for (idx, _), delta in zip(batch, error):
                self.memory.update(idx, (np.abs(delta) + self.eps) ** self.alpha)

        # compute our average minibatch loss
        loss = 0.5 * np.mean(error ** 2) + self.model.model_error()
        # compute our model order
        model_order = len(self.model.Q.D)
        # report metrics
        return (float(loss), float(model_order))

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')
