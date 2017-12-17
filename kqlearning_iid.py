from function_pre import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import pickle
from memory import PrioritizedMemory, Memory

# ==================================================
# A POLK Q-Learning Model

class KQLearningModel(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount+actionCount, 1, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
	self.phi = config.getfloat('Phi',1)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0 #np.zeros((0,1))
    def train(self, step, x, x_, nonterminal, delta, gamma):
        self.eta.step(step)
        self.beta.step(step)

	yy = self.y + self.beta.value * (delta - self.y)
        self.Q.shrink(1. - self.eta.value * self.lossL)

        # Stack sample points
        X = np.vstack((x, x_[nonterminal]))
        W = np.zeros((len(X), 1))
	N = float(len(delta))

	W[:len(x)] = self.eta.value / N * yy
	W[len(x):] = -self.phi * self.eta.value / N * gamma * yy[nonterminal]
	self.y = np.mean(yy)         # Running average of TAD error

        self.Q.append(X, W)
        # Prune
        self.Q.prune((self.eps/N)**2 * self.eta.value**2 / self.beta.value)
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
        return 0.5*self.lossL*self.Q.normsq()

# ==================================================
# An agent using Q-Learning

class KQLearningAgentIID(object):
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
        self.batchSize = config.getint('MinibatchSize', 1)

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'SCGD')
        self.save_steps = config.getint('SaveInterval', 10000000)
        self.folder = config.get('Folder', 'exp')
        self.train_steps = config.getint('TrainInterval', 4)
        if algorithm.lower() == 'scgd':
            self.model = KQLearningModel(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = KQLearningModelTD2(self.stateCount, self.actionCount, config)
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
        self.eps   = config.getfloat('ExperiencePriorityMinimum', 0.01)
        self.alpha = config.getfloat('ExperiencePriorityExponent', 1.)

    def _getStates(self, batch):
        no_state = np.zeros(self.stateCount+self.actionCount)
        def assemble(s,a):
            return np.concatenate((s.reshape((1,-1)), a.reshape((1,-1))), axis=1).flatten()
        x  = np.array([assemble(e[0],e[1]) for (_,e) in batch])
        x_ = np.zeros((len(batch),self.stateCount+self.actionCount))
        nonterminal = []
        for i, (_,e) in enumerate(batch):
            if e[3] is not None:
                a_ = self.model.maximizeOne(e[3])
                x_[i] = assemble(e[3],a_)

                nonterminal.append(i)

        r  = np.array([e[2] for (_,e) in batch])
        return x, x_, nonterminal, r
    def _computeError(self, x, x_, nonterminal, r):
        error = r.reshape((-1,1)) - self.model.evaluate(x)
        error[nonterminal] += self.gamma*self.model.evaluate(x_[nonterminal])
        return error


    def bellman_error(self, s, a, r, s_):
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r -  self.model.evaluateOne(x)
        else:
            a_ = self.model.maximizeOne(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            return r + self.gamma * self.model.evaluateOne(x_) -  self.model.evaluateOne(x)

    def act(self, s, stochastic=True):
        "Decide what action to take in state s."

        if stochastic and (random.random() < self.epsilon.value):
            a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
            # return self.action_space.sample()
        else:
            a = self.model.Q.argmax(s)
	    x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
	    #print self.model.Q(x)

        a_temp = np.reshape(np.clip(a, self.min_act, self.max_act),(-1,))
        return a_temp

    def observe(self, sample):

        error = self._computeError(*self._getStates([(0, sample)]))

        self.memory.add(sample, 0)#np.abs((error[0] + self.eps) ** self.alpha))
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
        # model order cutoff
        #if len(self.model.Q.D) > self.max_model_order:
        #    self.model.eps = self.model.eps * 2
        # periodically save policy
        #if self.steps % self.save_steps == 0:
        #    with open(self.folder + '/kpolicy_model_' + str(int(self.steps / self.save_steps)) + '.pkl','wb') as f:
        #        pickle.dump(self.model.Q, f)
        # pull a batch of points
        batch = self.memory.sample(self.batchSize)
        x, x_, nt, r = self._getStates(batch)

        # compute bellman error
        error = self._computeError(x, x_, nt, r)

        # update model
        self.model.train(self.steps, x, x_, nt, error, self.gamma)

        # compute updated error
        error = self._computeError(x, x_, nt, r)

        # compute our average minibatch loss
        loss = 0.5*np.mean(error**2) + self.model.model_error()
        # compute our model order
        modelOrder = len(self.model.Q.D)
        # report metrics
        return (float(loss), float(modelOrder))

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return ('Training Loss','Model Order')

# ==================================================
class RandomAgent:
    def __init__(self, env, cfg):
	self.act_mult = cfg.getfloat('ActMultiplier', 1)
        memoryCapacity = cfg.getint('MemoryCapacity', 10000)
        self.memory = Memory(memoryCapacity) #PrioritizedMemory(memoryCapacity)
        self.eps   = cfg.getfloat('ExperiencePriorityMinimum', 0.01)
        self.alpha = cfg.getfloat('ExperiencePriorityExponent', 1.0)
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high
    def act(self, s):
        a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
	return np.reshape(np.clip(a, self.min_act, self.max_act),(-1,))
    def observe(self, sample):

        error = abs(sample[2])
        self.memory.add(sample, 0) #(error + self.eps) ** self.alpha)
    def improve(self):
        return ()

