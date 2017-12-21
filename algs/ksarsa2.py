import random

import numpy as np

from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation


# ==================================================
# A POLK SARSA Model

class KSARSAModel2(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount+actionCount,1, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
        # Running estimate of our expected TD-loss
        self.y = 0.
    def bellman_error(self, s, a, r, s_, a_):
    	x = np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            x_ = np.concatenate((np.reshape(s_,(1,-1)), np.reshape(a_,(1,-1))),axis=1)
            return r + self.Q(x_) - self.Q(x)
    def model_error(self):
        return 0.5*self.lossL*self.Q.normsq()
    def predict(self, s):
	pass        
	#"Predict the Q function values for a batch of states."
        #return self.Q(s)
    def predictOne(self, s):
	pass        
	#"Predict the Q function values for a single state."
        #return self.Q(s.reshape(1, len(s))).flatten()
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order')

class KSARSAModelTD2(KSARSAModel2):
    def train(self, step, sample):
        self.eta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
        # Compute error
        delta = self.bellman_error(s,a,r,s_,a_)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)
        x = np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1)
        self.Q.append(x, self.eta.value * delta)
        # Prune
        modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value**2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_,a_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

class KSARSAModelSCGD2(KSARSAModel2):
    def __init__(self, stateCount, actionCount, config):
        super(KSARSAModelSCGD2, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.
    def train(self, step, sample):
        self.eta.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
        # Compute error
        delta = self.bellman_error(s,a,r,s_,a_)
        # Running average
        self.y += self.beta.value * (delta - self.y)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)

        x = np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1)
        if s_ is None:
            self.Q.append(x, self.eta.value * self.y)
        else:
	    x_ = np.concatenate((np.reshape(s_,(1,-1)), np.reshape(a_,(1,-1))),axis=1)
            W = np.zeros((2, 1))
            W[0]  = -1.
            W[1] = self.gamma
            self.Q.append(np.vstack((x,x_)), -self.eta.value * self.y * W)
        # Prune
        modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value**2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_,a_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

# ==================================================
# An agent using SARSA

class KSARSAAgent2(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
	self.max_act = 5
        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'TD')
        if algorithm.lower() == 'scgd':
            self.model = KSARSAModelSCGD2(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = KSARSAModelTD2(self.stateCount, self.actionCount, config)
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
	    return np.random.uniform(-self.max_act,self.max_act,(self.actionCount,1))
            #return random.randint(0, self.actionCount-1)
        else:
            return self.model.Q.argmax(s)
	    #return self.model.predictOne(s).argmax()
    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)
    def improve(self):
        return self.model.train(self.steps, self.lastSample)
    def bellman_error(self, s, a, r, s_, a_):
        return self.model.bellman_error(s,a,r,s_,a_)
    def model_error(self):
        return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names

