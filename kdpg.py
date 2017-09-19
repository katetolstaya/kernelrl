from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import copy
import pickle
import json

# ==================================================
# A POLK SARSA Model

class KDPGModel(object):
    def __init__(self, stateCount, actionCount, config):

        self.Q = KernelRepresentation(stateCount + actionCount , 1, config)
        self.pi = KernelRepresentation(stateCount, actionCount,  config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
	self.pi_step = config.getfloat('PolicyStep')
	self.eps_pi = config.getfloat('RepresentationErrorPi')
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
	#self.y_df = np.zeros((actionCount,1))

    def bellman_error(self, s, a, r, s_,a_):
        x = np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            x_ = np.concatenate((np.reshape(s_,(1,-1)), np.reshape(a_,(1,-1))),axis=1)
            return r + self.Q(x_) - self.Q(x)

    def bellman_error2(self, x, r, x_):
        if x_ is None:
            return r - self.Q(x)
        else:
            return r + self.Q(x_) - self.Q(x)
    def model_error(self):
        return 0.5*self.lossL*self.Q.normsq()
    def predict(self, s):
        "Predict the Q function values for a batch of states."
        return self.pi(s)
    def predictOne(self, s):
        "Predict the Q function values for a single state."
        return self.pi(s.reshape(1, len(s))).flatten()
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order')

class KDPGModelTD(KDPGModel):
    def train(self, step, sample):
        self.eta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
	x = np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1)

	if s_ is None or a_ is None:
	    x_ = None
	else: 
	    x_ = np.concatenate((np.reshape(s_,(1,-1)), np.reshape(a_,(1,-1))),axis=1)


        # Compute error
        delta = self.bellman_error2(x,r,x_)
        # Q update
        self.Q.shrink(1. - self.eta.value * self.lossL)
        self.Q.append(x, self.eta.value * delta)
        self.Q.prune(self.eps * self.eta.value**2 )

	# policy update
	df = self.Q.df(x)
	self.pi.shrink(1. - self.eta.value * self.lossL)
        self.pi.append(s, self.eta.value  * self.pi_step * df[-self.pi.W.shape[1]:]) #*delta)
        self.pi.prune(self.eps_pi * self.eta.value**2 * self.pi_step**2)

        modelOrder_ = len(self.Q.D) + len(self.pi.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_,a_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

class KDPGModelSCGD(KDPGModel):
    def __init__(self, stateCount, actionCount, config):
        super(KDPGModelSCGD, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.
    def train(self, step, sample):
        self.eta.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
	s = copy.deepcopy(s)
	s_ = copy.deepcopy(s_)
	x = copy.deepcopy(np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1))

	############
	# update Q
	if s_ is None or a_ is None:
	    x_ = None
	else: 
	    x_ = copy.deepcopy(np.concatenate((np.reshape(s_,(1,-1)), np.reshape(a_,(1,-1))),axis=1))
        # Compute error
        delta = self.bellman_error2(x,r,x_)
        # Running average
        self.y += self.beta.value * (delta - self.y)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)
        if s_ is None:
            self.Q.append(x, self.eta.value * self.y)
        else:
            W = np.zeros((2, 1))
            W[0]  = -1.
            W[1] = self.gamma
            self.Q.append(np.vstack((x,x_)), -self.eta.value * self.y * W)
	self.Q.prune(self.eps * self.eta.value**2)


	#############
	# update pi
	df = self.Q.df(x)[-self.pi.W.shape[1]:]
	#self.y_df += self.beta.value * (df - self.y_df)
	self.pi.shrink(1. - self.pi_step * self.lossL)
        self.pi.append(np.reshape(s,(1,-1)),  np.reshape(self.pi_step  * df,(1,-1))) #* -self.y
	#if not s_ is None:
	#    df_ = self.Q.df(x_)
        #    self.pi.append(np.reshape(s_,(1,-1)), - self.y * self.pi_step* self.gamma * self.eta.value  * df_[-self.pi.W.shape[1]:]) #* -self.y
        self.pi.prune(self.eps_pi *  self.pi_step**2)


	    
        # Prune
        #modelOrder = len(self.Q.D)
        modelOrder_ = len(self.pi.D)  + len(self.Q.D)

        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_,a_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

# ==================================================
# An agent using SARSA

class KDPGAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
	self.max_act = json.loads(config.get('MaxAction'))
	self.min_act = json.loads(config.get('MinAction'))
	# 

        # We can switch between SCGD and TD learning here
        self.save_steps = config.getint('SaveInterval', 100000)
        self.folder = config.get('Folder', 'exp')
        algorithm = config.get('Algorithm', 'SCGD')
        if algorithm.lower() == 'scgd':
            self.model = KDPGModelSCGD(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = KDPGModelTD(self.stateCount, self.actionCount, config)
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
            return np.random.uniform(self.min_act,self.max_act)
	    #return random.randint(0, self.actionCount-1)
        else:
	    a = np.clip(self.model.predictOne(s), self.min_act, self.max_act)
	    #print a
	    return a
    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)
    def improve(self):
	if self.steps % self.save_steps == 0:
            with open(self.folder + '/kpolicy_pi_' + str(int(self.steps / self.save_steps)) + '.pkl','wb') as f:
                pickle.dump(self.model.pi, f)
            with open(self.folder + '/kpolicy_model_' + str(int(self.steps / self.save_steps)) + '.pkl','wb') as f:
                pickle.dump(self.model.Q, f)

        return self.model.train(self.steps, self.lastSample)
    def bellman_error(self, s, a, r, s_, a_):
        return self.model.bellman_error(s,a,r,s_,a_)
    def model_error(self):
        return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names

