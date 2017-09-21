from function import KernelRepresentation
import numpy
import sys, math, random
from core import ScheduledParameter
import pickle
import json

# ==================================================
# A POLK Q-Learning Model

class KQLearningModel2(object):
    def __init__(self, stateCount, actionCount, config):
        self.Q = KernelRepresentation(stateCount+ actionCount,1, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
    def bellman_error(self, s, a, r, s_):
        x = numpy.concatenate((numpy.reshape(s,(1,-1)), numpy.reshape(a,(1,-1))),axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            a_ = self.Q.argmax(s_)
            x_ = numpy.concatenate((numpy.reshape(s_,(1,-1)), numpy.reshape(a_,(1,-1))),axis=1)
            return r + self.gamma*self.Q(x_) - self.Q(x)
    def bellman_error2(self, x, r, x_):
        if x_ is None:
            return r - self.Q(x)
        else:
            return r + self.gamma*self.Q(x_) - self.Q(x)
    def model_error(self):
        return 0.5*self.lossL*self.Q.normsq()
    def train(self, step, sample):
        pass
    def predict(self, s):
        pass
        #return self.Q(s)
    def predictOne(self, s):
        pass
        #"Predict the Q function values for a single state."
        #return self.Q.argmax(s)
        #return self.Q(s.reshape(1, len(s))).flatten()
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order')

class KQLearningModelTD2(KQLearningModel2):
    def train(self, step, sample):
        self.eta.step(step)
        # Unumpyack sample

        s, a, r, s_ = sample


	#r = r.astype(float)

	x = numpy.concatenate((numpy.reshape(s,(1,-1)), numpy.reshape(a,(1,-1))),axis=1)	
	#x = x.astype(float)
	
	if s_ is None:
	    a_ = None
	    x_ = None

	else: 
	    a_ = self.Q.argmax(s) 
	    x_ = numpy.concatenate((numpy.reshape(s_,(1,-1)), numpy.reshape(a_,(1,-1))),axis=1)
	    #x_ = x_.astype(float)

        # Compute error
        delta = self.bellman_error2(x,r,x_)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)
        self.Q.append(x, self.eta.value * delta)
        # Prune
        modelOrder = len(self.Q.D)
        self.Q.prune(self.eps * self.eta.value**2)
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

class KQLearningModelSCGD2(KQLearningModel2):
    def __init__(self, stateCount, actionCount, config):
        super(KQLearningModelSCGD2, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.
    def train(self, step, sample):
	self.eta.step(step)
	self.beta.step(step)
	# Unpyack sample
	
	s, a, r, s_ = sample
	# Compute error

	x = numpy.concatenate((numpy.reshape(numpy.array(s),(1,-1)), numpy.reshape(numpy.array(a),(1,-1))),axis=1)

	if s_ is None:
	    a_ = None
	    x_ = None
	else: 
	    a_ = self.Q.argmax(s_) 
	    x_ = numpy.concatenate((numpy.reshape(numpy.array(s_),(1,-1)), numpy.reshape(numpy.array(a_),(1,-1))),axis=1)	
	   
	delta = self.bellman_error2(x,r,x_)
	# Running average of TD-error
	self.y += self.beta.value * (delta - self.y)
	# Gradient step
	self.Q.shrink(1. - self.eta.value * self.lossL)

	if s_ is None:
	    self.Q.append(x, self.eta.value * self.y)
	else:
	    W = numpy.zeros((2,1))
	    W[0]  = -1.
	    W[1] = self.gamma
	    self.Q.append(numpy.vstack((x,x_)), -self.eta.value * self.y * W)

	# Prune
	modelOrder = len(self.Q.D)
	self.Q.prune(self.eps**2 * self.eta.value**2 / self.beta.value)
	modelOrder_ = len(self.Q.D)
	# Compute new error
	loss = 0.5*self.bellman_error2(x,r,x_)**2 + self.model_error()
	#print modelOrder_
	return (float(loss), float(modelOrder_))

# ==================================================
# An agent using Q-Learning

class KQLearningAgent2(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
	self.min_act = json.loads(config.get('MinAction'))
	self.max_act = json.loads(config.get('MaxAction'))
	self.max_model_order = config.getfloat('MaxModelOrder', 10000)
																
	self.min_act = numpy.reshape(self.min_act,(-1,1))
	self.max_act = numpy.reshape(self.max_act,(-1,1))

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'SCGD')
        self.save_steps = config.getint('SaveInterval', 100000)
        self.folder = config.get('Folder', 'exp')
	self.train_steps = config.getint('TrainInterval', 4)
        if algorithm.lower() == 'scgd':
            self.model = KQLearningModelSCGD2(self.stateCount, self.actionCount, config)
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
    def act(self, s, stochastic=True):
        "Decide what action to take in state s."
        #if stochastic and (random.random() < self.epsilon.value):
	#    #print("random")
	#    return numpy.random.uniform(self.min_act,self.max_act)
	#	#return self.action_space.sample()        
	#else:
	#print self.epsilon.value
        #a =  self.model.Q.argmax(s) + numpy.random.normal(0,   5* self.epsilon.value * (self.max_act - self.min_act))
        #a_temp = numpy.clip(a, self.min_act, self.max_act)
        #return a_temp

        "Decide what action to take in state s."
        if stochastic and (random.random() < self.epsilon.value):
	    #print("random")
	    a =  numpy.random.uniform(3*self.min_act,3*self.max_act)
            #a_temp = numpy.clip(a, self.min_act, self.max_act)
	    #return a_temp
		#return self.action_space.sample() 
 
	else:
            a =  self.model.Q.argmax(s)
	    #print a  
            #print numpy.clip(a, self.min_act, self.max_act)    
	    #print a
	a_temp = numpy.clip(a, self.min_act, self.max_act)
	
	return a_temp

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
	if len(self.model.Q.D) > self.max_model_order:
	    self.model.eps = self.model.eps * 2
	#self.model.eps = self.model.eps * 1
	if self.steps % self.save_steps == 0:
            with open(self.folder + '/kpolicy_model_' + str(int(self.steps / self.save_steps)) + '.pkl','wb') as f:
                pickle.dump(self.model.Q, f)

	#if self.steps % self.train_steps == 0:
        #	return self.model.train(self.steps, self.lastSample)
	#else:
	#	return (0,0)
	return self.model.train(self.steps, self.lastSample)
    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s,a,r,s_)
    def model_error(self):
        return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names
