from function import KernelRepresentation
import numpy as np
import sys, math, random
from core import ScheduledParameter
import copy
import cPickle as pickle

# ==================================================
# A POLK Policy improvements Model

class KQTabModel(object):
    def __init__(self, config):

        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        self.gamma = config.getfloat('RewardDiscount')

	self.max_a = 2
	self.max_p = 10
	self.max_v = 5
	
	self.var_a = 0.1
	self.var_p = 1
	self.var_v = 0.5
	
	self.n_a = 2*int(self.max_a/self.var_a)+1
	self.n_v = 2*int(self.max_v/self.var_v)+1
	self.n_p = 2*int(self.max_p/self.var_p)+1

	self.table = np.random.normal(0.,0.001,(self.n_p, self.n_v, self.n_a))
		
    #def bellman_error(self, s, a, r, s_, a_):
    # x = np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1)
    #    if s_ is None:
    #        return r - self.Q(x)
    #    else:
    #        x_ = np.concatenate((np.reshape(s_,(1,-1)), np.reshape(a_,(1,-1))),axis=1)
    #        return r + self.Q(x_) - self.Q(x)
    #def model_error(self):
    #    return 0.5*self.lossL*self.Q.normsq()
    #def predict(self, s):
	#pass        
	##"Predict the Q function values for a batch of states."
    #    #return self.Q(s)
    #def predictOne(self, s):
	#pass        
	#"Predict the Q function values for a single state."
        #return self.Q(s.reshape(1, len(s))).flatten()

    def getStateIndex(self,s):
	s1 = s[0]
	s2 = s[1]
	return  (int((s1 + self.max_p)/self.var_p), int((s2 + self.max_v)/self.var_v))

    def getActionIndex(self,a):
	return int((a + self.max_a)/self.var_a)

    def getAction(self,na):
	return na*self.var_a - self.max_a

    def getRandomAction(self):
	return np.random.uniform(-self.max_a,self.max_a,(1,1))
	

    @property
    def metrics_names(self):
        return ('Training Loss')

    def train(self, step, sample):
	s, a, r, s_ = sample
		
	(s1n,s2n) = self.getStateIndex(s)
	an = self.getActionIndex(a)
	
        if s_ is None:
		td = r-self.table[s1n,s2n,an]
	else:
		(s1_n,s2_n) = self.getStateIndex(s_)
		a_n = np.argmax(self.table[s1_n,s2_n,:])
		td = r+self.gamma*self.table[s1_n, s2_n,a_n]-self.table[s1n,s2n,an]

	self.table[s1n,s2n,an] += self.eta.value * td
	return float(td),0

# ==================================================
# An agent using policy improvement

class KQTabAgent(object):
    def __init__(self, env, config):
	
	self.config = config
	
	self.folder = config.get('Folder', 'exp')
	


	self.dim_s = env.stateCount
	self.dim_a = env.actionCount
	
	self.sarsa_steps = config.getint('SARSASteps', 100000)
	#self.prune_steps = 100

	self.last_avg_error = 0
	self.avg_error = 0
	
	self.model = KQTabModel(config)

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
	    action = self.model.getRandomAction()
            #return random.randint(0, self.actionCount-1)
        else:
	    (s1n,s2n) = self.model.getStateIndex(s)
            action = self.model.getAction(np.argmax(self.model.table[s1n,s2n,:]))
	    #print action
	return np.array([action])
	    #return self.model.predictOne(s).argmax()
    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
	loss = self.model.train(self.steps, self.lastSample)

	if self.steps % self.sarsa_steps == 0:
		#self.epsilon.step(self.steps)
		with open(self.folder + '/kq_model_' + str(int(self.steps / self.sarsa_steps)) + '.pkl','wb') as f:
			pickle.dump(self.model.table, f)


		#self.model.reset()
	return loss

    #def bellman_error(self, s, a, r, s_, a_):
    #    return self.model.bellman_error(s,a,r,s_,a_)
    #def model_error(self):
    #    return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names

