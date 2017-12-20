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
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.Q = KernelRepresentation(stateCount + actionCount , 1, config)
        self.pi = KernelRepresentation(stateCount, actionCount,  config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        self.pi_step = config.getfloat('PolicyStep')
        self.pivar = ScheduledParameter('PiVar', config)

        self.min_act = json.loads(config.get('MinAction'))
        self.max_act = json.loads(config.get('MaxAction'))

        self.min_act = np.reshape(self.min_act, (-1, 1))
        self.max_act = np.reshape(self.max_act, (-1, 1))

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
        self.pi_variance = np.reshape(json.loads(config.get('PiVariance')), (-1, 1)) #np.ones((actionCount,))

    def bellman_error(self, s, a, r, s_):
        x = np.concatenate((np.reshape(s, (1, -1)), np.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            a_ = self.Q.argmax(s_)
            x_ = np.concatenate((np.reshape(s_, (1, -1)), np.reshape(a_, (1, -1))), axis=1)
            return r + self.gamma * self.Q(x_) - self.Q(x)

    def bellman_error2(self, x, r, x_):
        if x_ is None:
            return r - self.Q(x)
        else:
            return r + self.gamma * self.Q(x_) - self.Q(x)

    def model_error(self):
        return 0.5 * self.lossL * self.Q.normsq()

    def train(self, step, sample):
        pass

    def predict(self, s):
        pass
        # return self.Q(s)

    def predictOne(self, s):
        return self.pi(s)
        # "Predict the Q function values for a single state."
        # return self.Q.argmax(s)
        # return self.Q(s.reshape(1, len(s))).flatten()

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')



class KDPGModelSCGD(KDPGModel):
    def __init__(self, stateCount, actionCount, config):
        super(KDPGModelSCGD, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.
    def train(self, step, sample):
        self.eta.step(step)
        self.pivar.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_, a_ = sample
        s = copy.deepcopy(s)
        s_ = copy.deepcopy(s_)
        #x = copy.deepcopy(np.concatenate((np.reshape(s,(1,-1)), np.reshape(a,(1,-1))),axis=1))

        ############
        x = copy.deepcopy(np.concatenate((np.reshape(np.array(s), (1, -1)), np.reshape(np.array(a), (1, -1))), axis=1))

        if s_ is None:
            a_ = None
            x_ = None
        else:
            #a_ = self.Q.argmax(s_)
            x_ = copy.deepcopy(np.concatenate((np.reshape(np.array(s_), (1, -1)), np.reshape(np.array(a_), (1, -1))),
                                   axis=1))
        ###############
        # Update Q
        # Compute error
        delta = self.bellman_error2(x, r, x_)
        # Running average of TD-error
        self.y += self.beta.value * (delta - self.y)
        # Gradient step
        self.Q.shrink(1. - self.eta.value * self.lossL)

        if s_ is None:
            self.Q.append(x, self.eta.value * self.y)
        else:
            W = np.zeros((2, 1))
            W[0] = -1.
            W[1] = self.gamma * 0
            self.Q.append(np.vstack((x, x_)), -self.eta.value * self.y * W)

        self.Q.prune(self.eps ** 2 * self.eta.value ** 2 / self.beta.value)

        #############
        # update pi


        temp_mean_a = self.pi(np.reshape(np.array(s), (1, -1)))

        mean_a = np.clip(temp_mean_a, self.min_act.T, self.max_act.T)

        #temp1 = np.logical_and(np.any(mean_a <= self.max_act, axis=1),np.any(mean_a >= self.min_act, axis=1))
        #print str(mean_a) + ' ' + str(a)

        a2 = np.clip(mean_a*2 - a, self.min_act.T, self.max_act.T)

        x2 = np.concatenate((np.reshape(np.array(s), (1, -1)), np.reshape(np.array(a2), (1, -1))),axis=1)
        #dpi = 1/(2*(1-self.gamma))*(self.Q(x) - self.Q(x2)) * np.reshape((a-a2), (-1,)) * 1/self.pi_variance

        dpi =  (self.Q(x) - self.Q(x2)) * (a - mean_a)

        self.pi.shrink(1. - self.pi_step * self.lossL)
        self.pi.append(np.reshape(s,(1,-1)),  np.reshape(self.pi_step * dpi,(1,-1))) #* -self.y

        self.pi.prune(self.eps_pi *  self.pi_step**2)





        # Prune
        #modelOrder = len(self.Q.D)
        #print str(self.pi.model_order()) + "    " + str(self.Q.model_order())
        modelOrder_ = self.pi.model_order()  + self.Q.model_order()
        #print self.pi.model_order()
        #print self.Q.model_order()

        # Compute new error
        loss = 0.5*self.bellman_error2(x, r, x_)**2 + self.model_error()
        return (float(loss), float(modelOrder_))

# ==================================================
# An agent using SARSA

class KDPGAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = json.loads(config.get('MinAction'))
        self.max_act = json.loads(config.get('MaxAction'))

        self.min_act = np.reshape(self.min_act, (-1, 1))
        self.max_act = np.reshape(self.max_act, (-1, 1))

        self.act_mult = config.getfloat('ActMultiplier', 1)
	# 

        # We can switch between SCGD and TD learning here
        self.save_steps = config.getint('SaveInterval', 1000000000)
        self.folder = config.get('Folder', 'exp')
        algorithm = config.get('Algorithm', 'SCGD')
        if algorithm.lower() == 'scgd':
            self.model = KDPGModelSCGD(self.stateCount, self.actionCount, config)
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
        self.model.pivar.step(self.steps)
        "Decide what action to take in state s."
        if stochastic and (random.random() < self.epsilon.value):
            a = np.random.uniform(self.act_mult * self.min_act, self.act_mult * self.max_act)
        else:
            mean_a = self.model.predictOne(np.reshape(s,(1,-1)))
            #print mean_a
            a = np.random.normal(mean_a, self.model.pivar.value * self.model.pi_variance.T)
            #print mean_a
        a_temp = np.reshape(np.clip(a, self.min_act.T, self.max_act.T), (-1,))
        return a_temp
    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)

    def improve(self):
	    #if self.steps % self.save_steps == 0:
        #    with open(self.folder + '/kpolicy_pi_' + str(int(self.steps / self.save_steps)) + '.pkl','wb') as f:
        #        pickle.dump(self.model.pi, f)
        #    with open(self.folder + '/kpolicy_model_' + str(int(self.steps / self.save_steps)) + '.pkl','wb') as f:
        #        pickle.dump(self.model.Q, f)
        return self.model.train(self.steps, self.lastSample)
    def bellman_error(self, s, a, r, s_, a_):
        return self.model.bellman_error(s,a,r,s_,a_)
    def model_error(self):
        return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names

