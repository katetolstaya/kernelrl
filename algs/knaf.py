import copy, json, random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
import pdb
import pickle
# =====
# =============================================
# A POLK Normalized Advantage Function Algorithm

class KNAFModel(object):
    def __init__(self, stateCount, actionCount, config):

        # Get dimensions of V, pi and L
        self.dim_v = 1
        self.dim_p = actionCount
        self.dim_l = 1 #(1 + actionCount) * actionCount / 2 #TODO
        self.dim_a = self.dim_v + self.dim_p + self.dim_l

        # Get action space
        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        # Initialize L
        self.init_l = config.getfloat('InitL', 0.01)

        # Represent V, pi, L in one RKHS
        self.vpl = KernelRepresentation(stateCount, self.dim_a, config)

        # Learning rates
        self.eta_v = ScheduledParameter('LearningRateV', config)
        self.eta_p = ScheduledParameter('LearningRateP', config)
        self.eta_l = ScheduledParameter('LearningRateL', config)

        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-6)

        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)

        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

    def get_q(self, s, a):
        lmat = self.get_lmat(s)
        pi = self.get_pi(s)

        if self.dim_p > 1:
            return self.get_v(s) - 0.5 * ((a-pi).T).dot(lmat).dot(lmat.T).dot(a-pi)
        else:
            return np.array([self.get_v(s) - 0.5 * (a - pi)*lmat*lmat*(a - pi)])

    def get_v(self, s):
        return np.array([self.predictOne(s)[0,0]])

    def get_pi(self, s):
        pi = self.predictOne(s)[0,1:self.dim_p + 1]
        return np.reshape(np.clip(pi, self.min_act, self.max_act), (-1,))

    def get_lmat(self, s):
        lmat = np.zeros((self.dim_p, self.dim_p))
        temp = self.predictOne(s)
        if self.dim_p > 1:
            lmat[np.tril_indices(self.dim_p)] = temp[self.dim_p + 1:]
            return lmat + self.init_l * np.eye(self.dim_p)
        else:
            return np.array([temp[0,2] + self.init_l])

    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.get_q(s, a)
        else:
            return r + self.gamma * self.get_v(s_) - self.get_q(s, a)

    def model_error(self):
        return 0.5 * self.lossL * self.vpl.normsq()

    def predict(self, s):  # Predict the Q function values for a batch of states.
        return self.vpl(s)

    def predictOne(self, s):  # Predict the Q function values for a single state.
        return self.vpl(np.reshape(s,(1,-1)))

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')

    def train(self, step, sample):
        self.eta_v.step(step)
        self.eta_p.step(step)
        self.eta_l.step(step)
        #self.beta.step(step)

        # Unpack sample
        s, a, r, s_ = sample

        # Compute error
        delta = self.bellman_error(s, a, r, s_)

        # Gradient step
        self.vpl.shrink(1. - self.lossL)

        # V gradient
        W = np.zeros((self.dim_a,))
        W[0] = -1 * self.eta_v.value
        lmat = self.get_lmat(s)
        pi = self.get_pi(s)


        # Pi gradient
        if self.dim_p > 1:
            W[1:self.dim_p + 1] = -self.eta_p.value * np.matmul(np.matmul(lmat, np.transpose(lmat)), a - pi)
            lgrad_temp = np.matmul(np.matmul(np.transpose(lmat), a - pi), np.transpose(a - pi))
        else:
            lgrad_temp = lmat * (a - pi) * (a - pi)
            W[1] = -self.eta_p.value * lmat * lmat *( a - pi)

        if self.dim_p > 1:
            W[self.dim_p + 1:self.dim_a] = np.reshape(lgrad_temp[np.tril_indices(self.dim_p)], (-1, 1)) * self.eta_l.value
        else:
            W[-1] = lgrad_temp  * self.eta_l.value

        # Check for model divergence!
        if np.abs(delta) > 50 and False:
            print ("Divergence!")
            print (pi)
            print (lmat)
            print (delta)
        self.vpl.append(np.array(s), - delta * np.reshape(W, (1, -1)))
        # Prune

        #if step % 10 == 0 :
        self.vpl.prune(self.eps)

        modelOrder_ = len(self.vpl.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_) ** 2  # + self.model_error()
        return (float(loss), float(modelOrder_))  # ==================================================

# An agent using Q-Learning

class KNAFAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.steps = 0         # How many steps we have observed
        self.gamma = config.getfloat('RewardDiscount')

        # ---- Configure exploration
        self.epsilon = ScheduledParameter('ExplorationRate', config)
        self.epsilon.step(0)
        self.noise_var = ScheduledParameter('NoiseVariance', config)
        self.noise_var.step(0)
        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        # ---- Initialize model


        if config.get('LoadModel'):
            fname = config.get('LoadModel')
            self.model = pickle.load(open(fname,"rb"))
        else:
           self.model = KNAFModel(self.stateCount, self.actionCount, config)

    def act(self, s, stochastic=True):
        # "Decide what action to take in state s."
        a = self.model.get_pi(s)
        #noise = np.sum(self.model.vpl.kernel.f(s, self.model.vpl.D))
        #print (noise)
        #if noise < 1:
        #    noise = 1
        

        if stochastic: # if exploration, add noise
            a = a + np.random.normal(0,self.noise_var.value,self.actionCount)
        a = np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))
        return a

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)
        self.noise_var.step(self.steps)

        #if self.steps % 10000 == 0 :
        #    with open('rob24_model'+str(int(self.steps/10000))+'.txt', 'wb') as f:
        #        pickle.dump(self.model, f)


    def improve(self):
        return self.model.train(self.steps, self.lastSample)

    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s, a, r, s_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
