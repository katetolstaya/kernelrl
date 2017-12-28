import copy, json, random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation

# ==================================================
# A POLK Normalized Advantage Function Algorithm

class KNAFModel(object):
    def __init__(self, stateCount, actionCount, config):

        # Get dimensions of V, pi and L
        self.dim_v = 1
        self.dim_p = actionCount
        self.dim_l = (1 + actionCount) * actionCount / 2
        self.dim_a = 1 + actionCount + (1 + actionCount) * actionCount / 2

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
        return self.get_v(s) - 0.5 * ((a-pi).T).dot(lmat).dot(lmat.T).dot(a-pi)

    def get_v(self, s):
        return self.predictOne(s)[0]

    def get_pi(self, s):
        pi = self.predictOne(s)[1:self.dim_p + 1]
        return np.reshape(np.clip(pi, self.min_act, self.max_act), (-1,))

    def get_lmat(self, s):
        lmat = np.zeros((self.dim_p, self.dim_p))
        lmat[np.tril_indices(self.dim_p)] = self.predictOne(s)[self.dim_p + 1:]
        return lmat + self.init_l * np.eye(self.dim_p)

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
        return self.vpl(s)

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')

    def train(self, step, sample):
        self.eta_v.step(step)
        self.eta_p.step(step)
        self.eta_l.step(step)
        self.beta.step(step)

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
        W[1:self.dim_p + 1] = -self.eta_p.value * np.matmul(np.matmul(lmat, np.transpose(lmat)), a - pi)

        # L matrix gradient
        lgrad_temp = np.matmul(np.matmul(np.transpose(lmat), a - pi), np.transpose(a - pi))
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
        self.vpl.append(s, - delta * np.reshape(W, (1, -1)))
        # Prune
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
        self.model = KNAFModel(self.stateCount, self.actionCount, config)

    def act(self, s, stochastic=True):
        # "Decide what action to take in state s."
        a = self.model.get_pi(s)
        if stochastic and (random.random() < self.epsilon.value): # if exploration, add noise
            a = a + np.random.normal(0,self.noise_var.value,self.actionCount)
        return np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
        self.epsilon.step(self.steps)
        self.noise_var.step(self.steps)

    def improve(self):
        return self.model.train(self.steps, self.lastSample)

    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s, a, r, s_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
