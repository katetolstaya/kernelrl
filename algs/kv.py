import numpy as np

from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
from policy import make_policy


# ==================================================
# Kernelized Policy Evaluation techniques

# SCGD-based Policy Evaluation
class KSCGDModel(object):
    def __init__(self, stateCount, config):
        self.V = KernelRepresentation(stateCount, 1, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-6)
        # Representation error budget
        #self.eps = config.getfloat('RepresentationError', 1.0)
        self.eps =ScheduledParameter('RepresentationError', config)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

        # TD-loss expectation approximation rate
        self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.
    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.V(s)
        else:
            return r + self.gamma*self.V(s_) - self.V(s)
    def model_error(self):
        return 0.5*self.lossL*self.V.normsq()
    def train(self, step, sample):
        self.eta.step(step)
        self.eps.step(step)
        self.beta.step(step)
        # Unpack sample
        s, a, r, s_ = sample
        # Compute error
        delta = self.bellman_error(s,a,r,s_)
        # Running average
        self.y += self.beta.value * (delta - self.y)
        # Gradient step
        self.V.shrink(1. - self.eta.value * self.lossL)
        if s_ is None:
            self.V.append(s, -self.eta.value*self.y*np.array([[-1.]]))
        else:
            W = np.zeros((2,1)); W[0] = -1; W[1] = self.gamma
            self.V.append(np.vstack((s,s_)), -self.eta.value*self.y*W)
        # Prune
        modelOrder = len(self.V.D)

        self.V.prune(self.eps.value * self.eta.value**2)
        modelOrder_ = len(self.V.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_)**2 + self.model_error()
        return (float(loss), float(modelOrder_), self.eta.value, self.beta.value)
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order', 'Step Size', 'Averaging Coefficient')

# TD-based Policy Evaluation
class KTDModel(object):
    def __init__(self, stateCount, config):
        self.V = KernelRepresentation(stateCount, 1, config)
        # Learning rate
        self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-6)
        # Representation error budget
        self.eps =ScheduledParameter('RepresentationError', config)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')
    def bellman_error(self, s, a, r, s_):
        if s_ is None:
            return r - self.V(s)
        else:
            return r + self.gamma*self.V(s_) - self.V(s)
    def model_error(self):
        return 0.5*self.lossL*self.V.normsq()
    def train(self, step, sample):
        self.eta.step(step)
        self.eps.step(step)
        # Unpack sample
        s, a, r, s_ = sample
        # Compute error
        delta = self.bellman_error(s,a,r,s_)
        # Gradient step
        self.V.shrink(1. - self.eta.value * self.lossL)
        self.V.append(s, self.eta.value * delta)
        # Prune
        modelOrder = len(self.V.D)
        self.V.prune(self.eps.value * self.eta.value**2)
        modelOrder_ = len(self.V.D)
        # Compute new error
        loss = 0.5*self.bellman_error(s,a,r,s_)**2 + self.model_error()
        return (float(loss), float(modelOrder_), self.eta.value)
    @property
    def metrics_names(self):
        return ('Training Loss','Model Order', 'Step Size')

# # ==================================================
# # Deep Network value function approximation
# class DVNModel(object):
#     def __init__(self, stateCount, config):
#         self.stateCount = stateCount
#         self.V = Sequential()
#         self.V.add(Dense(output_dim=64, activation='relu', input_dim=stateCount))
#         self.V.add(Dense(output_dim=1, activation='linear'))
#         self.V.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
#         self.gamma = config.getfloat('RewardDiscount')
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         state['V'] = {
#                 'configuration' : self.V.to_json(),
#                 'weights'       : self.V.get_weights()
#                 }
#         return state
#     def __setstate__(self, state):
#         model = model_from_json(state['V']['configuration'])
#         model.set_weights(state['V']['weights'])
#         model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
#         state['V'] = model
#         self.__dict__.update(state)
#     def predictOne(self, s):
#         return self.V.predict(s.reshape(1, self.stateCount)).flatten()
#     def bellman_error(self, s, a, r, s_):
#         if s_ is None:
#             return r - self.predictOne(s)
#         else:
#             return r + self.gamma*self.predictOne(s_) - self.predictOne(s)
#     def model_error(self):
#         return 0.
#     #def train(self, step, sample):


# ==================================================
# An agent doing policy evaluation
class KVAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.policy = make_policy(config)
        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'TD')
        if algorithm.lower() == 'scgd':
            self.model = KSCGDModel(self.stateCount, config)
        elif algorithm.lower() == 'td':
            self.model = KTDModel(self.stateCount, config)
        else:
            raise ValueError('Unknown algorithm: {}'.format(algorithm))
        self.steps = 0
    def act(self, s, stochastic=True):
        return self.policy.select(s)
    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1
    def improve(self):
        return self.model.train(self.steps, self.lastSample)
    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s,a,r,s_)
    def model_error(self):
        return self.model.model_error()
    @property
    def metrics_names(self):
        return self.model.metrics_names

