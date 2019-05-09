import json
import pickle

import numpy


# ==================================================
# A POLK Q-Learning Model

class QTestModel2(object):
    def __init__(self, stateCount, actionCount, config):
        self.policy_file = config.get('PolicyFile', 'file')
        # self.Q = KernelRepresentation(stateCount+ actionCount,1, config)


        f = open(self.policy_file + '.pkl', 'rb')
        self.Q = pickle.load(f)

        # Learning rate
        # self.eta   = ScheduledParameter('LearningRate', config)
        # Regularization
        self.lossL = config.getfloat('Regularization', 1e-4)
        # Representation error budget
        self.eps = config.getfloat('RepresentationError', 1.0)
        # Reward discount
        self.gamma = config.getfloat('RewardDiscount')

    def bellman_error(self, s, a, r, s_):
        x = numpy.concatenate((numpy.reshape(s, (1, -1)), numpy.reshape(a, (1, -1))), axis=1)
        if s_ is None:
            return r - self.Q(x)
        else:
            a_ = self.Q.argmax(s_)
            x_ = numpy.concatenate((numpy.reshape(s_, (1, -1)), numpy.reshape(a_, (1, -1))), axis=1)
            return r + self.Q(x_) - self.Q(x)

    def bellman_error2(self, x, r, x_):
        if x_ is None:
            return r - self.Q(x)
        else:
            return r + self.Q(x_) - self.Q(x)

    def model_error(self):
        return 0.5 * self.lossL * self.Q.normsq()

    def train(self, step, sample):
        pass

    def predict(self, s):
        pass
        # return self.Q(s)

    def predictOne(self, s):
        pass
        # "Predict the Q function values for a single state."
        # return self.Q.argmax(s)
        # return self.Q(s.reshape(1, len(s))).flatten()

    @property
    def metrics_names(self):
        return ('Training Loss', 'Model Order')


class QTestModelTD2(QTestModel2):
    def train(self, step, sample):
        # self.eta.step(step)
        s, a, r, s_ = sample
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_) ** 2 + self.model_error()
        return (float(loss), float(modelOrder_))


class QTestModelSCGD2(QTestModel2):
    def __init__(self, stateCount, actionCount, config):
        super(QTestModelSCGD2, self).__init__(stateCount, actionCount, config)
        # TD-loss expectation approximation rate
        # self.beta  = ScheduledParameter('ExpectationRate', config)
        # Running estimate of our expected TD-loss
        self.y = 0.

    def train(self, step, sample):
        # self.eta.step(step)
        # self.beta.step(step)

        s, a, r, s_ = sample
        modelOrder_ = len(self.Q.D)
        # Compute new error
        loss = 0.5 * self.bellman_error(s, a, r, s_) ** 2 + self.model_error()
        return (float(loss), float(modelOrder_))


# ==================================================
# An agent using Q-Learning

class QTestAgent2(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        self.min_act = json.loads(config.get('MinAction'))
        self.max_act = json.loads(config.get('MaxAction'))
        self.min_act = numpy.reshape(self.min_act, (-1, 1))
        self.max_act = numpy.reshape(self.max_act, (-1, 1))

        # self.max_model_order = config.getfloat('MaxModelOrder', 10000)

        # We can switch between SCGD and TD learning here
        algorithm = config.get('Algorithm', 'SCGD')
        # self.save_steps = config.getint('SaveInterval', 100000)
        # self.folder = config.get('Folder', 'exp')
        # self.train_steps = config.getint('TrainInterval', 4)
        if algorithm.lower() == 'scgd':
            self.model = QTestModelSCGD2(self.stateCount, self.actionCount, config)
        elif algorithm.lower() == 'td':
            self.model = QTestModelTD2(self.stateCount, self.actionCount, config)
        else:
            raise ValueError('Unknown algorithm: {}'.format(algorithm))
        # How many steps we have observed
        self.steps = 0
        # ---- Configure exploration
        # self.epsilon = ScheduledParameter('ExplorationRate', config)
        # self.epsilon.step(0)
        # ---- Configure rewards
        self.gamma = config.getfloat('RewardDiscount')

    def act(self, s, stochastic=True):
        "Decide what action to take in state s."
        a = self.model.Q.argmax(s)
        a_temp = numpy.clip(a, self.min_act, self.max_act)
        return a_temp

    def observe(self, sample):
        self.lastSample = sample
        self.steps += 1

    def improve(self):
        return self.model.train(self.steps, self.lastSample)

    def bellman_error(self, s, a, r, s_):
        return self.model.bellman_error(s, a, r, s_)

    def model_error(self):
        return self.model.model_error()

    @property
    def metrics_names(self):
        return self.model.metrics_names
