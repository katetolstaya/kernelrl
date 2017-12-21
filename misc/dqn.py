from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
import keras.backend as K
import numpy as np
import sys, math, random

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

def huber_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean(K.sqrt(1+K.square(err))-1, axis=-1)

# ==================================================
# The DQN approximation of the Q function

class DQNModel(object):
    def __init__(self, stateCount, actionCount):
        self.stateCount  = stateCount
        self.model = Sequential()
        self.model.add(Dense(output_dim=64, activation='relu', input_dim=stateCount))
        self.model.add(Dense(output_dim=actionCount, activation='linear'))
        self.model.compile(loss='mse', optimizer=RMSprop(lr=0.00025), metrics=[mean_q])
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = {
                'configuration' : self.model.to_json(),
                'weights'       : self.model.get_weights()
                }
        return state
    def __setstate__(self, state):
        model = model_from_json(state['model']['configuration'])
        model.set_weights(state['model']['weights'])
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025), metrics=[mean_q])
        state['model'] = model
        self.__dict__.update(state)
    def train(self, x, y):
        "Perform supervised training step with batch."
        return self.model.train_on_batch(x, y)
    def predict(self, s):
        "Predict the Q function values for a batch of states."
        return self.model.predict(s)
    def predictOne(self, s):
        "Predict the Q function values for a single state."
        return self.model.predict(s.reshape(1, self.stateCount)).flatten()
    def copy(self, other):
        self.model.set_weights(other.model.get_weights())
    @property
    def metrics_names(self):
        return self.model.metrics_names

# ==================================================
# An agent using Double Q-Learning with Prioritized Experience Replay

class DDQNPERAgent(object):
    def __init__(self, env, config):
        self.stateCount = env.stateCount
        self.actionCount = env.actionCount
        # Moving model
        self.modelMoving = DQNModel(self.stateCount, self.actionCount)
        # Target model
        self.modelTarget = DQNModel(self.stateCount, self.actionCount)
        # Our memory gets copied over from initialization
        self.memory = None
        # How many steps we have observed
        self.steps = 0

        # ---- Configure batch size
        self.batchSize = config.getint('MinibatchSize', 16)

        # ---- Configure exploration
        # Exploration rates
        epsilonStart = config.getfloat('InitialExplorationRate', 1.)
        epsilonFinal = config.getfloat('FinalExplorationRate', 0.)
        epsilonSteps = config.getint('ExplorationStop', sys.maxint)
        # Epsilon decay rate
        self.epsilonDecay = -math.log(0.01) / float(epsilonSteps)
        # Epsilon minimum
        self.epsilonMin = epsilonFinal
        # Epsilon range
        self.epsilonRange = epsilonStart - epsilonFinal
        # Current epsilon
        self.epsilon = epsilonStart

        # ---- Configure rewards
        self.gamma = config.getfloat('RewardDiscount')

        # ---- Configure priority experience replay
        self.eps   = config.getfloat('ExperiencePriorityMinimum', 0.01)
        self.alpha = config.getfloat('ExperiencePriorityExponent', 1.)

        # ---- Configure moving/target switching
        self.targetUpdate = config.getint('TargetUpdateFrequency', 1000)

    def _getTargets(self, batch):
        no_state = np.zeros(self.stateCount)

        states  = np.array([e[0] for (_,e) in batch])
        states_ = np.array([no_state if e[3] is None else e[3] for (_,e) in batch])

        p   = self.modelMoving.predict(states)
        p_  = self.modelMoving.predict(states_)
        pT_ = self.modelTarget.predict(states_)

        x = np.zeros((len(batch), self.stateCount))
        y = np.zeros((len(batch), self.actionCount))
        errors = np.zeros(len(batch))

        for i, (_, e) in enumerate(batch):
            s = e[0]; a = e[1]; r = e[2]; s_ = e[3]
            t = p[i]
            oldQsa = t[a]
            if s_ is None:
                t[a] = r
            else:
                # Double Q-Learning return
                t[a] = r + self.gamma * pT_[i][p_[i].argmax()]
            x[i] = s
            y[i] = t
            errors[i] = abs(oldQsa - t[a])

        return x, y, errors

    def act(self, s, stochastic=True):
        "Decide what action to take in state s."
        if stochastic and (random.random() < self.epsilon):
            return random.randint(0, self.actionCount-1)
        else:
            return self.modelMoving.predictOne(s).argmax()

    def observe(self, sample):
        "Add sample (s, a, r, s') to memory."
        _, _, errors = self._getTargets([(0, sample)])
        self.memory.add(sample, (errors[0] + self.eps) ** self.alpha)

        if self.steps % self.targetUpdate == 0:
            self.modelTarget.copy(self.modelMoving)

        # exploration decrease schedule
        self.steps += 1
        self.epsilon = self.epsilonMin + self.epsilonRange*math.exp(-self.epsilonDecay*self.steps)

    def improve(self):
        "Replay memories and improves."
        batch = self.memory.sample(self.batchSize)
        x, y, errors = self._getTargets(batch)

        # update errors
        for (idx, _), error in zip(batch, errors):
            self.memory.update(idx, (error + self.eps) ** self.alpha)

        return self.modelMoving.train(x, y)

    @property
    def metrics_names(self):
        return self.modelMoving.metrics_names

# ==================================================

class RandomAgent:
    def __init__(self, env, cfg):
        memoryCapacity = cfg.getint('MemoryCapacity', 100000)
        self.memory = PrioritizedMemory(memoryCapacity)
        self.eps   = cfg.getfloat('ExperiencePriorityMinimum', 0.01)
        self.alpha = cfg.getfloat('ExperiencePriorityExponent', 1.0)
        self.actionCount = env.actionCount
    def act(self, s):
        return random.randint(0, self.actionCount-1)
    def observe(self, sample):
        error = abs(sample[2])
        self.memory.add(sample, (error + self.eps) ** self.alpha)
    def improve(self):
        return ()

