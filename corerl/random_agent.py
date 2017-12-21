import numpy as np

from corerl.memory import PrioritizedMemory, Memory


# ==================================================
class RandomAgent:
    def __init__(self, env, cfg):
        memoryCapacity = cfg.getint('MemoryCapacity', 10000)
        memoryType = cfg.get('MemoryType','Priority')
        if memoryType == 'Priority':
            self.memory = PrioritizedMemory(memoryCapacity)
            self.eps = cfg.getfloat('ExperiencePriorityMinimum', 0.01)
            self.alpha = cfg.getfloat('ExperiencePriorityExponent', 1.0)
        else:
            self.memory = Memory(memoryCapacity)
        self.min_act = env.env.action_space.low
        self.max_act = env.env.action_space.high

    def act(self, s):
        a = np.random.uniform(self.min_act,self.max_act)
        return np.reshape(np.clip(a, self.min_act, self.max_act), (-1,))

    def observe(self, sample):
        error = abs(sample[2])
        if self.memory is PrioritizedMemory:
            self.memory.add(sample, (error + self.eps) ** self.alpha)
        else:
            self.memory.add(sample, 0)

    def improve(self):
        return ()