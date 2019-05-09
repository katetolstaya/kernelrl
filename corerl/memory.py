import numpy as np
import random


def pairs(l):
    for i in range(0, len(l) - 1):
        yield l[i], l[i + 1]


# ==================================================

class Memory(object):
    '''circularlist storage of samples.'''

    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0
        self.length = 0
        self.data = [None for _ in range(capacity)]

    def add(self, value, p):
        "Add sample to memory with score (that is ignored)."
        self.data[self.index] = [p, value]
        self.index = (self.index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, n):
        if n < self.length:
            return random.sample(self.data[:self.length], n)
        else:
            return random.sample(self.data, n)

    def is_full(self):
        return self.length == self.capacity

    def remaining(self):
        return (self.capacity - self.length)


# ==================================================

class PrioritizedMemory(object):
    '''Binary tree stored in breadth-first order as array.

    For unbalanced trees, the order in memory is NOT the expected order.
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0
        self.length = 0
        self.nodes = np.zeros(2 * capacity - 1)
        self.data = [None for _ in range(capacity)]

    def total(self):
        "Total score in memory."
        return self.nodes[0]

    def add(self, value, p):
        "Add sample to memory with score."
        self.data[self.index] = value
        self.update(self.capacity - 1 + self.index, p)
        self.index = (self.index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def update(self, idx, p):
        "Update score at given index."
        delta = p - self.nodes[idx]
        while idx > 0:
            self.nodes[idx] += delta
            idx = (idx - 1) // 2
        self.nodes[0] += delta

    def get(self, p):
        "Get last sample whose cumulative sum does not exceed value."
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1

            if p <= self.nodes[left]:
                idx = left
            else:
                idx = right
                p -= self.nodes[left]
        return idx, self.data[idx - self.capacity + 1]

    def sample(self, n):
        '''Return weighted batch of n samples.

        Binned according to Appendix B.2.1 of https://arxiv.org/pdf/1511.05952
        '''
        samples = range(n)
        for i, bounds in enumerate(pairs(np.linspace(0, self.total(), n + 1))):
            samples[i] = self.get(random.uniform(bounds[0], bounds[1]))
        return samples

    def is_full(self):
        return self.length == self.capacity

    def remaining(self):
        return self.capacity - self.length


# ==================================================

def make_memory(config):
    memory_type = config.get('MemoryType', fallback='Memory').lower()
    if memory_type == 'memory':
        return Memory(config.getint('MemoryCapacity', fallback=100000))
    elif memory_type == 'prioritizedmemory':
        return PrioritizedMemory(config.getint('MemoryCapacity', fallback=100000))
    else:
        raise ValueError('Unknown memory type: s' % config.get('MemoryType'))
