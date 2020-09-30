import numpy as np
import random


class Memory():
    def __init__(self, size):
        self.max_size = size
        self.memory = []
        self.position = 0


    def push(self,state , action, reward, n_state):
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.position] = (state,action, reward, n_state)
        self.position = (self.position+1) % self.max_size

    def sample(self, batch_size):
        if 200 > len(self.memory):
            return None
        else:
            return random.sample(self.memory, batch_size)

