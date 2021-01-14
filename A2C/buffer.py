import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#PPO replay buffer，return all
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.dist_entropy = []
        self.rewards = []
        self.is_terminals = []

    def reset_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.dist_entropy
        del self.rewards[:]
        del self.is_terminals[:]
        self.actions = []
        self.states = []
        self.logprobs = []
        self.dist_entropy = []
        self.rewards = []
        self.is_terminals = []


    def split(self,batch_size):
        split_res = []
        length = len(self.actions)
        if batch_size is None:
            split_res.append((0, length-1))
            return split_res
        for idx in range(0, length, batch_size):
            if idx + batch_size < length:
                split_res.append((idx, idx + batch_size))
            else:
                split_res.append((length-idx-1, length-1))
        return split_res

