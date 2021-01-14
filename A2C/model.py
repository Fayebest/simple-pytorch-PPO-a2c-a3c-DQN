import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):  # 对actor结果进行采样得到需要执行的动作
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)  # 按照action_probs中的概率对其进行采样，采样出一个值，其是action_probs中对应的下标
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))  # 计算其对概率对数
        memory.dist_entropy.append(dist.entropy())

        return action.item()  # 返回一对action 的 sample



class CNNModel(nn.Module):
    def __init__(self):
        return


class RNNModel(nn.Module):
    def __init__(self):
        return