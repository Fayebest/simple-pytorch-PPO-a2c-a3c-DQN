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

        return action.item()  # 返回一对action 的 sample

    def evaluate(self, state, action):  # 计算输出动作概率，状态价值，交叉熵等信息，评估器
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()  # 计算一下action_probs的交叉熵  p_log_p = self.logits * self.probs

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class CNNModel(nn.Module):
    def __init__(self):
        return


class RNNModel(nn.Module):
    def __init__(self):
        return