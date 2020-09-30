import torch
import torch.nn as nn
from simPPO import model
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)


        self.MseLoss = nn.MSELoss()

    def ramdom_sample(self):
        index = np.random

    def update(self, memory, batch_size = None):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)


        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        split_res = memory.split(batch_size)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for split_idx in split_res:
                split_old_states = old_states[split_idx[0]:split_idx[1]]
                split_old_actions = old_actions[split_idx[0]:split_idx[1]]
                split_old_logprobs = old_logprobs[split_idx[0]:split_idx[1]]
                split_rewards = rewards[split_idx[0]:split_idx[1]]

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(split_old_states, split_old_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - split_old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = split_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,
                                                                     split_rewards) - 0.01 * dist_entropy  # loss加个负号，梯度上升就成了梯度下降
                #loss = loss.double()
              #  print(loss)
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()





