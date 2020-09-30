from DQN import Model
import torch.nn as nn
import torch
import math
import random
import torch.nn.functional as F

class dqn(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_layer,gamma,lr):
        super(dqn, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = Model.net(self.state_dim, self.action_dim, n_latent_layer)

        # for m in self.policy_net.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.xavier_uniform_(m.weight)

        self.target_net = Model.net(self.state_dim, self.action_dim, n_latent_layer)
        self.gamma = gamma
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimzer = torch.optim.RMSprop(self.policy_net.parameters(),lr=lr)

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

    # epison-greedy, with exponential decay for epison
    def act(self, state, step):
        epison = self.EPS_END + (self.EPS_START - self.EPS_END) \
            * math.exp(-1. * step / self.EPS_DECAY)
        sample = random.random()
        if sample >= epison:
            with torch.no_grad():
                logits = self.policy_net(state)

                return logits.max(-1)[1].view(1, 1)

        else:
           # return torch.tensor([1])
            return torch.tensor(random.randint(0,self.action_dim-1))


    def learn(self,memory, batch_size):
        batch = memory.sample(batch_size)
        if batch is None:
            return

        state_batch = torch.Tensor([s[0] for s in batch])
        action_batch = torch.tensor([s[1] for s in batch])
        reward_batch = torch.Tensor([s[2] for s in batch])
        n_state_batch = [ s[3] for s in batch ]

        non_final_mask = torch.tensor(tuple(map(lambda x: x is not None, n_state_batch)), dtype=torch.bool)
        non_final_next_state = torch.Tensor( [x for x in n_state_batch if x is not None] )

        state_action_value = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))
       # print(state_action_value.size())
        n_state_action_value = torch.zeros(batch_size)
        n_state_action_value[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()

        #print(reward_batch)
        expect_state_action_value = reward_batch + self.gamma* n_state_action_value
        #state_action_value = state_action_value.squeeze(-1)
        #expect_state_action_value = expect_state_action_value.unsqueeze(-1)
       # print(state_action_value.size())
        loss = F.smooth_l1_loss(state_action_value, expect_state_action_value.unsqueeze(-1))
      #  print(loss)
        self.optimzer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.clamp(-1,1)
        self.optimzer.step()

