import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class net(nn.Module):
    def __init__(self,state_dim,action_dim, n_latent_var):
        super(net, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var,action_dim)
        )

    def forward(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float().to(device)
        #print(state)
        # print(state)
        logits = self.ff(state)
        return logits
