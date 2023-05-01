import src.settings as settings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ListenerPragmatics(nn.Module):
    def __init__(self, feature_dim, output_dim, num_imgs=1):
        super(ListenerPragmatics, self).__init__()
        self.num_imgs = num_imgs
        self.hidden_dim1 = 32
        self.hidden_dim2 = 16
        self.fc1 = nn.Linear(feature_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, output_dim)
   
    def forward(self, reconstructions, features):
        # only take the target reconstruction
        target_recons = reconstructions[:, 0:1, :]
        # concatenate with the listener representations
        concat = torch.cat((target_recons, features), dim=1)
        # reshape
        l_input = concat.view(features.shape[0], -1)
        h1 = F.relu(self.fc1(l_input))
        h2 = F.relu(self.fc2(h1))
        y = self.fc3(h2)
        return y
