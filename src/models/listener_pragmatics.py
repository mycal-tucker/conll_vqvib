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



class ListenerPragmaticsCosines(nn.Module):
    def __init__(self, feature_dim):
        super(ListenerPragmaticsCosines, self).__init__()
        self.feature_embed_dim = 16
        self.comm_embedder = nn.Linear(feature_dim, self.feature_embed_dim)
        self.feature_embedder = nn.Linear(feature_dim, self.feature_embed_dim)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, reconstructions, features):
        target_recons = reconstructions[:, 0:1, :]
        embedded_comm = self.comm_embedder(target_recons)
        num_imgs = features.shape[1]
        embedded_comm = embedded_comm.repeat(1, num_imgs, 1)
        embedded_features = self.feature_embedder(features)
        # Get cosine similarities
        cosines = self.cos(embedded_comm, embedded_features)
        logits = cosines
        return logits
