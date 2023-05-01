import src.settings as settings

import torch.nn as nn
import torch.nn.functional as F


class Listener(nn.Module):
    def __init__(self, feature_dim):
        super(Listener, self).__init__()
        self.feature_embed_dim = 16
        self.comm_embedder = nn.Linear(feature_dim, self.feature_embed_dim)
        self.feature_embedder = nn.Linear(feature_dim, self.feature_embed_dim)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, comm_embedding, features):
        embedded_comm = self.comm_embedder(comm_embedding)
        num_imgs = features.shape[1]
        embedded_comm = embedded_comm.repeat(1, num_imgs, 1)
        embedded_features = self.feature_embedder(features)
        # Get cosine similarities
        cosines = self.cos(embedded_comm, embedded_features)
        #logits = F.log_softmax(cosines, dim=1)
        logits = cosines
        return logits
