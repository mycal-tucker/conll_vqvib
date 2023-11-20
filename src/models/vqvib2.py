import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import src.settings as settings
from src.models.network_utils import gumbel_softmax


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, beta=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.beta = beta
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))

        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        # self.prototypes.data.uniform_(-1, 1)
        self.prototypes.data.uniform_(-1 / self.num_protos, 1 / self.num_protos)

    def forward(self, latents):
        vector_diffs = latents.unsqueeze(1) - self.prototypes
        normalized_dists = torch.sum(vector_diffs ** 2, dim=2)
        neg_dists = -1 * normalized_dists
        onehot, logprobs = gumbel_softmax(neg_dists, hard=True, return_dist=True)
        quantized_latents = torch.matmul(onehot, self.prototypes)

        # Compute capacity as the KL divergence from the prior.
        epsilon = 0.000001  # Need a fuzz factor for numerical stability.
        true_prior = torch.mean(logprobs.exp() + epsilon, dim=0, keepdim=True)
        true_prior = true_prior.expand(logprobs.shape[0], -1)
        # prior = true_prior if not settings.fixed_prior else torch.ones_like(logprobs) / logprobs.shape[1] # Estimate the prior based on the batch.
        # prior = true_prior
        prior = torch.ones_like(logprobs) / logprobs.shape[1]
        capacity = self.kl_loss_fn(logprobs, prior)
        # total_loss = settings.kl_weight * capacity

        # Penalize the entropy of the prior, just to reduce codebook size
        ent = torch.sum(-1 * true_prior[0] * true_prior[0].log())  # Just the first row, because it's repeated for batch size.
        # total_loss += settings.entropy_weight * ent

        total_loss = settings.kl_weight * (capacity + settings.entropy_weight * ent)

        # Clustering losses
        # embedding_loss = F.mse_loss(quantized_latents, latents.detach())  # Move prototypes to be near embeddings
        # commitment_loss = F.mse_loss(quantized_latents.detach(), latents)  # Move embeddings to be near prototypes
        # total_loss += 0.1 * (embedding_loss + 0.25 * commitment_loss)  # Worsens performance by a ton!

        if np.random.random() < 0.00:
            print("ent", ent.item())
            print("Cap", capacity.item())
            # print("Embedding", embedding_loss.item())
            # print("Commitment", commitment_loss.item())

        return quantized_latents, total_loss, capacity

"""
This version of VQ-VIB doesn't sample from a Gaussian and then discretize, but rather computes the l2 distance to
each quantized vector and samples from the categorical distribution of prototypes with

P(vq_i|z) \propto exp(-1 * (z -  vq_i) ** 2)

This architecture seems much more stable, faster to converge, etc., the marginal entropy can be exactly computed
instead of our hacky approximation from the other method, and we don't need clustering/embedding losses either.
"""
class VQVIB2(nn.Module):
    def __init__(self, input_dim, output_dim,  num_layers, num_protos, num_simultaneous_tokens=1, num_images=1):
        super(VQVIB2, self).__init__()
        self.output_dim = output_dim
        self.comm_dim = output_dim
        self.proto_latent_dim = int(self.output_dim / num_simultaneous_tokens)
        self.hidden_dim = 64
        self.num_tokens = num_protos  # Need this general variable for num tokens
        self.num_simultaneous_tokens = num_simultaneous_tokens
        self.num_images = num_images
        
        input_dim = input_dim * self.num_images 
        self.feature_embedder = nn.Linear(input_dim, self.hidden_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

        self.vq_layer = VQLayer(num_protos, self.proto_latent_dim)

    def forward(self, x):

        if self.num_images > 1:
            x = torch.cat([x[:, i, :] for i in range(self.num_images)], dim=1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
        output, total_loss, capacity = self.vq_layer(reshaped)

        reshaped_comm = torch.reshape(output, (-1, self.comm_dim))
        return reshaped_comm, total_loss, capacity

    def get_token_dist(self, x):
        with torch.no_grad():
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            # Measure dist to each prototype, like normal stuff.
            vector_diffs = x.unsqueeze(1) - self.vq_layer.prototypes
            normalized_dists = torch.sum(vector_diffs ** 2, dim=2)
            neg_dists = -1 * normalized_dists
            _, logprobs = gumbel_softmax(neg_dists, hard=True, return_dist=True)
        likelihoods = np.mean(logprobs.exp().cpu().numpy(), axis=0)
        return likelihoods
