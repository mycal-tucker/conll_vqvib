import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.settings as settings
from src.models.network_utils import reparameterize, gumbel_softmax


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, init_vectors=None, beta=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.trainable_quantizers = init_vectors is None
        # self.beta = beta
        #self.beta = 0.01 if settings.alpha == 0 else 0.25
        self.beta = 0.25
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))
        if init_vectors is not None:
            print("Manually specifying vector quantization for", len(init_vectors), "vectors.")
            self.prototypes.data = torch.from_numpy(init_vectors).type(torch.FloatTensor)
            self.prototypes.requires_grad = not settings.hardcoded_vq
        else:
            self.prototypes.data.uniform_(-1 / self.num_protos, 1 / self.num_protos)

    def forward(self, latents, mus=None, logvar=None):
        dists_to_protos = torch.sum(latents ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(latents, self.prototypes.t())
        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(closest_protos.size(0), self.num_protos).to(settings.device)
        encoding_one_hot.scatter_(1, closest_protos, 1)
        # encoding_one_hot = gumbel_softmax(-dists_to_protos, hard=True, temperature=0.1)
        do_print = np.random.random() < 0.00

        quantized_latents = torch.matmul(encoding_one_hot, self.prototypes)

        # Compute the VQ Losses
        if mus is None:
            commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
            embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        else:
            commitment_loss = F.mse_loss(quantized_latents.detach(), mus)
            embedding_loss = F.mse_loss(quantized_latents, mus.detach())

        # vq_loss = 1.0 * (commitment_loss * self.beta + 1.0 * embedding_loss)  # Lower VQ loss term seems useful when there's no autoencoding.
        vq_loss = 1.0 * (commitment_loss * self.beta + 1.0 * embedding_loss)  

        # Compute the entropy of the distribution for which prototypes are used. Uses a differentiable approximation
        # for the distributions.
        if logvar is not None and settings.entropy_weight != 0:
            epsilon = 0.00000001
            prior = torch.mean(encoding_one_hot + epsilon, dim=0)
            normalizer = torch.sum(prior)
            prior = prior / normalizer
            true_ent = torch.sum(-1 * prior * prior.log())

            # The correct thing is to approximate the entropy of the batch
            vector_diffs = mus.unsqueeze(1) - self.prototypes

            # scaling = torch.exp(0.5 * logvar).unsqueeze(1) + 0.00001  # Don't want to divide by zero ever.
            # normalized_diffs = torch.div(vector_diffs, scaling)
            normalized_diffs = vector_diffs / 2  # Hacky and gross. But seems to work
            square_distances = torch.square(normalized_diffs)
            normalized_dists = torch.sum(square_distances, dim=2)
            neg_dist = -0.5 * normalized_dists
            exponents = neg_dist.exp() + epsilon
            row_sums = torch.sum(exponents, dim=1, keepdim=True)
            row_probs = torch.div(exponents, row_sums)
            approx_probs = torch.mean(row_probs, dim=0)
            # if do_print:
            #     print("Vector diffs", vector_diffs)
            #     print("Scale", scaling)
            #     print("Normalized diffs", normalized_diffs)
            #     print("True prior", prior)
            #     print("Approx prob", approx_probs)
            logdist = approx_probs.log()
            ent = torch.sum(-1 * approx_probs * logdist)

            # Instead of doing the batch entropy, take entropy of each individual and then average those entropies.
            # vector_diffs = mus.unsqueeze(1) - self.prototypes
            # scaling = torch.exp(0.5 * logvar).unsqueeze(1) + 0.00001  # Don't want to divide by zero ever.
            # normalized_diffs = torch.div(vector_diffs, scaling)
            # square_distances = torch.square(normalized_diffs)
            # normalized_dists = torch.sum(square_distances, dim=2)
            # neg_dist = -0.5 * normalized_dists
            # exponents = neg_dist.exp() + epsilon
            # row_sums = torch.sum(exponents, dim=1, keepdim=True)
            # row_probs = torch.div(exponents, row_sums)
            # log_probs = row_probs.log()
            # row_ents = -1 * torch.sum(torch.mul(row_probs, log_probs), dim=1)
            # ent = torch.mean(row_ents)
        else:  # It's fine because this will never be the case in actual training.
            ent = 0

        vq_loss += settings.entropy_weight * ent
        if do_print:
            print("Ent val", ent)
            # print("Stds", stds)
            print("True ent", true_ent.item())
            # if ent < true_ent:
            #     # print("Vector diffs", vector_diffs)
            #     # print("Scale", scaling)
            #     # print("Normalized diffs", normalized_diffs)
            #     print("True prior", prior)
            #     print("Approx prob", approx_probs)
            # print("Commitment loss", commitment_loss.item())
            # print("Embedding loss", embedding_loss.item())

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss


class VQ(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos, specified_tok=None, num_simultaneous_tokens=1, variational=True, num_imgs=1):
        super(VQ, self).__init__()
        self.input_dim = input_dim
        self.comm_dim = output_dim
        self.proto_latent_dim = int(self.comm_dim / num_simultaneous_tokens)
        self.hidden_dim = 64
        if specified_tok is not None:
            num_protos = specified_tok.shape[0]
        self.num_tokens = num_protos  # Need this general variable for num tokens
        self.num_simultaneous_tokens = num_simultaneous_tokens
        self.variational = variational
        self.num_imgs = num_imgs
        self.feature_embedder = nn.Linear(input_dim, self.hidden_dim)
        in_dim = self.hidden_dim * num_imgs
        out_dim = self.hidden_dim if num_layers > 1 else self.comm_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else self.comm_dim
        self.vq_layer = VQLayer(num_protos, self.proto_latent_dim, init_vectors=specified_tok)
        self.fc_mu = nn.Linear(self.proto_latent_dim, self.proto_latent_dim)
        self.fc_var = nn.Linear(self.proto_latent_dim, self.proto_latent_dim)
        self.variational = variational
        self.eval_mode = False

    def forward(self, x):
        embedded_features = self.feature_embedder(x)
        x = torch.reshape(embedded_features, (-1, self.hidden_dim * self.num_imgs))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        if self.variational:
            # Sample in the space and then choose the closest prototype
            # This reshaping handles the desire to have multiple tokens in a single message.
            reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
            mu = self.fc_mu(reshaped)
            logvar = self.fc_var(reshaped)
            # sample = reparameterize(mu, logvar) if not self.eval_mode else mu
            sample = reparameterize(mu, logvar)
            # Quantize the vectors
            output, proto_loss = self.vq_layer(sample, mu, logvar)
            # Regroup the tokens into messages, now with possibly multiple tokens.
            output = torch.reshape(output, (-1, self.comm_dim))
            # Compute the KL divergence
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            total_loss = settings.kl_weight * kld_loss + proto_loss
            capacity = kld_loss
        else:
            reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
            x = self.fc_mu(reshaped)
            output, total_loss = self.vq_layer(x)
            output = torch.reshape(output, (-1, self.comm_dim))
            capacity = torch.tensor(0)
        return output, total_loss, capacity

    # Helper method calculates the distribution over prototypes given an input
    def get_token_dist(self, x):
        assert settings.sample_first, "There's no distribution over protos if you don't sample first."
        with torch.no_grad():

            # added
            embedded_features = self.feature_embedder(x)
            x = torch.reshape(embedded_features, (-1, self.hidden_dim * self.num_imgs))
            ###
        
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = F.relu(x)
            if self.variational:
                logvar = self.fc_var(x)
                mu = self.fc_mu(x)
                samples = reparameterize(mu, logvar)
            else:
                samples = self.fc_mu(x)
            # Now discretize
            dists_to_protos = torch.sum(samples ** 2, dim=1, keepdim=True) + \
                              torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                              torch.matmul(samples, self.vq_layer.prototypes.t())
            closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
            encoding_one_hot = torch.zeros(closest_protos.size(0), self.vq_layer.num_protos).to(settings.device)
            encoding_one_hot.scatter_(1, closest_protos, 1)
            likelihoods = np.mean(encoding_one_hot.detach().cpu().numpy(), axis=0)
        return likelihoods

    def snap_comms(self, x):
        reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
        quantized, _ = self.vq_layer(reshaped)
        return torch.reshape(quantized, (-1, self.comm_dim))

    def get_comm_id(self, comm):
        # Return the index of the closest prototype.
        # Or, if there are multiple tokens, do the clever multiplication
        vocab_size = self.num_tokens ** self.num_simultaneous_tokens
        reshaped = torch.reshape(comm, (-1, self.proto_latent_dim))
        dists_to_protos = torch.sum(reshaped ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(reshaped, self.vq_layer.prototypes.t())

        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        # idx = 0
        # for i, closest_proto in enumerate(closest_protos):
        #     idx += (self.num_tokens ** i) * closest_proto
        idx = closest_protos  # FIXME: do the math for multiple tokens
        onehot = torch.zeros((1, vocab_size))
        onehot[0, idx] = 1
        return onehot
