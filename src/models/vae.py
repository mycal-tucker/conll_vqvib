import torch
import torch.nn as nn

from src.models.network_utils import reparameterize


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, hidden_dim)
        self.fc_var = nn.Linear(64, hidden_dim)
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim))

    def forward(self, x):
        hidden = self.enc(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = reparameterize(mu, log_var)
        dec = self.dec(z)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        recons_loss = torch.mean((dec - x) ** 2)
        total_loss = recons_loss + 0.0 * kl_loss
        return dec, total_loss
