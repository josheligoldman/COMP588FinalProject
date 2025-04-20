import torch
import torch.nn as nn


class FusionEncoder(nn.Module):
    """
    Encodes inorganic + organic descriptors into a shared latent
    representation for conditional generative models.
    """

    def __init__(self, zeo_dim: int, osda_dim: int, hidden_dim: int):
        super().__init__()
        self.zeo_net = nn.Sequential(
            nn.Linear(zeo_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.osda_net = nn.Sequential(
            nn.Linear(osda_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, zeolite, osda):
        return self.merge(
            torch.cat([self.zeo_net(zeolite), self.osda_net(osda)], dim=-1)
        )
