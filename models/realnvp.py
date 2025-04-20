import numpy as np
import torch
import torch.nn as nn

from models.fusion import FusionEncoder


# ------------------------------------------------------------------ #
#  Coupling layer
# ------------------------------------------------------------------ #
class _Coupling(nn.Module):
    def __init__(self, dim, cond_dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.m = mask
        self.s_t = nn.Sequential(
            nn.Linear(mask.sum().item() + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, (dim - mask.sum().item()) * 2),
        )

    def forward(self, x, cond):
        xa, xb = x[:, self.m], x[:, ~self.m]
        s, t = self.s_t(torch.cat([xa, cond], -1)).chunk(2, -1)
        s = torch.tanh(s)
        yb = xb * torch.exp(s) + t
        log_det = s.sum(-1)
        y = torch.zeros_like(x)
        y[:, self.m], y[:, ~self.m] = xa, yb
        return y, log_det

    def inverse(self, y, cond):
        ya, yb = y[:, self.m], y[:, ~self.m]
        s, t = self.s_t(torch.cat([ya, cond], -1)).chunk(2, -1)
        s = torch.tanh(s)
        xb = (yb - t) * torch.exp(-s)
        x = torch.zeros_like(y)
        x[:, self.m], x[:, ~self.m] = ya, xb
        return x


# ------------------------------------------------------------------ #
#  RealNVP architecture
# ------------------------------------------------------------------ #
class RealNVP(nn.Module):
    def __init__(self, dim, cond_dim, hidden, layers=6):
        super().__init__()
        mask = torch.zeros(dim, dtype=torch.bool)
        mask[::2] = True
        blocks = []
        for _ in range(layers):
            blocks.append(_Coupling(dim, cond_dim, hidden, mask.clone()))
            mask = ~mask
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        log_det = 0.0
        for c in self.blocks:
            x, ld = c(x, cond)
            log_det += ld
        return x, log_det

    def inverse(self, z, cond):
        for c in reversed(self.blocks):
            z = c.inverse(z, cond)
        return z


# ------------------------------------------------------------------ #
#  High‑level wrapper
# ------------------------------------------------------------------ #
class FlowModel:
    def __init__(self, cfg, zeo_dim, osda_dim, param_dim, param_ranges):
        self.cfg = cfg
        self.flow = RealNVP(param_dim, cfg["hidden_dim"], cfg["hidden_dim"])
        self.fusion_encoder = FusionEncoder(zeo_dim, osda_dim, cfg["hidden_dim"])
        self.optim = torch.optim.Adam(self.flow.parameters(), lr=cfg["learning_rate"])

        self.param_min = torch.as_tensor(param_ranges[0], dtype=torch.float32)
        self.param_max = torch.as_tensor(param_ranges[1], dtype=torch.float32)

    # .............................................................. #
    def train_step(self, batch):
        zeo, osda, x = batch["zeolite"], batch["osda"], batch["params"]
        cond = self.fusion_encoder(zeo, osda)
        z, log_det = self.flow(x, cond)
        prior_lp = -0.5 * torch.sum(z ** 2 + np.log(2 * np.pi), dim=1)
        loss = -(prior_lp + log_det).mean()

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss.item()

    # .............................................................. #
    @torch.no_grad()
    def sample(self, zeo, osda):
        cond = self.fusion_encoder(zeo, osda)
        z = torch.randn((zeo.size(0), 4))
        x = self.flow.inverse(z, cond)
        return torch.clamp(x, self.param_min.to(x.device), self.param_max.to(x.device))
