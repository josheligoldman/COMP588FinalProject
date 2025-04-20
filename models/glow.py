import numpy as np
import torch
import torch.nn as nn

from models.fusion import FusionEncoder


# ------------------------------------------------------------------ #
#  Supporting layers
# ------------------------------------------------------------------ #
class _ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.initialised = False

    def forward(self, x):
        if not self.initialised:
            with torch.no_grad():
                self.bias.data = -x.mean(0)
                self.log_scale.data = torch.log(1 / (x.std(0) + 1e-6))
                self.initialised = True
        y = (x + self.bias) * torch.exp(self.log_scale)
        ld = self.log_scale.sum()
        return y, ld

    def inverse(self, y):
        return y * torch.exp(-self.log_scale) - self.bias


class _InvertibleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        W = torch.linalg.qr(torch.randn(dim, dim), mode="reduced")[0]
        self.W = nn.Parameter(W)

    def forward(self, x):
        return x @ self.W, torch.slogdet(self.W)[1]

    def inverse(self, y):
        return y @ self.W.T


class _AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim // 2 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, (dim - dim // 2) * 2),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, cond):
        x1, x2 = x.chunk(2, dim=-1)
        s, t = self.net(torch.cat([x1, cond], -1)).chunk(2, -1)
        s = torch.tanh(s) * self.scale.exp()
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], -1), s.sum(-1)

    def inverse(self, y, cond):
        y1, y2 = y.chunk(2, dim=-1)
        s, t = self.net(torch.cat([y1, cond], -1)).chunk(2, -1)
        s = torch.tanh(s) * self.scale.exp()
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], -1)


# ------------------------------------------------------------------ #
#  Single flow step
# ------------------------------------------------------------------ #
class _FlowStep(nn.Module):
    def __init__(self, dim, cond_dim, hidden):
        super().__init__()
        self.actnorm = _ActNorm(dim)
        self.linear = _InvertibleLinear(dim)
        self.coupling = _AffineCoupling(dim, cond_dim, hidden)

    def forward(self, x, cond):
        x, ld1 = self.actnorm(x)
        x, ld2 = self.linear(x)
        x, ld3 = self.coupling(x, cond)
        return x, ld1 + ld2 + ld3

    def inverse(self, z, cond):
        z = self.coupling.inverse(z, cond)
        z = self.linear.inverse(z)
        return self.actnorm.inverse(z)


# ------------------------------------------------------------------ #
#  Glow model
# ------------------------------------------------------------------ #
class GlowModel(nn.Module):
    def __init__(self, dim, cond_dim, hidden, flows=8):
        super().__init__()
        self.flows = nn.ModuleList(
            [_FlowStep(dim, cond_dim, hidden) for _ in range(flows)]
        )

    def forward(self, x, cond):
        log_det = 0.0
        for f in self.flows:
            x, ld = f(x, cond)
            log_det += ld
        return x, log_det

    def inverse(self, z, cond):
        for f in reversed(self.flows):
            z = f.inverse(z, cond)
        return z


# ------------------------------------------------------------------ #
#  High‑level wrapper
# ------------------------------------------------------------------ #
class GlowPredictor:
    def __init__(self, cfg, zeo_dim, osda_dim, param_dim, param_ranges):
        self.cfg = cfg
        self.glow = GlowModel(param_dim, cfg["hidden_dim"], cfg["hidden_dim"])
        self.fusion_encoder = FusionEncoder(zeo_dim, osda_dim, cfg["hidden_dim"])
        self.optim = torch.optim.Adam(self.glow.parameters(), lr=cfg["learning_rate"])

        self.param_min = torch.as_tensor(param_ranges[0], dtype=torch.float32)
        self.param_max = torch.as_tensor(param_ranges[1], dtype=torch.float32)

    # .............................................................. #
    def train_step(self, batch):
        zeo, osda, x = batch["zeolite"], batch["osda"], batch["params"]
        cond = self.fusion_encoder(zeo, osda)
        z, log_det = self.glow(x, cond)
        lp = -0.5 * torch.sum(z ** 2 + np.log(2 * np.pi), dim=1)
        loss = -(lp + log_det).mean()

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss.item()

    # .............................................................. #
    @torch.no_grad()
    def sample(self, zeo, osda):
        cond = self.fusion_encoder(zeo, osda)
        z = torch.randn((zeo.size(0), 4))
        x = self.glow.inverse(z, cond)
        return torch.clamp(x, self.param_min.to(x.device), self.param_max.to(x.device))
