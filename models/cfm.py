import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import FusionEncoder


class _FlowMatching(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x, t, cond):
        return self.net(torch.cat([x, cond, t], dim=-1))


class CFMPredictor:
    def __init__(self, cfg, zeo_dim, osda_dim, param_dim, param_ranges):
        self.cfg = cfg
        self.model = _FlowMatching(param_dim, cfg["hidden_dim"], cfg["hidden_dim"])
        self.fusion_encoder = FusionEncoder(zeo_dim, osda_dim, cfg["hidden_dim"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])

        self.param_min = torch.as_tensor(param_ranges[0], dtype=torch.float32)
        self.param_max = torch.as_tensor(param_ranges[1], dtype=torch.float32)

    # .............................................................. #
    def train_step(self, batch):
        zeo, osda, x1 = batch["zeolite"], batch["osda"], batch["params"]
        t = torch.rand(x1.size(0), 1)
        x0 = torch.randn_like(x1)
        xt = (1 - t) * x0 + t * x1

        cond = self.fusion_encoder(zeo, osda)
        pred_v = self.model(xt, t, cond)
        target_v = x1 - x0
        loss = F.mse_loss(pred_v, target_v)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss.item()

    # .............................................................. #
    @torch.no_grad()
    def sample(self, zeo, osda, steps=50):
        cond = self.fusion_encoder(zeo, osda)
        x = torch.randn((zeo.size(0), 4))
        for s in range(steps):
            t = torch.full((x.size(0), 1), s / steps)
            v = self.model(x, t, cond)
            x = x + v / steps
        return torch.clamp(x, self.param_min.to(x.device), self.param_max.to(x.device))
