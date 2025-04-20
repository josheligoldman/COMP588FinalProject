import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import FusionEncoder


# ------------------------------------------------------------------ #
#  UNet‑like backbone
# ------------------------------------------------------------------ #
class DiffusionUNet(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        self.down1 = nn.Sequential(nn.Linear(in_dim + cond_dim + hidden, hidden), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.ReLU())

        self.up1 = nn.Sequential(nn.Linear(hidden // 4 + hidden // 2, hidden // 2), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(hidden // 2 + hidden, hidden), nn.ReLU())
        self.final = nn.Linear(hidden, in_dim)

    def forward(self, x, t, cond):
        t_emb = self.t_embed(t.unsqueeze(-1))
        h = torch.cat([x, cond, t_emb], dim=-1)
        d1 = self.down1(h)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(torch.cat([d3, d2], dim=-1))
        u2 = self.up2(torch.cat([u1, d1], dim=-1))
        return self.final(u2)


# ------------------------------------------------------------------ #
#  Exponential Moving Average helper (internal class)
# ------------------------------------------------------------------ #
class _EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model, self.decay = model, decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[n].clone()


# ------------------------------------------------------------------ #
#  High‑level wrapper used by the training script
# ------------------------------------------------------------------ #
class ChemDiffusion:
    def __init__(self, cfg, zeo_dim, osda_dim, param_dim, param_ranges):
        self.cfg = cfg
        self.model = DiffusionUNet(param_dim, cfg["hidden_dim"], cfg["hidden_dim"])
        self.fusion_encoder = FusionEncoder(zeo_dim, osda_dim, cfg["hidden_dim"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])

        self.param_min = torch.as_tensor(param_ranges[0], dtype=torch.float32)
        self.param_max = torch.as_tensor(param_ranges[1], dtype=torch.float32)

        self.ema = _EMA(self.model, cfg["ema_decay"])
        self.betas = self._cosine_beta_schedule(cfg["timesteps"])
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    # .............................................................. #
    def _cosine_beta_schedule(self, T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
        ac /= ac[0].clone()
        betas = 1 - ac[1:] / ac[:-1]
        return torch.clip(betas, 1e-4, 0.9999)

    # .............................................................. #
    def train_step(self, batch):
        zeo, osda, x0 = batch["zeolite"], batch["osda"], batch["params"]
        t = torch.randint(0, self.cfg["timesteps"], (x0.size(0),))
        cond = self.fusion_encoder(zeo, osda)
        if torch.rand(1) < self.cfg["uncond_prob"]:
            cond = torch.zeros_like(cond)  # classifier‑free guidance

        noise = torch.randn_like(x0)
        alpha_t = self.alpha_cumprod[t].unsqueeze(-1)
        xt = (alpha_t.sqrt()) * x0 + (1 - alpha_t).sqrt() * noise

        pred = self.model(xt, t / self.cfg["timesteps"], cond)
        loss = F.mse_loss(pred, noise)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.ema.update()
        return loss.item()

    # .............................................................. #
    @torch.no_grad()
    def sample(self, zeo, osda, num_steps=1000):
        self.ema.apply_shadow()
        cond = self.fusion_encoder(zeo, osda)
        x = torch.randn((zeo.size(0), 4))
        for i in reversed(range(num_steps)):
            t = torch.full((x.size(0),), i, dtype=torch.float32) / self.cfg["timesteps"]
            pred = self.model(x, t, cond)
            alpha_t, alpha_bar = self.alphas[i], self.alpha_cumprod[i]
            x = (x - (1 - alpha_t).sqrt() * pred) / alpha_t.sqrt()
            if i:
                x += (1 - alpha_bar).sqrt() * torch.randn_like(x)
        return torch.clamp(x, self.param_min.to(x.device), self.param_max.to(x.device))
