# ---------------------------------------------------------------------
# Main script: loads data, trains four generative models, evaluates and
# produces plots.
# ---------------------------------------------------------------------
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import config
from data import ZeoliteOSDADataset
from models import ChemDiffusion, FlowModel, GlowPredictor, CFMPredictor
from utils import (
    count_parameters,
    compute_wasserstein,
    compute_coverage,
    mean_absolute_distribution_error,
    plot_wasserstein,
    plot_coverage_radar,
    plot_mae_heatmap,
    plot_param_distributions,
)

# ------------------------------------------------------------------ #
#  1.  Load dataset
# ------------------------------------------------------------------ #
df = pd.read_excel("dataset/ZEOSYN.xlsx")
dataset = ZeoliteOSDADataset(df)
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

param_min, param_max = [0.0] * 4, [1.0] * 4  # normalised range

# ------------------------------------------------------------------ #
#  2.  Construct models
# ------------------------------------------------------------------ #
models = {
    "Diffusion": ChemDiffusion(
        config, zeo_dim=10, osda_dim=10, param_dim=4, param_ranges=(param_min, param_max)
    ),
    "RealNVP": FlowModel(
        config, zeo_dim=10, osda_dim=10, param_dim=4, param_ranges=(param_min, param_max)
    ),
    "Glow": GlowPredictor(
        config, zeo_dim=10, osda_dim=10, param_dim=4, param_ranges=(param_min, param_max)
    ),
    "CFM": CFMPredictor(
        config, zeo_dim=10, osda_dim=10, param_dim=4, param_ranges=(param_min, param_max)
    ),
}

print("\n# Model parameter counts")
for name, mdl in models.items():
    print(f"{name:9s}: {count_parameters(mdl):,}")

# ------------------------------------------------------------------ #
#  3.  Train each model
# ------------------------------------------------------------------ #
num_epochs = 5
for name, mdl in models.items():
    print(f"\n=== Training {name} ===")
    for epoch in range(1, num_epochs + 1):
        running = 0.0
        for batch in loader:
            running += mdl.train_step(batch)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | loss = {running / len(loader):.4f}")

# ------------------------------------------------------------------ #
#  4.  Sampling + metrics
# ------------------------------------------------------------------ #
all_zeo = torch.tensor(dataset.inorganic_features, dtype=torch.float32)
all_osda = torch.tensor(dataset.osda_features, dtype=torch.float32)
true_params = dataset.synthesis_params

generated = {n: m.sample(all_zeo, all_osda).cpu().numpy() for n, m in models.items()}

wass = {n: compute_wasserstein(true_params, g) for n, g in generated.items()}
cov = {n: compute_coverage(true_params, g) for n, g in generated.items()}
mae = {n: mean_absolute_distribution_error(true_params, g) for n, g in generated.items()}

os.makedirs("plots", exist_ok=True)
plot_wasserstein(wass)
plot_coverage_radar(cov)
plot_mae_heatmap(mae)
plot_param_distributions(true_params, generated)

print("\nTraining & evaluation complete. Plots saved to ./plots/")
