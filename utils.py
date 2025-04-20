# ---------------------------------------------------------------------
# Generic utilities shared by many modules: data cleaning, featurisation,
#   evaluation metrics, plotting helpers and a parameter counter.
# ---------------------------------------------------------------------
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from scipy.stats import wasserstein_distance
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch


# ------------------------------------------------------------------ #
#  Data‑cleaning / featurisation helpers
# ------------------------------------------------------------------ #
def clean_smiles(smiles: str) -> str:
    """Remove curly braces occasionally found in SMILES strings."""
    if pd.isna(smiles):
        return ""
    return re.sub(r"[{}]", "", str(smiles))


def featurize_osda(smiles: str) -> np.ndarray:
    """Return a 10‑d vector of RDKit descriptors (zeros on failure)."""
    try:
        mol = Chem.MolFromSmiles(clean_smiles(smiles))
        if mol is None:
            return np.zeros(10)
        return np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol),
            ]
        )
    except Exception:
        return np.zeros(10)


# ------------------------------------------------------------------ #
#  Evaluation metrics
# ------------------------------------------------------------------ #
def compute_wasserstein(true_params: np.ndarray, gen_params: np.ndarray) -> float:
    """Average 1‑Wasserstein distance over all columns."""
    return np.mean(
        [
            wasserstein_distance(true_params[:, i], gen_params[:, i])
            for i in range(true_params.shape[1])
        ]
    )


def compute_coverage(
    true_params: np.ndarray, gen_params: np.ndarray, bins: int = 20
) -> np.ndarray:
    """COV‑F1 score per column (higher == better)."""
    scores = []
    for i in range(true_params.shape[1]):
        t_hist, _ = np.histogram(
            true_params[:, i],
            bins=bins,
            range=(true_params[:, i].min(), true_params[:, i].max()),
        )
        g_hist, _ = np.histogram(
            gen_params[:, i],
            bins=bins,
            range=(true_params[:, i].min(), true_params[:, i].max()),
        )
        t_mask, g_mask = t_hist > 0, g_hist > 0
        intersection = np.logical_and(t_mask, g_mask).sum()
        precision = intersection / g_mask.sum() if g_mask.sum() else 0
        recall = intersection / t_mask.sum() if t_mask.sum() else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        scores.append(f1)
    return np.asarray(scores)


def mean_absolute_distribution_error(
    true_params: np.ndarray, gen_params: np.ndarray
) -> np.ndarray:
    return np.abs(true_params.mean(axis=0) - gen_params.mean(axis=0))


# ------------------------------------------------------------------ #
#  Plotting helpers (each saves a PNG under ``plots/``)
# ------------------------------------------------------------------ #
def plot_wasserstein(results, save_path="plots"):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.ylabel("Wasserstein distance (↓)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/wasserstein.png", dpi=300)
    plt.close()


def plot_coverage_radar(coverage_dict, save_path="plots"):
    labels = [
        "Si‑Al Ratio",
        "Cryst. Temperature",
        "Cryst. Time",
        "H₂O Content",
    ]
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    plt.figure(figsize=(6, 6))
    for model, scores in coverage_dict.items():
        data = list(scores) + [scores[0]]
        plt.polar(angles, data, label=model)
    plt.xticks(angles[:-1], labels)
    plt.ylim(0, 1)
    plt.title("COV‑F1 Score (↑)")
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/coverage.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_mae_heatmap(mae_dict, save_path="plots"):
    df = (
        pd.DataFrame(mae_dict)
        .T.rename(
            columns={
                0: "Si‑Al Ratio",
                1: "Cryst. Temperature",
                2: "Cryst. Time",
                3: "H₂O Content",
            }
        )
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="viridis", cbar_kws={"label": "MAE"})
    plt.title("Mean Absolute Error (↓)")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/mae.png", dpi=300)
    plt.close()


def plot_param_distributions(true_params, model_outputs, save_path="plots"):
    labels = [
        "Si‑Al Ratio",
        "Cryst. Temperature",
        "Cryst. Time",
        "H₂O Content",
    ]
    for model_name, gen in model_outputs.items():
        # Per‑parameter KDEs
        for i, label in enumerate(labels):
            plt.figure(figsize=(6, 4))
            sns.kdeplot(true_params[:, i], label="Experimental", fill=True, color="gray")
            sns.kdeplot(gen[:, i], label=model_name, linestyle="--", color="red")
            plt.title(f"{label}: {model_name} vs Experimental")
            plt.xlabel("Normalised value")
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_dir = os.path.join(save_path, model_name)
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(
                os.path.join(out_dir, f"{label.replace(' ', '_')}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 2×2 grid combining all parameters
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.subplot(2, 2, i + 1)
            sns.kdeplot(true_params[:, i], label="Experimental", fill=True, color="gray")
            sns.kdeplot(gen[:, i], label=model_name, linestyle="--", color="red")
            plt.title(label)
            plt.grid(True, alpha=0.3)
        plt.suptitle(f"{model_name} – parameter distributions", y=1.02)
        plt.tight_layout()
        out_dir = os.path.join(save_path, model_name)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "combined.png"), dpi=300, bbox_inches="tight")
        plt.close()


# ------------------------------------------------------------------ #
#  Misc.
# ------------------------------------------------------------------ #
def count_parameters(obj) -> int:
    """Trainable parameters in both the main model and its fusion encoder."""
    total = 0
    if hasattr(obj, "fusion_encoder"):
        total += sum(p.numel() for p in obj.fusion_encoder.parameters() if p.requires_grad)
    for attr in ("model", "flow", "glow"):
        if hasattr(obj, attr):
            total += sum(
                p.numel() for p in getattr(obj, attr).parameters() if p.requires_grad
            )
    return total
