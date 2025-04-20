# ---------------------------------------------------------------------
# PyTorch Dataset for zeolite + OSDA experiments.
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from utils import featurize_osda


class ZeoliteOSDADataset(Dataset):
    """Prepare inorganic + organic features and target synthesis params."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # --- inorganic (numeric columns) -----------------------------------
        inorg_cols = ["Si", "Al", "Na", "K", "Li", "Ca", "Mg", "Zn", "F", "OH"]
        self.inorganic_features = (
            df[inorg_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        )

        # --- organic (SMILES strings) --------------------------------------
        self.osda_features = self._process_osdas()

        # --- synthesis parameters (targets) --------------------------------
        synth_cols = ["Si/Al", "cryst_temp", "cryst_time", "H2O"]
        synth_numeric = df[synth_cols].apply(pd.to_numeric, errors="coerce")
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.synthesis_params = self.scaler.fit_transform(
            self.imputer.fit_transform(synth_numeric)
        )

    # ------------------------------------------------------------------ #
    #  private helpers
    # ------------------------------------------------------------------ #
    def _process_osdas(self) -> np.ndarray:
        records = []
        for _, row in self.df.iterrows():
            osdas = [
                featurize_osda(row.get("osda1 smiles", "")),
                featurize_osda(row.get("osda2 smiles", "")),
                featurize_osda(row.get("osda3 smiles", "")),
            ]
            valid = [o for o in osdas if np.any(o)]
            records.append(np.mean(valid, axis=0) if valid else np.zeros(10))
        return np.asarray(records)

    # ------------------------------------------------------------------ #
    #  PyTorch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "zeolite": torch.tensor(self.inorganic_features[idx], dtype=torch.float32),
            "osda": torch.tensor(self.osda_features[idx], dtype=torch.float32),
            "params": torch.tensor(self.synthesis_params[idx], dtype=torch.float32),
        }
