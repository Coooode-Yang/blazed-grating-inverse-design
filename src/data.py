from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .models import WAVELENGTHS_NM, normalize_physical


class FixedSpectrumDataset:
    """Load the bundled TE/Al dataset with the workflow's single normalization rule."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu")
        
        self.x_physical = torch.as_tensor(payload.get("x", payload.get("inputs")), dtype=torch.float32)
        self.y = torch.as_tensor(payload.get("y", payload.get("spectra")), dtype=torch.float32)
        self.wavelengths = torch.as_tensor(
            payload.get("wavelength_nm", payload.get("wavelengths", WAVELENGTHS_NM)), dtype=torch.float32
        )

        self.x = normalize_physical(self.x_physical)
        self.splits = {name: torch.as_tensor(value, dtype=torch.long) for name, value in payload["splits"].items()}
        if self.x.shape != (len(self.y), 5) or self.y.shape[1] != 131:
            raise ValueError(f"Expected x=[N,5], y=[N,131], got {tuple(self.x.shape)}, {tuple(self.y.shape)}")
        if not torch.allclose(self.wavelengths, WAVELENGTHS_NM):
            raise ValueError("The workflow requires the 400:10:1700 nm wavelength grid")
        merged = torch.cat([self.splits["train"], self.splits["val"], self.splits["test"]])
        if len(torch.unique(merged)) != len(self.y):
            raise ValueError("Dataset splits must be disjoint and cover all samples")


def load_targets(path: str | Path, names: list[str] | None = None) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    frame = pd.read_csv(path)
    if "wavelength_nm" not in frame:
        raise ValueError("Target CSV must contain wavelength_nm")
    wavelengths = torch.tensor(frame.pop("wavelength_nm").to_numpy(), dtype=torch.float32)
    if not torch.allclose(wavelengths, WAVELENGTHS_NM):
        raise ValueError("Target wavelengths do not match the fixed 400:10:1700 nm grid")
    selected = names or list(frame.columns)
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise KeyError(f"Unknown target names: {missing}")
    return torch.tensor(frame[selected].to_numpy().T, dtype=torch.float32), selected, wavelengths
