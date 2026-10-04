from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import PARAM_NAMES, denormalize_structure


def candidate_frame(structures_norm: torch.Tensor, incidence_deg: float) -> pd.DataFrame:
    physical = denormalize_structure(structures_norm).detach().cpu().numpy()
    rows = []
    for index, row in enumerate(physical):
        line_density, w1, theta1, theta2 = row.tolist()
        if theta1 > theta2:
            theta1, theta2 = theta2, theta1
            w1 = 1.0 - w1
        rows.append(
            {
                "candidate_index": index,
                "LineDensity_per_mm": line_density,
                "w1": w1,
                "Theta1_deg": theta1,
                "Theta2_deg": theta2,
                "Inc_Angle_deg": incidence_deg,
            }
        )
    return pd.DataFrame(rows)


def run_matlab_batch(
    structures_norm: torch.Tensor,
    incidence_deg: float,
    output_dir: str | Path,
    matlab_dir: str | Path,
    material_csv: str | Path,
    matlab_executable: str | Path,
    order: int,
    n_layers: int,
    workers: int = 1,
) -> np.ndarray:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_csv = output_dir / "rcwa_input.csv"
    frame = candidate_frame(structures_norm, incidence_deg)
    frame.to_csv(input_csv, index=False)

    command = (
        f"addpath('{Path(matlab_dir).resolve().as_posix()}'); "
        f"generate_te_al_dataset('{input_csv.resolve().as_posix()}','{output_dir.resolve().as_posix()}',"
        f"{order},{n_layers},{workers},1,'{Path(material_csv).resolve().as_posix()}');"
    )
    completed = subprocess.run(
        [str(matlab_executable), "-batch", command],
        check=False,
        capture_output=True,
        text=True,
    )
    (output_dir / "matlab_stdout.txt").write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    (output_dir / "matlab_stderr.txt").write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB RCWA failed with exit code {completed.returncode}. "
            f"See {output_dir / 'matlab_stderr.txt'}"
        )
    spectrum_path = output_dir / "te_al_spectra.csv"
    if not spectrum_path.exists():
        raise FileNotFoundError(f"MATLAB did not create {spectrum_path}")
    return pd.read_csv(spectrum_path, header=None).to_numpy(dtype=float)

