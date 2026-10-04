from __future__ import annotations

from pathlib import Path

import torch

from .models import MultiCandidateTargetInverseNet, denormalize_structure, normalize_incidence
from .training import forward_stack, load_forward_ensemble


def load_inverse(path: str | Path, device: torch.device) -> MultiCandidateTargetInverseNet:
    checkpoint = torch.load(path, map_location=device)
    model = MultiCandidateTargetInverseNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_candidates(
    target: torch.Tensor,
    incidence_deg: float,
    forward_dir: str | Path,
    inverse_checkpoint: str | Path,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = target.to(device)
    incidence = normalize_incidence(incidence_deg).to(device).reshape(1)
    
    inverse = load_inverse(inverse_checkpoint, device)
    forward_models = load_forward_ensemble(forward_dir, device)
    structures_norm = inverse(target, incidence.expand(len(target)))
    batch, candidates, _ = structures_norm.shape
    inputs = torch.cat(
        [structures_norm.reshape(batch * candidates, 4), incidence.expand(batch * candidates)[:, None]], dim=1
    )
    stack = forward_stack(forward_models, inputs).reshape(len(forward_models), batch, candidates, -1)
    return {
        "structures_norm": structures_norm.cpu(),
        "structures_physical": denormalize_structure(structures_norm).cpu(),
        "forward_stack": stack.cpu(),
        "forward_mean": stack.mean(0).cpu(),
        "forward_std": stack.std(0, unbiased=False).cpu(),
        "incidence_norm": incidence.cpu(),
    }

