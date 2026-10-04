from __future__ import annotations

import json
import hashlib
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import FixedSpectrumDataset
from .models import ForwardNet, MultiCandidateTargetInverseNet


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_config() -> dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }


def _dataset_config(dataset_path: str | Path, dataset: FixedSpectrumDataset) -> dict[str, object]:
    path = Path(dataset_path).resolve()
    payload = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "num_samples": int(len(dataset.y)),
        "input_shape": list(dataset.x_physical.shape),
        "output_shape": list(dataset.y.shape),
        "wavelength_start_nm": float(dataset.wavelengths[0]),
        "wavelength_end_nm": float(dataset.wavelengths[-1]),
        "wavelength_step_nm": float(dataset.wavelengths[1] - dataset.wavelengths[0]),
        "split_sizes": {
            name: int(len(index))
            for name, index in dataset.splits.items()
            if name != "seed"
        },
        "split_seed": int(payload["splits"].get("seed", -1)),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_forward_ensemble(directory: str | Path, device: torch.device) -> list[ForwardNet]:
    directory = Path(directory)
    paths = sorted(directory.glob("*_best_forward.pt"))
    if not paths:
        paths = sorted(directory.glob("member_seed_*/best_forward.pt"))
    if not paths:
        raise FileNotFoundError(f"No forward checkpoints in {directory}")
    models: list[ForwardNet] = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device)
        model = ForwardNet(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return models


def forward_stack(models: list[ForwardNet], value: torch.Tensor) -> torch.Tensor:
    return torch.stack([model(value) for model in models])


def spectrum_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).square().mean()
    derivative = ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).square().mean()
    integral = (prediction.mean(dim=1) - target.mean(dim=1)).square().mean()
    return mse + 0.1 * derivative + 0.05 * integral


def train_forward_ensemble(
    dataset_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    epochs: int = 400,
    batch_size: int = 32,
    num_threads: int = 2,
) -> None:
    torch.set_num_threads(num_threads)
    dataset = FixedSpectrumDataset(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = _dataset_config(dataset_path, dataset)
    common_config: dict[str, object] = {
        "task": "forward_ensemble",
        "dataset": dataset_info,
        "runtime": _runtime_config(),
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_threads": int(num_threads),
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "loss": "MSE + 0.1*first_difference_MSE + 0.05*mean_efficiency_MSE",
        "model_config": {
            "input_dim": 5,
            "output_dim": int(dataset.y.shape[1]),
            "hidden_dim": 256,
            "num_blocks": 4,
            "activation": "gelu",
            "dropout": 0.0,
            "use_layer_norm": True,
            "output_activation": "sigmoid",
        },
    }
    member_results: list[dict[str, object]] = []
    for seed in seeds:
        seed_all(seed)
        model = ForwardNet(output_dim=dataset.y.shape[1], output_activation="sigmoid")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        best = math.inf
        best_state = None
        best_epoch = 0
        train_idx, val_idx = dataset.splits["train"], dataset.splits["val"]

        for epoch in range(1, epochs + 1):
            model.train()
            order = train_idx[torch.randperm(len(train_idx))]
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = spectrum_loss(model(dataset.x[idx]), dataset.y[idx])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(spectrum_loss(model(dataset.x[val_idx]), dataset.y[val_idx]))
            if val_loss < best:
                best = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("Forward training produced no checkpoint")
        member = output_dir / f"member_seed_{seed}"
        member.mkdir(parents=True, exist_ok=True)
        training_config = {**common_config, "seed": int(seed), "best_epoch": int(best_epoch)}
        torch.save(
            {
                "model_state_dict": best_state,
                "model_config": common_config["model_config"],
                "seed": seed,
                "best_val_loss": best,
                "best_epoch": best_epoch,
                "training_config": training_config,
            },
            member / "best_forward.pt",
        )
        member_results.append(
            {
                "seed": int(seed),
                "checkpoint": str((member / "best_forward.pt").resolve()),
                "best_val_loss": float(best),
            }
        )
    common_config["members"] = member_results
    _write_json(output_dir / "training_config.json", common_config)


def _candidate_score(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (mean - target[:, None, :]).square().mean(dim=-1)
    derivative = (
        (mean[..., 1:] - mean[..., :-1]) - (target[:, None, 1:] - target[:, None, :-1])
    ).square().mean(dim=-1)
    return mse + 0.1 * derivative


def _candidate_predictions(models, structures, incidence_norm):
    batch, candidates, _ = structures.shape
    flat = structures.reshape(batch * candidates, 4)
    incidence = incidence_norm[:, None].expand(batch, candidates).reshape(-1)
    inputs = torch.cat([flat, incidence[:, None]], dim=1)
    return forward_stack(models, inputs).reshape(len(models), batch, candidates, -1)


def train_inverse(
    dataset_path: str | Path,
    forward_dir: str | Path,
    output_dir: str | Path,
    seed: int = 101,
    epochs: int = 400,
    batch_size: int = 64,
    num_threads: int = 2,
) -> None:
    torch.set_num_threads(num_threads)
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FixedSpectrumDataset(dataset_path)
    models = load_forward_ensemble(forward_dir, device)
    models = [model.to(device) for model in models]
    forward_dir = Path(forward_dir).resolve()
    forward_paths = sorted(forward_dir.glob("*_best_forward.pt"))
    if not forward_paths:
        forward_paths = sorted(forward_dir.glob("member_seed_*/best_forward.pt"))
    forward_checkpoint_info = [
        {"path": str(path.resolve()), "sha256": _sha256(path)} for path in forward_paths
    ]
    model = MultiCandidateTargetInverseNet(spectrum_dim=131, num_candidates=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best, best_state, best_epoch = math.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_idx = dataset.splits["train"]
        order = train_idx[torch.randperm(len(train_idx))]
        train_total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            target = dataset.y[idx].to(device)
            incidence = dataset.x[idx, 4].to(device)
            optimizer.zero_grad(set_to_none=True)
            structures = model(target, incidence)
            stack = _candidate_predictions(models, structures, incidence)
            mean = stack.mean(0)
            score = _candidate_score(mean, target)
            diversity = torch.abs(structures[:, :, None] - structures[:, None, :]).mean(-1)
            mask = ~torch.eye(4, dtype=torch.bool, device=device)

            loss = score.min(dim=1).values.mean() + 0.1 * score.mean() - 0.002 * diversity[:, mask].mean()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            idx = dataset.splits["val"]
            target = dataset.y[idx].to(device)
            incidence = dataset.x[idx, 4].to(device)
            structures = model(target, incidence)
            val_score = _candidate_score(_candidate_predictions(models, structures, incidence).mean(0), target)
            val_loss = float(val_score.min(dim=1).values.mean())
        history.append({"epoch": epoch, "train_loss": train_total / len(train_idx), "val_best_response": val_loss})
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Inverse training produced no checkpoint")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "spectrum_dim": 131,
        "hidden_dim": 256,
        "num_blocks": 4,
        "dropout": 0.0,
        "num_candidates": 4,
    }
    training_config: dict[str, object] = {
        "task": "multi_candidate_inverse",
        "dataset": _dataset_config(dataset_path, dataset),
        "forward_ensemble": forward_checkpoint_info,
        "runtime": _runtime_config(),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_threads": int(num_threads),
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "loss": "best_candidate + 0.1*candidate_average - 0.002*candidate_diversity",
        "model_config": model_config,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best),
    }
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": model_config,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "normalization": "all five input parameters normalized to [0,1]; inverse receives normalized incidence",
        "training_config": training_config,
    }
    torch.save(checkpoint, output_dir / "best_multicandidate_inverse.pt")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    _write_json(output_dir / "summary.json", training_config)
    _write_json(output_dir / "training_config.json", training_config)
