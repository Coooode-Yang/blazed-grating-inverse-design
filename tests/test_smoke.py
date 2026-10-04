from pathlib import Path
import json

import numpy as np
import torch

from src.adaptive_voi4 import run_adaptive_voi4
from src.data import FixedSpectrumDataset, load_targets
from src.inference import predict_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_and_target_contract():
    dataset = FixedSpectrumDataset(ROOT / "data" / "dataset_2000.pt")
    targets, names, wavelengths = load_targets(ROOT / "data" / "target_library" / "prescribed_targets.csv")
    assert dataset.x.shape == (2000, 5)
    assert dataset.y.shape == (2000, 131)
    assert targets.shape[1] == 131
    assert len(names) == 6
    assert wavelengths.shape == (131,)
    assert float(dataset.x.min()) >= 0.0
    assert float(dataset.x.max()) <= 1.0


def test_bundled_checkpoints_and_four_candidates():
    targets, _, _ = load_targets(ROOT / "data" / "target_library" / "prescribed_targets.csv", ["flat_0p30"])
    prediction = predict_candidates(
        targets,
        15.0,
        ROOT / "checkpoints" / "forward_ensemble_2000",
        ROOT / "checkpoints" / "inverse" / "seed_101_2000" / "best_multicandidate_inverse.pt",
        torch.device("cpu"),
    )
    assert prediction["structures_norm"].shape == (1, 4, 4)
    assert prediction["forward_stack"].shape == (3, 1, 4, 131)


def test_active_training_manifests_use_2000_dataset():
    forward_config = json.loads(
        (ROOT / "checkpoints" / "forward_ensemble_2000" / "training_config.json").read_text(encoding="utf-8")
    )
    inverse_config = json.loads(
        (ROOT / "checkpoints" / "inverse" / "seed_101_2000" / "training_config.json").read_text(encoding="utf-8")
    )
    assert forward_config["dataset"]["num_samples"] == 2000
    assert forward_config["dataset"]["split_sizes"] == {"train": 1600, "val": 200, "test": 200}
    assert inverse_config["dataset"]["num_samples"] == 2000
    assert len(inverse_config["forward_ensemble"]) == 3


def test_adaptive_voi_budget_and_early_stop():
    target = np.zeros(5)
    proxy = np.zeros((4, 5))
    std = np.zeros((4, 5))
    result = run_adaptive_voi4(target, proxy, std, lambda _: np.zeros(5), max_calls=4, min_queries=1)
    assert result["num_calls"] == 1
    assert result["selected_candidate"] == 0
