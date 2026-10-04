from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_voi4 import run_adaptive_voi4
from src.data import FixedSpectrumDataset, load_targets
from src.inference import predict_candidates
from src.models import WAVELENGTHS_NM
from src.rcwa import run_matlab_batch
from src.training import train_forward_ensemble, train_inverse


DEFAULTS = {
    "dataset": ROOT / "data" / "dataset_2000.pt",
    "targets": ROOT / "data" / "target_library" / "prescribed_targets.csv",
    "forward": ROOT / "checkpoints" / "forward_ensemble_2000",
    "inverse": ROOT / "checkpoints" / "inverse" / "seed_101_2000" / "best_multicandidate_inverse.pt",
    "material": ROOT / "data" / "materials" / "Al_Rakic_BB_400_1700nm.csv",
    "matlab_dir": ROOT / "matlab",
    "matlab_executable": Path(r"E:\MATLAB2021\bin\matlab.exe"),
}


def _target_names(value: str) -> list[str]:
    return [] if value.lower() in {"all", "*"} else [value]


def command_train_forward(args):
    train_forward_ensemble(args.dataset, args.output_dir, args.seeds, args.epochs, args.batch_size, args.threads)


def command_train_inverse(args):
    train_inverse(args.dataset, args.forward_dir, args.output_dir, args.seed, args.epochs, args.batch_size, args.threads)


def _save_prediction_files(output_dir: Path, names, prediction, target, incidence_deg):
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = prediction["structures_physical"].numpy()
    norm = prediction["structures_norm"].numpy()
    mean = prediction["forward_mean"].numpy()
    std = prediction["forward_std"].numpy()
    rows = []
    for target_index, name in enumerate(names):
        for candidate_index in range(4):
            row = {
                "target": name,
                "incidence_deg": incidence_deg,
                "candidate_index": candidate_index,
                "LineDensity_per_mm": structures[target_index, candidate_index, 0],
                "w1": structures[target_index, candidate_index, 1],
                "Theta1_deg": structures[target_index, candidate_index, 2],
                "Theta2_deg": structures[target_index, candidate_index, 3],
            }
            row.update({f"structure_norm_{j}": norm[target_index, candidate_index, j] for j in range(4)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "candidate_structures.csv", index=False)
    spectra_rows = []
    for target_index, name in enumerate(names):
        for candidate_index in range(4):
            for wavelength_index, wavelength in enumerate(WAVELENGTHS_NM.tolist()):
                spectra_rows.append(
                    {
                        "target": name,
                        "incidence_deg": incidence_deg,
                        "candidate_index": candidate_index,
                        "wavelength_nm": wavelength,
                        "target_efficiency": target[target_index, wavelength_index].item(),
                        "forward_mean": mean[target_index, candidate_index, wavelength_index],
                        "forward_std": std[target_index, candidate_index, wavelength_index],
                    }
                )
    pd.DataFrame(spectra_rows).to_csv(output_dir / "forward_ensemble_predictions.csv", index=False)


def command_run(args):
    names = _target_names(args.target_name)
    target_matrix, names, _ = load_targets(args.targets, names or None)
    prediction = predict_candidates(target_matrix, args.incidence_deg, args.forward_dir, args.inverse_checkpoint)
    output_dir = Path(args.output_dir).resolve()
    _save_prediction_files(output_dir, names, prediction, target_matrix, args.incidence_deg)

    proxy_mean = prediction["forward_mean"].numpy()
    proxy_std = prediction["forward_std"].numpy()
    all_traces = []
    final_rows = []
    for target_index, name in enumerate(names):
        target = target_matrix[target_index].numpy()
        candidates = prediction["structures_norm"][target_index]
        query_root = output_dir / "rcwa_queries" / name
        if args.rcwa_mode == "matlab":
            if not Path(args.matlab_executable).exists():
                raise FileNotFoundError(f"MATLAB executable not found: {args.matlab_executable}")

            def query(candidate_index: int):
                result = run_matlab_batch(
                    candidates[candidate_index : candidate_index + 1],
                    args.incidence_deg,
                    query_root / f"query_{candidate_index}",
                    args.matlab_dir,
                    args.material_csv,
                    args.matlab_executable,
                    args.low_order,
                    args.low_layers,
                    args.workers,
                )
                return result[0]

        else:
            def query(candidate_index: int):
                return proxy_mean[target_index, candidate_index]

        decision = run_adaptive_voi4(
            target,
            proxy_mean[target_index],
            proxy_std[target_index],
            query,
            member_spectra=prediction["forward_stack"][:, target_index].numpy(),
            max_calls=4,
            min_queries=1,
            target_tolerance=args.stop_rmse,
        )
        for item in decision["trace"]:
            all_traces.append({"target": name, **item})
        winner = decision["selected_candidate"]
        final_spectrum = None
        if args.rcwa_mode == "matlab":
            final_spectrum = run_matlab_batch(
                candidates[winner : winner + 1],
                args.incidence_deg,
                output_dir / "final_highres" / name,
                args.matlab_dir,
                args.material_csv,
                args.matlab_executable,
                args.high_order,
                args.high_layers,
                args.workers,
            )[0]
        else:
            final_spectrum = proxy_mean[target_index, winner]
        final_rows.append(
            {
                "target": name,
                "incidence_deg": args.incidence_deg,
                "selected_candidate": winner,
                "num_rcwa_calls": decision["num_calls"],
                "stop_reason": decision["stop_reason"],
                "proxy_rmse": float(np.sqrt(np.mean((proxy_mean[target_index, winner] - target) ** 2))),
                "final_rmse": float(np.sqrt(np.mean((final_spectrum - target) ** 2))),
                "rcwa_mode": args.rcwa_mode,
            }
        )
    pd.DataFrame(all_traces).to_csv(output_dir / "adaptive_voi4_trace.csv", index=False)
    pd.DataFrame(final_rows).to_csv(output_dir / "final_selection.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "workflow": "four-candidate TNN -> three-seed forward ensemble -> Adaptive-VOI-4 -> final RCWA",
                "normalization": "physical inputs and incidence normalized by src.models; target spectra remain efficiency values",
                "dataset_path": str(Path(DEFAULTS["dataset"]).resolve()),
                "forward_checkpoint_dir": str(Path(args.forward_dir).resolve()),
                "inverse_checkpoint": str(Path(args.inverse_checkpoint).resolve()),
                "rcwa_mode": args.rcwa_mode,
                "low_resolution": {"order": args.low_order, "layers": args.low_layers},
                "high_resolution": {"order": args.high_order, "layers": args.high_layers},
                "targets": names,
                "incidence_deg": args.incidence_deg,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(final_rows).to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Complete grating inverse-design workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    forward = sub.add_parser("train-forward")
    forward.add_argument("--dataset", default=str(DEFAULTS["dataset"]))
    forward.add_argument("--output-dir", default=str(DEFAULTS["forward"]))
    forward.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    forward.add_argument("--epochs", type=int, default=400)
    forward.add_argument("--batch-size", type=int, default=32)
    forward.add_argument("--threads", type=int, default=2)
    forward.set_defaults(func=command_train_forward)

    inverse = sub.add_parser("train-inverse")
    inverse.add_argument("--dataset", default=str(DEFAULTS["dataset"]))
    inverse.add_argument("--forward-dir", default=str(DEFAULTS["forward"]))
    inverse.add_argument("--output-dir", default=str(Path(DEFAULTS["inverse"]).parent))
    inverse.add_argument("--seed", type=int, default=101)
    inverse.add_argument("--epochs", type=int, default=400)
    inverse.add_argument("--batch-size", type=int, default=64)
    inverse.add_argument("--threads", type=int, default=2)
    inverse.set_defaults(func=command_train_inverse)

    run = sub.add_parser("run", aliases=["all"])
    run.add_argument("--targets", default=str(DEFAULTS["targets"]))
    run.add_argument("--target-name", default="flat_0p30", help="target name, or all")
    run.add_argument("--incidence-deg", type=float, default=15.0)
    run.add_argument("--forward-dir", default=str(DEFAULTS["forward"]))
    run.add_argument("--inverse-checkpoint", default=str(DEFAULTS["inverse"]))
    run.add_argument("--output-dir", default=str(ROOT / "outputs" / "complete_run"))
    run.add_argument("--rcwa-mode", choices=["dry-run", "matlab"], default="dry-run")
    run.add_argument("--matlab-executable", default=str(DEFAULTS["matlab_executable"]))
    run.add_argument("--matlab-dir", default=str(DEFAULTS["matlab_dir"]))
    run.add_argument("--material-csv", default=str(DEFAULTS["material"]))
    run.add_argument("--low-order", type=int, default=15)
    run.add_argument("--low-layers", type=int, default=50)
    run.add_argument("--high-order", type=int, default=20)
    run.add_argument("--high-layers", type=int, default=80)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--stop-rmse", type=float, default=0.03)
    run.set_defaults(func=command_run)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
