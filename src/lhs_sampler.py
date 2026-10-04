"""用于双分区光栅的拉丁超立方体抽样。

五个采样坐标为

    [line density, w1, blaze angle 1, blaze angle 2, incidence angle]

前四个坐标描述光栅结构；入射角作为正向模型的工作条件进行抽样。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc

from .models import PARAM_HIGH, PARAM_LOW, PARAM_NAMES


DEFAULT_NUM_SAMPLES = 2000
DEFAULT_LHS_SEED = 42


def generate_lhs_samples(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_LHS_SEED,
) -> np.ndarray:
    """生成物理五维拉丁超立方体样本。

    每个坐标由 :class:`scipy.stats.qmc.LatinHypercube` 分层为 ``num_samples`` 个区间，
    然后缩放到 ``src.models`` 使用的物理边界。舍入规则与原始项目相匹配：
    线密度为整数，而 ``w1`` 和两个光栅角度保留到小数点后两位。入射角保持连续。
    """

    if int(num_samples) <= 0:
        raise ValueError("num_samples 必须是正整数")
    num_samples = int(num_samples)
    seed = int(seed)

    bounds_low = PARAM_LOW.detach().cpu().numpy().astype(np.float64)
    bounds_high = PARAM_HIGH.detach().cpu().numpy().astype(np.float64)

    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    sample_unit = sampler.random(n=num_samples)
    samples = qmc.scale(sample_unit, bounds_low, bounds_high)

    line_density_idx = PARAM_NAMES.index("LineDensity_per_mm")
    rounded_two_decimal_indices = [
        PARAM_NAMES.index("w1"),
        PARAM_NAMES.index("Theta1_deg"),
        PARAM_NAMES.index("Theta2_deg"),
    ]
    samples[:, line_density_idx] = np.round(samples[:, line_density_idx])
    samples[:, rounded_two_decimal_indices] = np.round(
        samples[:, rounded_two_decimal_indices], decimals=2
    )

    return samples


def save_lhs_samples(
    samples: np.ndarray,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_LHS_SEED,
    csv_name: str | None = None,
    pt_name: str | None = None,
) -> tuple[Path, Path]:
    """将采样参数保存为 CSV 和 PyTorch 张量包。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = int(len(samples))
    csv_path = output_dir / (csv_name or f"lhs_inputs_{n}.csv")
    pt_path = output_dir / (pt_name or f"lhs_inputs_{n}.pt")

    frame = pd.DataFrame(samples, columns=list(PARAM_NAMES))
    frame.to_csv(csv_path, index=False)

    payload = {
        "x": torch.tensor(samples, dtype=torch.float32),
        "inputs": torch.tensor(samples, dtype=torch.float32),
        "input_columns": list(PARAM_NAMES),
        "normalization": {
            "low": PARAM_LOW.tolist(),
            "high": PARAM_HIGH.tolist(),
        },
        "metadata": {
            "sampler": "scipy.stats.qmc.LatinHypercube",
            "num_samples": n,
            "dimension": len(PARAM_NAMES),
            "seed": int(seed),
            "rounding": {
                "LineDensity_per_mm": "integer",
                "w1": "0.01",
                "Theta1_deg": "0.01",
                "Theta2_deg": "0.01",
                "Inc_Angle_deg": "continuous",
            },
        },
    }
    torch.save(payload, pt_path)
    return csv_path, pt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 LHS 光栅输入样本")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_LHS_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()

    samples = generate_lhs_samples(args.num_samples, args.seed)
    csv_path, pt_path = save_lhs_samples(
        samples,
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps({
        "num_samples": int(len(samples)),
        "dimension": int(samples.shape[1]),
        "seed": int(args.seed),
        "csv": str(csv_path),
        "pt": str(pt_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
