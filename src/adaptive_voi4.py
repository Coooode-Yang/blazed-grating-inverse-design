from __future__ import annotations

from typing import Callable

import numpy as np


Array = np.ndarray


def _as_vector(value: Array, name: str) -> Array:
    """将输入转换为有限的一维浮点数组。"""

    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _rmse(left: Array, right: Array) -> float:
    left = _as_vector(left, "left")
    right = _as_vector(right, "right")
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left={left.shape}, right={right.shape}")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _objective_components(spectrum: Array, target: Array, derivative_weight: float) -> dict[str, float]:
    """返回论文决策规则使用的各项目标函数指标。"""

    spectrum = _as_vector(spectrum, "spectrum")
    target = _as_vector(target, "target")
    if spectrum.shape != target.shape:
        raise ValueError(f"Spectrum/target shape mismatch: {spectrum.shape} vs {target.shape}")
    if derivative_weight < 0:
        raise ValueError("derivative_weight must be non-negative")

    error = spectrum - target
    mse = float(np.mean(error**2))
    derivative_error = np.diff(spectrum) - np.diff(target)
    derivative_mse = float(np.mean(derivative_error**2)) if derivative_error.size else 0.0
    return {
        "objective": mse + float(derivative_weight) * derivative_mse,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "derivative_mse": derivative_mse,
    }


def _objective_matrix(member_spectra: Array, target: Array, derivative_weight: float) -> Array:
    """根据集成光谱计算形状为 [成员数, 候选数] 的目标函数值。"""

    member_spectra = np.asarray(member_spectra, dtype=float)
    if member_spectra.ndim != 3:
        raise ValueError(
            "member_spectra must have shape [n_members, n_candidates, n_wavelengths]"
        )
    target = _as_vector(target, "target")
    if member_spectra.shape[-1] != target.size:
        raise ValueError(
            "member_spectra wavelength dimension does not match target: "
            f"{member_spectra.shape[-1]} vs {target.size}"
        )
    if not np.all(np.isfinite(member_spectra)):
        raise ValueError("member_spectra contains NaN or infinite values")

    error = member_spectra - target[None, None, :]
    mse = np.mean(error**2, axis=-1)
    if target.size > 1:
        derivative_error = np.diff(member_spectra, axis=-1) - np.diff(target)[None, None, :]
        derivative_mse = np.mean(derivative_error**2, axis=-1)
    else:
        derivative_mse = np.zeros_like(mse)
    return mse + float(derivative_weight) * derivative_mse


def _validate_inputs(
    target: Array, proxy_mean: Array, proxy_std: Array
) -> tuple[Array, Array, Array]:
    target = _as_vector(target, "target")
    proxy_mean = np.asarray(proxy_mean, dtype=float)
    proxy_std = np.asarray(proxy_std, dtype=float)
    if proxy_mean.ndim != 2:
        raise ValueError("proxy_mean must have shape [n_candidates, n_wavelengths]")
    if proxy_std.shape != proxy_mean.shape:
        raise ValueError("proxy_std must have the same shape as proxy_mean")
    if proxy_mean.shape[-1] != target.size:
        raise ValueError(
            "proxy_mean wavelength dimension does not match target: "
            f"{proxy_mean.shape[-1]} vs {target.size}"
        )
    if not np.all(np.isfinite(proxy_mean)) or not np.all(np.isfinite(proxy_std)):
        raise ValueError("proxy_mean/proxy_std contains NaN or infinite values")
    if np.any(proxy_std < 0):
        raise ValueError("proxy_std must be non-negative")
    return target, proxy_mean, proxy_std


def _certified(
    observed_objectives: dict[int, float],
    remaining: list[int],
    lower: Array,
    tolerance: float,
    min_queries: int,
) -> tuple[bool, float | None, float | None, int]:
    """检查剩余候选的下界是否不可能优于当前已验证的最佳候选。"""

    if len(observed_objectives) < min_queries or not observed_objectives:
        return False, None, None, len(remaining)
    current_best = float(min(observed_objectives.values()))
    if not remaining:
        return True, current_best, None, 0
    remaining_lower = np.asarray([lower[index] for index in remaining], dtype=float)
    if not np.all(np.isfinite(remaining_lower)):
        return False, current_best, None, len(remaining)
    minimum_lower = float(np.min(remaining_lower))
    able_to_beat = int(np.sum(remaining_lower < current_best - tolerance))
    return able_to_beat == 0, current_best, minimum_lower, able_to_beat


def run_adaptive_voi4(
    target: Array,
    proxy_mean: Array,
    proxy_std: Array,
    query: Callable[[int], Array],
    *,
    member_spectra: Array | None = None,
    max_calls: int = 4,
    min_queries: int = 2,
    derivative_weight: float = 0.1,
    risk_multiplier: float = 1.96,
    uncertainty_bonus_weight: float = 0.1,
    rank_reward_weight: float = 1e-6,
    risk_tolerance: float = 0.0,
    target_tolerance: float | None = None,
) -> dict:
    """运行 Adaptive-VOI-4 策略。

    参数
    ----------
    target:
        形状为 ``[波长数]`` 的目标效率光谱。
    proxy_mean, proxy_std:
        形状为 ``[候选数, 波长数]`` 的集成均值和标准差。
        当未提供 ``member_spectra`` 时，标准差仅用于兼容旧调用方式。
    query:
        接收候选索引并返回其物理 RCWA 光谱的可调用对象。
        每个候选索引最多查询一次。
    member_spectra:
        可选的精确前向集成预测，形状为
        ``[3, 候选数, 波长数]``。论文中使用的目标函数空间不确定性需要提供此数组。
    max_calls:
        物理查询预算，正式实验中的默认值为 4。
    min_queries:
        在允许进行风险区间认证或基于容差提前停止之前，必须完成的最少物理查询次数。
    target_tolerance:
        可选的 RMSE 停止条件。正式实验通常将其设为 ``None``，
        仅通过风险认证或预算耗尽停止。
    """

    target, proxy_mean, proxy_std = _validate_inputs(target, proxy_mean, proxy_std)
    if not callable(query):
        raise TypeError("query must be callable")
    n_candidates = proxy_mean.shape[0]
    if max_calls < 0 or min_queries < 0:
        raise ValueError("max_calls and min_queries must be non-negative")
    if risk_multiplier < 0 or uncertainty_bonus_weight < 0 or rank_reward_weight < 0:
        raise ValueError("risk and reward weights must be non-negative")
    if risk_tolerance < 0:
        raise ValueError("risk_tolerance must be non-negative")

    # 精确的目标函数空间集成统计量是论文中规定的路径。
    # 回退路径用于兼容旧的 dry-run 调用方，但信息量明确较低，
    # 因为光谱离散程度不等同于目标函数离散程度。
    if member_spectra is not None:
        member_spectra = np.asarray(member_spectra, dtype=float)
        if member_spectra.ndim != 3:
            raise ValueError(
                "member_spectra must have shape [n_members, n_candidates, n_wavelengths]"
            )
        if member_spectra.shape[1:] != proxy_mean.shape:
            raise ValueError(
                "member_spectra candidate/spectrum shape does not match proxy_mean: "
                f"{member_spectra.shape[1:]} vs {proxy_mean.shape}"
            )
        member_objectives = _objective_matrix(member_spectra, target, derivative_weight)
        proxy_objective = member_objectives.mean(axis=0)
        objective_uncertainty = member_objectives.std(axis=0, ddof=0)
        ensemble_mean = member_spectra.mean(axis=0)
        uncertainty_source = "objective_ensemble_std"
    else:
        # 为只有均值/标准差光谱的调用方提供向后兼容的近似方法。
        # 不应将其描述为经过校准的目标函数风险。
        proxy_objective = np.asarray(
            [_objective_components(row, target, derivative_weight)["objective"] for row in proxy_mean]
        )
        objective_uncertainty = np.sqrt(np.mean(proxy_std**2, axis=1))
        ensemble_mean = proxy_mean
        member_objectives = None
        uncertainty_source = "rms_spectral_std_fallback"

    proxy_rmse = np.asarray([_rmse(row, target) for row in ensemble_mean], dtype=float)
    spectral_uncertainty = np.sqrt(np.mean(proxy_std**2, axis=1))
    risk_lower = proxy_objective - float(risk_multiplier) * objective_uncertainty
    risk_upper = proxy_objective + float(risk_multiplier) * objective_uncertainty
    proxy_rank = np.argsort(np.argsort(proxy_objective, kind="stable"), kind="stable") + 1

    queried: list[int] = []
    observed_rmse: dict[int, float] = {}
    observed_objective: dict[int, float] = {}
    trace: list[dict] = []
    reason = "budget_exhausted"
    budget = min(int(max_calls), n_candidates)
    minimum_queries = min(int(min_queries), budget)

    while len(queried) < budget:
        remaining = [index for index in range(n_candidates) if index not in queried]
        certified, current_best, minimum_lower, able_to_beat = _certified(
            observed_objective, remaining, risk_lower, float(risk_tolerance), minimum_queries
        )
        if certified:
            reason = "risk_interval_certified"
            break

        if not remaining:
            break

        if current_best is None:
            current_best = float(np.min(proxy_objective))
        voi_rows: dict[int, dict[str, float]] = {}
        for index in remaining:
            lower = float(risk_lower[index])
            upper = float(risk_upper[index])
            width = upper - lower
            if np.isfinite(width) and width >= 0:
                width = max(width, 1e-12)
                probability = float(np.clip((current_best - lower) / width, 0.0, 1.0))
                expected_gain = probability * max(
                    current_best - float(proxy_objective[index]) + 0.5 * width, 0.0
                )
                uncertainty_bonus = float(uncertainty_bonus_weight) * probability * width
            else:
                # 无效区间会被降低优先级，但仍然允许查询。
                width = float("nan")
                probability = 0.0
                expected_gain = max(current_best - float(proxy_objective[index]), 0.0)
                uncertainty_bonus = 0.0
            rank_bonus = probability * (1.0 / float(proxy_rank[index])) * float(rank_reward_weight)
            voi_rows[index] = {
                "predicted_selection_probability": probability,
                "expected_gain": expected_gain,
                "uncertainty_bonus": uncertainty_bonus,
                "rank_bonus": rank_bonus,
                "voi": expected_gain + uncertainty_bonus + rank_bonus,
                "risk_width": width,
                "risk_lower": lower,
                "risk_upper": upper,
            }

        candidate = max(
            remaining,
            key=lambda index: (
                voi_rows[index]["voi"],
                voi_rows[index]["predicted_selection_probability"],
                -int(proxy_rank[index]),
                -int(index),
            ),
        )

        spectrum = _as_vector(query(int(candidate)), "query result")
        if spectrum.shape != target.shape:
            raise ValueError(
                f"query({candidate}) returned {spectrum.shape}; expected {target.shape}"
            )
        metrics = _objective_components(spectrum, target, derivative_weight)
        queried.append(int(candidate))
        observed_rmse[int(candidate)] = metrics["rmse"]
        observed_objective[int(candidate)] = metrics["objective"]

        best_observed_rmse = float(min(observed_rmse.values()))
        best_observed_objective = float(min(observed_objective.values()))
        row = voi_rows[candidate].copy()
        row.update(
            {
                "call": len(queried),
                "candidate_index": int(candidate),
                "proxy_rmse": float(proxy_rmse[candidate]),
                "proxy_objective": float(proxy_objective[candidate]),
                "proxy_uncertainty": float(objective_uncertainty[candidate]),
                "spectral_uncertainty": float(spectral_uncertainty[candidate]),
                "rcwa_rmse": metrics["rmse"],
                "rcwa_objective": metrics["objective"],
                "best_observed_rmse": best_observed_rmse,
                "best_observed_objective": best_observed_objective,
                "uncertainty_source": uncertainty_source,
                "risk_certified_before_query": False,
                "minimum_unverified_lower": minimum_lower,
                "unverified_candidates_still_able_to_beat": able_to_beat,
            }
        )
        trace.append(row)

        # 容差是可选的运行捷径，不能替代风险区间规则。
        # 因此，在完成规定数量的物理观测之前不会启用该条件。
        if (
            target_tolerance is not None
            and len(queried) >= minimum_queries
            and best_observed_rmse <= float(target_tolerance)
        ):
            reason = "early_stop_target_tolerance"
            break

    if observed_objective:
        winner = min(observed_objective, key=observed_objective.get)
    else:
        winner = int(np.argmin(proxy_objective))
        reason = "no_physical_query_proxy_fallback"

    return {
        "selected_candidate": int(winner),
        "num_calls": len(queried),
        "queried_candidates": queried,
        "observed_rmse": observed_rmse,
        "observed_objective": observed_objective,
        "proxy_rmse": proxy_rmse.tolist(),
        "proxy_objective": proxy_objective.tolist(),
        "proxy_uncertainty": objective_uncertainty.tolist(),
        "spectral_uncertainty": spectral_uncertainty.tolist(),
        "risk_lower": risk_lower.tolist(),
        "risk_upper": risk_upper.tolist(),
        "proxy_rank": proxy_rank.astype(int).tolist(),
        "uncertainty_source": uncertainty_source,
        "stop_reason": reason,
        "trace": trace,
    }
__all__ = ["run_adaptive_voi4"]
