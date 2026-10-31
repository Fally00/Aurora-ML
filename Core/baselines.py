"""
Naive baseline predictors for skill score computation.

Provides repeat-last-value, moving average, and exponential smoothing baselines
for time-series forecasting tasks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def repeat_last_value(y: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Repeat the last observed value for each forecast step."""
    if len(y) == 0:
        return np.array([])
    last = y[-1]
    return np.full(len(y) - horizon, last)


def moving_average(y: np.ndarray, window: int = 5, horizon: int = 1) -> np.ndarray:
    """Simple moving average baseline."""
    if len(y) <= window:
        return np.array([])
    preds = []
    for i in range(window, len(y) - horizon + 1):
        preds.append(np.mean(y[i - window:i]))
    return np.array(preds)


def exponential_smoothing(y: np.ndarray, alpha: float = 0.3, horizon: int = 1) -> np.ndarray:
    """Exponential smoothing baseline."""
    if len(y) == 0:
        return np.array([])
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * y[i - 1] + (1 - alpha) * smoothed[i - 1]
    return smoothed[:-horizon] if horizon > 0 else smoothed


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE and RMSE for baseline predictions."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan")}
    # Align lengths
    n = min(len(y_true), len(y_pred))
    y_true = y_true[-n:]
    y_pred = y_pred[-n:]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": float(mae), "RMSE": float(rmse)}


def compute_skill_score(model_mae: float, baseline_mae: float) -> float:
    """Skill score = 1 - model_err / baseline_err. Higher is better."""
    if baseline_mae <= 0 or np.isnan(baseline_mae) or np.isnan(model_mae):
        return float("nan")
    return 1.0 - (model_mae / baseline_mae)


def evaluate_baselines(
    y: np.ndarray,
    horizons: list[int] = None,
    baseline_types: list[Literal["repeat_last", "moving_avg", "exp_smooth"]] = None,
) -> dict:
    """Evaluate all baseline types at multiple horizons."""
    if horizons is None:
        horizons = [1, 5, 30]
    if baseline_types is None:
        baseline_types = ["repeat_last", "moving_avg", "exp_smooth"]

    results = {}
    for h in horizons:
        results[h] = {}
        y_true = y[h:]
        for bt in baseline_types:
            if bt == "repeat_last":
                y_pred = repeat_last_value(y, horizon=h)
            elif bt == "moving_avg":
                y_pred = moving_average(y, window=5, horizon=h)
            elif bt == "exp_smooth":
                y_pred = exponential_smoothing(y, alpha=0.3, horizon=h)
            else:
                continue
            metrics = compute_baseline_metrics(y_true, y_pred)
            results[h][bt] = metrics
    return results


def evaluate_model_vs_baselines(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    baseline_results: dict,
    model_name: str = "model",
) -> dict:
    """Compare model MAE/RMSE against baselines at each horizon."""
    report = {}
    for horizon, baselines in baseline_results.items():
        report[horizon] = {}
        for bt, metrics in baselines.items():
            if np.isnan(metrics["MAE"]):
                continue
            model_metrics = compute_baseline_metrics(
                y_true[horizon:], y_pred_model[horizon:]
            )
            skill = compute_skill_score(model_metrics["MAE"], metrics["MAE"])
            report[horizon][bt] = {
                f"{model_name}_MAE": model_metrics["MAE"],
                f"{model_name}_RMSE": model_metrics["RMSE"],
                f"{bt}_MAE": metrics["MAE"],
                f"{bt}_RMSE": metrics["RMSE"],
                "skill_score": skill,
            }
    return report