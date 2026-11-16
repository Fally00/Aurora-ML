"""
Resource usage predictor.

The predictor is trained from the new per-agent predictor dataset by default
and can also be retrained from a merged live dataset prepared by
pipeline/retrain.py.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.baselines import (
    evaluate_baselines,
    evaluate_model_vs_baselines,
    compute_skill_score,
)
from Core.schema import (
    PREDICTOR_FEATURE_COLS,
    PREDICTOR_RAW_COLS,
    PREDICTOR_TARGET_COLS,
    coerce_numeric_columns,
    ensure_columns,
    normalize_columns,
)
from Core.base_model import BaseModel
from config import LIVE_CSV, MODELS_DIR, PREDICTOR_DATASET_CSV, PREDICTOR_MODEL_PARAMS, RANDOM_STATE, WINDOW_SIZE


FEATURE_COLS = PREDICTOR_FEATURE_COLS
RAW_COLS = PREDICTOR_RAW_COLS
TARGET_COLS = PREDICTOR_TARGET_COLS


def _load_training_dataset(path: str) -> Optional[pd.DataFrame]:
    """Load a raw predictor dataset if it exists."""
    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = ensure_columns(df, RAW_COLS)
    df = coerce_numeric_columns(df, RAW_COLS)
    df = df.dropna(subset=RAW_COLS)
    return df if len(df) > WINDOW_SIZE + 1 else None


def build_windows(df: pd.DataFrame, window: int = WINDOW_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Build predictor feature windows from the raw resource telemetry columns."""
    if len(df) <= window:
        raise ValueError(
            f"Predictor dataset needs more than {window} rows, got {len(df)}."
        )

    cpu = df["cpu_pct"].to_numpy(dtype=np.float32)
    ram = df["ram_pct"].to_numpy(dtype=np.float32)
    disk = df["disk_pct"].to_numpy(dtype=np.float32)
    bw = df["bandwidth_kbps"].to_numpy(dtype=np.float32)
    pkts = df["packet_rate_pps"].to_numpy(dtype=np.float32)

    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []

    for idx in range(window, len(df)):
        cpu_win = cpu[idx - window : idx]
        ram_win = ram[idx - window : idx]
        bw_win = bw[idx - window : idx]

        bw_mean = float(np.mean(bw_win)) if float(np.sum(bw_win)) > 0 else 0.0
        bw_spike = float(bw[idx - 1] / bw_mean) if bw_mean > 0 else 1.0

        features = {
            "cpu_pct": float(cpu[idx - 1]),
            "ram_pct": float(ram[idx - 1]),
            "disk_pct": float(disk[idx - 1]),
            "bandwidth_kbps": float(bw[idx - 1]),
            "packet_rate_pps": float(pkts[idx - 1]),
            "cpu_delta": float(cpu[idx - 1] - cpu[idx - 2]) if idx >= 2 else 0.0,
            "ram_delta": float(ram[idx - 1] - ram[idx - 2]) if idx >= 2 else 0.0,
            "cpu_rolling_mean": float(np.mean(cpu_win)),
            "ram_rolling_mean": float(np.mean(ram_win)),
            "cpu_rolling_std": float(np.std(cpu_win)),
            "ram_rolling_std": float(np.std(ram_win)),
            "bw_rolling_mean": bw_mean,
            "bw_spike": bw_spike,
        }

        x_rows.append([features[col] for col in FEATURE_COLS])
        y_rows.append([float(cpu[idx]), float(ram[idx]), float(disk[idx])])

    return (
        np.array(x_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
    )


def train(dataset_path: str | None = None, window: int = WINDOW_SIZE) -> dict:
    """
    Train the predictor with walk-forward validation and baseline comparison.

    Priority:
      1. Explicit dataset_path
      2. Base predictor dataset
      3. Unified live_metrics.csv
    """
    source = "none"
    df = None

    if dataset_path:
        df = _load_training_dataset(dataset_path)
        source = dataset_path
    if df is None:
        df = _load_training_dataset(PREDICTOR_DATASET_CSV)
        source = PREDICTOR_DATASET_CSV
    if df is None:
        df = _load_training_dataset(LIVE_CSV)
        source = LIVE_CSV
    if df is None:
        raise FileNotFoundError(
            "No predictor dataset found. Expected one of: "
            f"{dataset_path!r}, {PREDICTOR_DATASET_CSV!r}, or {LIVE_CSV!r}."
        )

    print(f"[Predictor] Dataset source : {source}")
    print(f"[Predictor] Dataset shape  : {df.shape}")
    print(f"[Predictor] Building windows (size={window})...")

    x, y = build_windows(df, window)
    print(f"[Predictor] Training samples: X={x.shape}, y={y.shape}")

    # Chronological split - no random shuffle for time series
    if "timestamp" in df.columns:
        # The build_windows function uses sequential rows, so x/y are already in time order
        split_idx = int(len(x) * 0.8)
        x_train, x_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"[Predictor] Chronological split: train={len(x_train)}, test={len(x_test)}")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        print(f"[Predictor] Random split: train={len(x_train)}, test={len(x_test)}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestRegressor(**PREDICTOR_MODEL_PARAMS)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    # Walk-forward validation on test set
    n_splits = 5
    test_size = len(x_test)
    fold_size = test_size // n_splits
    wf_mae = {target: [] for target in TARGET_COLS}
    wf_rmse = {target: [] for target in TARGET_COLS}
    wf_r2 = {target: [] for target in TARGET_COLS}

    for fold in range(n_splits):
        start = fold * fold_size
        end = start + fold_size if fold < n_splits - 1 else test_size
        if end - start < 10:
            continue
        x_fold = x_test_scaled[start:end]
        y_fold = y_test[start:end]
        y_fold_pred = model.predict(x_fold)
        for idx, target in enumerate(TARGET_COLS):
            mae = mean_absolute_error(y_fold[:, idx], y_fold_pred[:, idx])
            rmse = np.sqrt(mean_squared_error(y_fold[:, idx], y_fold_pred[:, idx]))
            r2 = r2_score(y_fold[:, idx], y_fold_pred[:, idx])
            wf_mae[target].append(mae)
            wf_rmse[target].append(rmse)
            wf_r2[target].append(r2)

    print("\n" + "=" * 60)
    print("  Predictor Walk-Forward Validation (5 folds)")
    print("=" * 60)
    metrics: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, idx], y_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_test[:, idx], y_pred[:, idx]))
        r2 = r2_score(y_test[:, idx], y_pred[:, idx])
        metrics[target] = {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4), "R2": round(float(r2), 4)}
        print(f"  {target:10s} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    if any(wf_mae[t] for t in TARGET_COLS):
        print("\n  Walk-forward averages:")
        for target in TARGET_COLS:
            if wf_mae[target]:
                print(f"    {target:10s} -> MAE: {np.mean(wf_mae[target]):.4f} ± {np.std(wf_mae[target]):.4f} | RMSE: {np.mean(wf_rmse[target]):.4f} ± {np.std(wf_rmse[target]):.4f} | R2: {np.mean(wf_r2[target]):.4f} ± {np.std(wf_r2[target]):.4f}")

    # Baseline comparison on test set (using raw target values from original df)
    # We need the raw time series for baselines - use the last column (cpu) as proxy
    # Actually, let's use the test target values directly for baseline comparison
    print("\n" + "=" * 60)
    print("  Baseline Comparison (vs Repeat-Last-Value)")
    print("=" * 60)

    # For baseline comparison, we need the original time series values
    # The test set corresponds to the last 20% of the windowed data
    # We'll compare on the first target (cpu) as representative
    test_cpu = y_test[:, 0]  # cpu target
    pred_cpu = y_pred[:, 0]
    model_mae = mean_absolute_error(test_cpu, pred_cpu)
    model_rmse = np.sqrt(mean_squared_error(test_cpu, pred_cpu))

    # Repeat-last-value baseline on test set
    baseline_pred = np.roll(test_cpu, 1)
    baseline_pred[0] = test_cpu[0]  # first prediction has no previous
    baseline_mae = mean_absolute_error(test_cpu[1:], baseline_pred[1:])
    baseline_rmse = np.sqrt(mean_squared_error(test_cpu[1:], baseline_pred[1:]))
    skill = compute_skill_score(model_mae, baseline_mae)

    print(f"  Model MAE:  {model_mae:.4f} | RMSE: {model_rmse:.4f}")
    print(f"  Naive MAE:  {baseline_mae:.4f} | RMSE: {baseline_rmse:.4f}")
    print(f"  Skill Score: {skill:.4f} (1 - model/naive, higher=better)")

    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances), key=lambda item: -item[1])
    print("\n[Predictor] Feature importances:")
    for feat, importance in ranked[:8]:
        print(f"  {feat:22s} {importance:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save and hash model
    model_path = os.path.join(MODELS_DIR, "predictor_model.pkl")
    joblib.dump(model, model_path)
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    hash_path = model_path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(model_hash)
    # Save and hash scaler
    scaler_path = os.path.join(MODELS_DIR, "predictor_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    with open(scaler_path, "rb") as f:
        scaler_hash = hashlib.sha256(f.read()).hexdigest()
    hash_path = scaler_path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(scaler_hash)
    print(f"\n[Predictor] Model and scaler saved to {MODELS_DIR}")

    return metrics


class ResourcePredictor(BaseModel):
    """Load the trained model and predict next-cycle resource usage."""

    model_files = ["predictor_model.pkl", "predictor_scaler.pkl"]

    def __init__(self, models_dir: str = MODELS_DIR):
        super().__init__(models_dir)
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None

    def load(self) -> bool:
        """Load the predictor model and scaler."""
        if not super().load():
            return False

        self.model = self.get_artifact("predictor_model.pkl")
        self.scaler = self.get_artifact("predictor_scaler.pkl")

        if any(artifact is None for artifact in [self.model, self.scaler]):
            print("[Predictor] One or more model artifacts not found.")
            self.loaded = False
            return False

        expected_features = len(FEATURE_COLS)
        model_features = getattr(self.model, "n_features_in_", None)
        scaler_features = getattr(self.scaler, "n_features_in_", None)
        if model_features != expected_features or scaler_features != expected_features:
            print(
                "[Predictor] Predictor artifacts are from an older schema. "
                "Retrain with `python Core/predictor.py` or `python pipeline/retrain.py --force`."
            )
            self.loaded = False
            return False

        self.loaded = True
        print(f"[Predictor] Model loaded with {len(FEATURE_COLS)} live features")
        return True

    def predict(self, feature_dict: dict[str, float]) -> dict[str, float]:
        """Predict the next CPU, RAM, and disk percentages."""
        if not self.loaded:
            raise RuntimeError("Call .load() before .predict()")
        assert self.model is not None and self.scaler is not None

        row = [float(feature_dict.get(col, 0.0)) for col in FEATURE_COLS]
        x = np.array(row, dtype=np.float32).reshape(1, -1)
        pred = self.model.predict(self.scaler.transform(x))[0]

        return {
            "cpu_pct": round(float(np.clip(pred[0], 0, 100)), 2),
            "ram_pct": round(float(np.clip(pred[1], 0, 100)), 2),
            "disk_pct": round(float(np.clip(pred[2], 0, 100)), 2),
        }

    def feature_importances(self) -> dict[str, float]:
        """Return model feature importances for inspection tools."""
        if not self.loaded or self.model is None:
            return {}
        return dict(
            sorted(
                zip(FEATURE_COLS, [round(float(v), 4) for v in self.model.feature_importances_]),
                key=lambda item: -item[1],
            )
        )

    @property
    def is_ready(self) -> bool:
        return self.loaded


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  SIT-Py | ResourcePredictor | Training Mode")
    print("=" * 60)
    metrics = train(dataset_path=dataset)

    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    for target, values in metrics.items():
        print(f"  {target:12s} -> MAE: {values['MAE']} | R2: {values['R2']}")

    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)
    predictor = ResourcePredictor()
    predictor.load()
    result = predictor.predict(
        {
            "cpu_pct": 12.0,
            "ram_pct": 56.0,
            "disk_pct": 34.7,
            "bandwidth_kbps": 150.0,
            "packet_rate_pps": 60.0,
            "cpu_delta": -2.0,
            "ram_delta": 0.1,
            "cpu_rolling_mean": 10.5,
            "ram_rolling_mean": 55.8,
            "cpu_rolling_std": 2.1,
            "ram_rolling_std": 0.3,
            "bw_rolling_mean": 120.0,
            "bw_spike": 1.25,
        }
    )
    print(result)
