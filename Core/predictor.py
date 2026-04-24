"""
Core/predictor.py - SIT-Py Resource Predictor (v2)
====================================================
Predicts next-cycle CPU%, RAM%, and Disk% usage.

Key changes from v1:
  - Random Forest Regressor replaces Linear Regression
    -> handles non-linear workload spikes, scheduling noise
    -> single multi-output model replaces 3 separate models
    -> feature importances available for dashboard explainability
  - Feature set uses engineered temporal statistics instead of
    a raw flattened time window (25 dumb floats -> 13 meaningful ones)
  - WINDOW_SIZE increased from 5 -> 15 (30 seconds of context)
  - Trains on unified schema columns from live_metrics.csv
  - Falls back to Kaggle motherboard dataset for cold start

LSTM upgrade path:
  - Feature engineering layer (feature_engineering.py) is already in place
  - When 10k+ live rows exist, LSTM can replace RF by swapping train/predict
    while keeping the same schema interface intact
"""
from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Optional

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.schema import SystemSnapshot, PREDICTOR_FEATURE_COLS, PREDICTOR_TARGET_COLS
from config import (
    MODELS_DIR, LIVE_CSV, KAGGLE_MOTHERBOARD_CSV, WINDOW_SIZE,
)

# ── Feature / target columns ──────────────────────────────────────────────────
FEATURE_COLS = PREDICTOR_FEATURE_COLS   # defined in schema.py
TARGET_COLS  = PREDICTOR_TARGET_COLS    # ["cpu_pct", "ram_pct", "disk_pct"]


# ── Dataset adapters ──────────────────────────────────────────────────────────

def _load_live_dataset(path: str) -> Optional[pd.DataFrame]:
    """Loads live_metrics.csv if it has enough rows for window building."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    return df if len(df) > WINDOW_SIZE + 10 else None


def _load_kaggle_motherboard(path: str) -> Optional[pd.DataFrame]:
    """
    Loads the Kaggle motherboard CSV and maps its columns to unified schema.
    Used as cold-start training data when live data is insufficient.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={
        "CPUUsage":    "cpu_pct",
        "RAMUsage":    "ram_pct",
        "DiskUsage":   "disk_pct",
        "Temperature": "temperature_c",
        "FanSpeed":    "fan_rpm",
    })
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maps old schema column names to unified names."""
    rename_map = {
        "CPUUsage":    "cpu_pct",
        "RAMUsage":    "ram_pct",
        "DiskUsage":   "disk_pct",
        "CPU_Usage":   "cpu_pct",
        "Memory_Usage":"ram_pct",
        "Bandwidth":   "bandwidth_kbps",
        "Packet_Rate": "packet_rate_pps",
        "Temperature": "temperature_c",
        "FanSpeed":    "fan_rpm",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# ── Sliding Window Builder ─────────────────────────────────────────────────────

def build_windows(df: pd.DataFrame, window: int = WINDOW_SIZE):
    """
    Builds (X, y) training pairs using a sliding temporal window.
    Computes rolling stats on-the-fly during training.
    """
    required = ["cpu_pct", "ram_pct", "disk_pct"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: '{col}'")

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    cpu  = df["cpu_pct"].values
    ram  = df["ram_pct"].values
    disk = df["disk_pct"].values
    bw   = df["bandwidth_kbps"].values if "bandwidth_kbps" in df.columns else np.zeros(len(df))
    pkts = df["packet_rate_pps"].values if "packet_rate_pps" in df.columns else np.zeros(len(df))

    X_list, y_list = [], []

    for i in range(window, len(df)):
        cpu_win = cpu[i - window : i]
        ram_win = ram[i - window : i]
        bw_win  = bw[i - window : i]

        bw_mean  = float(np.mean(bw_win)) if bw_win.sum() > 0 else 0.0
        bw_spike = float(bw[i - 1] / bw_mean) if bw_mean > 0 else 1.0

        features = {
            "cpu_pct":          cpu[i - 1],
            "ram_pct":          ram[i - 1],
            "disk_pct":         disk[i - 1],
            "bandwidth_kbps":   bw[i - 1],
            "packet_rate_pps":  pkts[i - 1],
            "cpu_delta":        cpu[i - 1] - cpu[i - 2] if i >= 2 else 0.0,
            "ram_delta":        ram[i - 1] - ram[i - 2] if i >= 2 else 0.0,
            "cpu_rolling_mean": float(np.mean(cpu_win)),
            "ram_rolling_mean": float(np.mean(ram_win)),
            "cpu_rolling_std":  float(np.std(cpu_win)),
            "ram_rolling_std":  float(np.std(ram_win)),
            "bw_rolling_mean":  bw_mean,
            "bw_spike":         bw_spike,
        }

        X_list.append([features[c] for c in FEATURE_COLS])
        y_list.append([cpu[i], ram[i], disk[i]])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: str = None, window: int = WINDOW_SIZE) -> dict:
    """
    Trains a multi-output Random Forest Regressor.
    Priority: live data -> Kaggle motherboard fallback.
    """
    df = None
    source = "none"

    if dataset_path:
        df = _load_live_dataset(dataset_path)
        source = "provided"
    if df is None:
        df = _load_live_dataset(LIVE_CSV)
        source = "live_metrics.csv"
    if df is None:
        df = _load_kaggle_motherboard(KAGGLE_MOTHERBOARD_CSV)
        source = "kaggle_motherboard (cold start)"
    if df is None:
        raise FileNotFoundError(
            "No training dataset found. Run Data/collect.py to accumulate live data, "
            f"or ensure {KAGGLE_MOTHERBOARD_CSV} exists."
        )

    print(f"[Predictor] Dataset source : {source}")
    print(f"[Predictor] Dataset shape  : {df.shape}")
    print(f"[Predictor] Building windows (size={window})...")

    X, y = build_windows(df, window)
    print(f"[Predictor] Training samples: X={X.shape}, y={y.shape}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("[Predictor] Training Random Forest Regressor (multi-output)...")
    model = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 15,
        min_samples_leaf = 3,
        max_features     = "sqrt",
        random_state     = 42,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    metrics = {}
    for idx, target in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, idx], y_pred[:, idx])
        r2  = r2_score(y_test[:, idx], y_pred[:, idx])
        metrics[target] = {"MAE": round(mae, 4), "R2": round(r2, 4)}
        print(f"  {target:10s} -> MAE: {mae:.4f} | R2: {r2:.4f}")

    # Feature importances
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    print("\n[Predictor] Feature Importances:")
    for feat, imp in ranked[:8]:
        bar = "#" * int(imp * 40)
        print(f"  {feat:22s} {bar} {imp:.4f}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODELS_DIR, "predictor_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "predictor_scaler.pkl"))
    print(f"\n[Predictor] Model + scaler saved to {MODELS_DIR} OK")

    return metrics


# ── Inference ──────────────────────────────────────────────────────────────────

class ResourcePredictor:
    """
    Loads the trained Random Forest and predicts next-cycle resource usage.

    Usage:
        predictor = ResourcePredictor()
        predictor.load()
        result = predictor.predict(snap.to_predictor_input())
        # -> {"cpu_pct": 12.3, "ram_pct": 57.1, "disk_pct": 34.7}
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model:  Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler]        = None
        self.loaded: bool = False

    def load(self):
        model_path  = os.path.join(self.models_dir, "predictor_model.pkl")
        scaler_path = os.path.join(self.models_dir, "predictor_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Predictor model not found in {self.models_dir}. "
                "Run 'python Core/predictor.py' to train."
            )

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.loaded = True
        print(f"[Predictor] Model + scaler loaded OK (features: {len(FEATURE_COLS)})")

    def predict(self, feature_dict: dict) -> dict:
        """
        Predicts next-cycle CPU, RAM, Disk.

        Args:
            feature_dict: output of SystemSnapshot.to_predictor_input()

        Returns:
            {"cpu_pct": float, "ram_pct": float, "disk_pct": float}
        """
        if not self.loaded:
            raise RuntimeError("Call .load() before .predict()")

        assert self.model is not None and self.scaler is not None, \
            "Model or scaler not initialised — call .load() first"

        row      = [float(feature_dict.get(col, 0.0)) for col in FEATURE_COLS]
        X        = np.array(row, dtype=np.float32).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        pred     = self.model.predict(X_scaled)[0]   # shape: (3,)

        return {
            "cpu_pct":  round(float(np.clip(pred[0], 0, 100)), 2),
            "ram_pct":  round(float(np.clip(pred[1], 0, 100)), 2),
            "disk_pct": round(float(np.clip(pred[2], 0, 100)), 2),
        }

    def feature_importances(self) -> dict:
        """Feature importances dict (for dashboard explainability)."""
        if not self.loaded or self.model is None:
            return {}
        imps = self.model.feature_importances_
        return dict(sorted(
            zip(FEATURE_COLS, [round(float(v), 4) for v in imps]),
            key=lambda x: -x[1]
        ))

    @property
    def is_ready(self) -> bool:
        return self.loaded


# ── Entry (train mode) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    dataset = _sys.argv[1] if len(_sys.argv) > 1 else None

    print("=" * 60)
    print("  SIT-Py | ResourcePredictor v2 | Training Mode")
    print("=" * 60)

    metrics = train(dataset_path=dataset)

    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    for target, m in metrics.items():
        print(f"  {target:12s} -> MAE: {m['MAE']} | R²: {m['R2']}")

    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)
    predictor = ResourcePredictor()
    predictor.load()

    dummy = {
        "cpu_pct": 12.0, "ram_pct": 56.0, "disk_pct": 34.7,
        "bandwidth_kbps": 150.0, "packet_rate_pps": 60.0,
        "cpu_delta": -2.0, "ram_delta": 0.1,
        "cpu_rolling_mean": 10.5, "ram_rolling_mean": 55.8,
        "cpu_rolling_std": 2.1, "ram_rolling_std": 0.3,
        "bw_rolling_mean": 120.0, "bw_spike": 1.25,
    }
    result = predictor.predict(dummy)
    print(f"  Input:  CPU=12%, RAM=56%, Disk=34.7%")
    print(f"  Predicted next -> {result}")
    print(f"\n  Top feature importances:")
    for feat, imp in list(predictor.feature_importances().items())[:5]:
        print(f"    {feat:22s}: {imp:.4f}")