"""
Core/anomaly.py - SIT-Py Anomaly Detector (v2)
===============================================
Dual-mode anomaly detection:

  Mode A - Statistical Bridge (default at startup)
    Uses z-score rolling statistics via FeatureEngineer.
    Activates immediately, no training data required.
    No more 100% false positive rate on idle Windows systems.

  Mode B - Isolation Forest (activated after live data accumulation)
    Trained on real live_metrics.csv data to learn THIS machine's normal profile.
    Replaces Mode A once enough live data exists (RETRAIN_ROW_THRESHOLD rows).
    Includes temporal features: deltas, rolling mean/std, bandwidth spike ratio.

Key fixes from v1:
  - No longer trained on Kaggle security simulation data
  - No longer returns "Attack" for every idle system reading
  - Temporal features allow detection of gradual drift (memory leaks, etc.)
  - Label changed from "Attack/Normal" to "Alert/Normal" to reflect the
    system-monitoring (not security) use case
"""
from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.schema import SystemSnapshot, ANOMALY_FEATURE_COLS
from config import (
    MODELS_DIR, LIVE_CSV, RETRAIN_ROW_THRESHOLD, CONTAMINATION,
    ANOMALY_Z_THRESHOLD,
)

# ── Feature columns for IF model ───────────────────────────────────────────────
# Subset of ANOMALY_FEATURE_COLS that are numeric (task_priority is categorical)
IF_NUMERIC_COLS = [
    "cpu_pct", "ram_pct", "bandwidth_kbps", "packet_rate_pps",
    "cpu_delta", "ram_delta",
    "cpu_rolling_mean", "cpu_rolling_std",
    "bw_rolling_mean", "bw_spike",
]


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: str = LIVE_CSV):
    """
    Trains Isolation Forest on real live system data (NOT the Kaggle security set).
    The model learns what normal looks like for THIS machine.

    Requires at least RETRAIN_ROW_THRESHOLD rows in dataset_path.
    The CSV must have the unified schema columns (output of collect.py v2).

    Args:
        dataset_path: path to a CSV with unified schema columns.

    Returns:
        (model, scaler) - also saved to MODELS_DIR.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"[AnomalyDetector] Dataset not found: {dataset_path}\n"
            "Run Data/collect.py first to accumulate live metrics."
        )

    df = pd.read_csv(dataset_path)
    print(f"[AnomalyDetector] Dataset: {dataset_path} -> {df.shape}")

    # Ensure required columns exist (fill missing temporal cols with 0)
    for col in IF_NUMERIC_COLS:
        if col not in df.columns:
            print(f"  [warn] Column '{col}' missing - filling with 0")
            df[col] = 0.0

    df = df[IF_NUMERIC_COLS].dropna()
    print(f"[AnomalyDetector] Training rows after dropna: {len(df)}")

    if len(df) < 50:
        raise ValueError(
            f"[AnomalyDetector] Not enough data: {len(df)} rows (need ?50). "
            "Collect more live data first."
        )

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    model = IsolationForest(
        n_estimators  = 200,
        contamination = CONTAMINATION,
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X_scaled)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "anomaly_scaler.pkl"))

    # Quick self-eval
    preds  = model.predict(X_scaled)
    n_anom = int((preds == -1).sum())
    print(f"[AnomalyDetector] Training anomaly rate: {n_anom}/{len(df)} "
          f"({100*n_anom/len(df):.1f}%) - expected ?{CONTAMINATION*100:.0f}%")
    print("[AnomalyDetector] Model saved OK")

    return model, scaler


# ── Inference ──────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Dual-mode anomaly detector.

    Start-up behavior:
      - If anomaly_model.pkl exists -> loads and uses Isolation Forest (Mode B).
      - Otherwise -> operates in statistical bridge mode (Mode A) via FeatureEngineer.

    In either case, .predict() returns the same output schema:
        {
            "label":         "Normal" | "Alert",
            "anomaly_score": float,       # more negative = more anomalous
            "is_anomaly":    bool,
            "source":        "isolation_forest" | "stat_bridge" | "stat_bridge_warmup",
        }
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir  = models_dir
        self.model:  Optional[IsolationForest]  = None
        self.scaler: Optional[StandardScaler]   = None
        self.loaded: bool = False

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Attempts to load the IF model.
        Returns True if successful, False if no model exists yet.
        In the False case, the detector automatically falls back to stat bridge.
        """
        model_path  = os.path.join(self.models_dir, "anomaly_model.pkl")
        scaler_path = os.path.join(self.models_dir, "anomaly_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("[AnomalyDetector] No IF model found -> using statistical bridge mode.")
            self.loaded = False
            return False

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.loaded = True
        print("[AnomalyDetector] Isolation Forest model loaded OK")
        return True

    # ── Predict (IF mode) ──────────────────────────────────────────────────────

    def predict(self, snap_input: dict) -> dict:
        """
        Runs Isolation Forest on a pre-computed feature dict.

        Args:
            snap_input: dict from SystemSnapshot.to_anomaly_input()
                        (must contain IF_NUMERIC_COLS keys)

        Returns:
            {"label", "anomaly_score", "is_anomaly", "source"}
        """
        if not self.loaded:
            raise RuntimeError(
                "IF model not loaded. Call .load() first, or use "
                "FeatureEngineer.statistical_anomaly() for bridge mode."
            )

        assert self.model is not None and self.scaler is not None, \
            "Model or scaler not initialised — call .load() first"

        # Build feature vector (numeric only)
        row = [float(snap_input.get(col, 0.0)) for col in IF_NUMERIC_COLS]
        X   = np.array(row).reshape(1, -1)

        try:
            X_scaled  = self.scaler.transform(X)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}") from e

        raw_pred = int(self.model.predict(X_scaled)[0])        # -1 or 1
        score    = float(self.model.score_samples(X_scaled)[0])

        label = "Alert" if raw_pred == -1 else "Normal"

        return {
            "label":         label,
            "anomaly_score": round(score, 4),
            "is_anomaly":    raw_pred == -1,
            "source":        "isolation_forest",
        }

    def predict_batch(self, inputs: list) -> list:
        """Runs predict() on a list of input dicts."""
        return [self.predict(s) for s in inputs]

    @property
    def mode(self) -> str:
        return "isolation_forest" if self.loaded else "stat_bridge"


# ── Entry (train mode) ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys

    dataset = _sys.argv[1] if len(_sys.argv) > 1 else LIVE_CSV

    print("=" * 60)
    print("  SIT-Py | AnomalyDetector v2 | Training Mode")
    print("=" * 60)
    print(f"  Dataset: {dataset}")
    print()

    model, scaler = train(dataset_path=dataset)

    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)
    detector = AnomalyDetector()
    detector.load()

    # Simulate a normal snapshot (no enriched temporal features yet -> defaults to 0)
    normal = {
        "cpu_pct": 10.0, "ram_pct": 55.0, "bandwidth_kbps": 100.0,
        "packet_rate_pps": 50.0, "cpu_delta": 0.5, "ram_delta": 0.0,
        "cpu_rolling_mean": 9.0, "cpu_rolling_std": 1.5,
        "bw_rolling_mean": 120.0, "bw_spike": 0.8,
    }
    # Simulate a spike snapshot
    spike = {
        "cpu_pct": 92.0, "ram_pct": 87.0, "bandwidth_kbps": 50000.0,
        "packet_rate_pps": 8000.0, "cpu_delta": 45.0, "ram_delta": 12.0,
        "cpu_rolling_mean": 10.0, "cpu_rolling_std": 2.0,
        "bw_rolling_mean": 200.0, "bw_spike": 250.0,
    }

    r1 = detector.predict(normal)
    r2 = detector.predict(spike)
    print(f"\n  Normal snapshot -> {r1}")
    print(f"  Spike  snapshot -> {r2}")