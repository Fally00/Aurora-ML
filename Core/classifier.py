"""
Core/classifier.py - SIT-Py Workload Classifier (v2)
======================================================
Classifies live system state into: Low / Medium / High / Critical

Key changes from v1:
  - Feature set purged of dead live-mode features:
      REMOVED: Failed_Logins, Malware_Alerts, Intrusion_Alerts (always 0)
      REMOVED: Traffic_Type (hardcoded to "HTTP")
      ADDED:   process_count, cpu_rolling_mean, ram_rolling_mean, bw_spike
  - Now uses unified schema column names from SystemSnapshot
  - Rule-based label generator added for auto-labeling live_metrics.csv
  - Training supports both Kaggle dataset (adapted) and live data
"""
from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Dict

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.schema import SystemSnapshot, CLASSIFIER_FEATURE_COLS
from config import MODELS_DIR, LIVE_CSV, KAGGLE_SECURITY_CSV, WORKLOAD_THRESHOLDS

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = CLASSIFIER_FEATURE_COLS   # from schema.py
TARGET_COL   = "task_priority"

CLASS_NAMES  = ["Low", "Medium", "High", "Critical"]


# ── Dataset adapters ──────────────────────────────────────────────────────────

def _rule_based_label(cpu: float, ram: float) -> str:
    """Auto-labels a row using resource thresholds for live data retraining."""
    t = WORKLOAD_THRESHOLDS
    if cpu > t["Critical"]["cpu"] or ram > t["Critical"]["ram"]:
        return "Critical"
    elif cpu > t["High"]["cpu"] or ram > t["High"]["ram"]:
        return "High"
    elif cpu > t["Medium"]["cpu"] or ram > t["Medium"]["ram"]:
        return "Medium"
    return "Low"


def _prepare_live_dataset(path: str) -> Optional[pd.DataFrame]:
    """Prepares live_metrics.csv for classifier training by auto-labeling."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    # Normalize column names
    rename_map = {
        "CPUUsage": "cpu_pct", "CPU_Usage": "cpu_pct",
        "RAMUsage": "ram_pct", "Memory_Usage": "ram_pct",
        "Bandwidth": "bandwidth_kbps", "Packet_Rate": "packet_rate_pps",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Auto-label if no task_priority column or all same value
    if "task_priority" not in df.columns or df["task_priority"].nunique() <= 1:
        import numpy as np
        cpu = df["cpu_pct"].fillna(0).to_numpy(dtype=float) if "cpu_pct" in df.columns \
              else np.zeros(len(df), dtype=float)
        ram = df["ram_pct"].fillna(0).to_numpy(dtype=float) if "ram_pct" in df.columns \
              else np.zeros(len(df), dtype=float)
        t = WORKLOAD_THRESHOLDS
        labels = np.where(
            (cpu > t["Critical"]["cpu"]) | (ram > t["Critical"]["ram"]), "Critical",
            np.where(
                (cpu > t["High"]["cpu"]) | (ram > t["High"]["ram"]), "High",
                np.where(
                    (cpu > t["Medium"]["cpu"]) | (ram > t["Medium"]["ram"]),
                    "Medium", "Low"
                )
            )
        )
        df["task_priority"] = list(labels)

    # Fill missing engineered features with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df if len(df) >= 50 else None


def _prepare_kaggle_dataset(path: str) -> Optional[pd.DataFrame]:
    """
    Adapts Kaggle security dataset for classifier training.
    Maps old features to new unified schema and drops dead features.
    Uses Task_Priority as the label.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    rename_map = {
        "CPU_Usage":    "cpu_pct",
        "Memory_Usage": "ram_pct",
        "Bandwidth":    "bandwidth_kbps",
        "Packet_Rate":  "packet_rate_pps",
        "Task_Priority":"task_priority",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Fill all new features with 0 (Kaggle set has no process_count or rolling stats)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    if "task_priority" not in df.columns:
        return None

    return df


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: Optional[str] = None) -> dict:
    """
    Trains Random Forest classifier with real observable features only.
    Falls back to adapted Kaggle dataset if live data is insufficient.
    """
    df = None
    source = "none"

    if dataset_path:
        df = _prepare_live_dataset(dataset_path)
        source = "provided"
    if df is None:
        df = _prepare_live_dataset(LIVE_CSV)
        source = "live_metrics.csv"
    if df is None:
        df = _prepare_kaggle_dataset(KAGGLE_SECURITY_CSV)
        source = "kaggle_security (adapted)"
    if df is None:
        raise FileNotFoundError(
            "No usable training dataset found. Run collect.py first."
        )

    print(f"[WorkloadClassifier] Dataset source : {source}")
    print(f"[WorkloadClassifier] Shape          : {df.shape}")
    print(f"[WorkloadClassifier] Label distribution:")
    print(df["task_priority"].value_counts().to_dict())

    # Encode target
    target_encoder = LabelEncoder()
    target_encoder.fit(CLASS_NAMES)   # ensure fixed class order
    y = target_encoder.transform(df["task_priority"].astype(str))

    X = df[FEATURE_COLS].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[WorkloadClassifier] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = None,
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred      = model.predict(X_test)
    class_names = list(target_encoder.classes_)
    print("\n" + "=" * 60)
    print("  WorkloadClassifier Evaluation Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Feature importances
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    print("[WorkloadClassifier] Feature Importances:")
    for feat, imp in ranked:
        bar = "#" * int(imp * 40)
        print(f"  {feat:22s} {bar} {imp:.4f}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,          os.path.join(MODELS_DIR, "classifier_model.pkl"))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, "classifier_target_encoder.pkl"))
    joblib.dump(scaler,         os.path.join(MODELS_DIR, "classifier_scaler.pkl"))
    print("\n[WorkloadClassifier] Models saved OK")

    return {"source": source, "classes": class_names}


# ── Inference ──────────────────────────────────────────────────────────────────

class WorkloadClassifier:
    """
    Loads trained Random Forest and classifies live system state.

    Usage:
        classifier = WorkloadClassifier()
        classifier.load()
        result = classifier.predict(snap.to_classifier_input())
        # -> {"workload": "Low", "confidence": 0.87, "probabilities": {...}}
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir     = models_dir
        self.model:         Optional[RandomForestClassifier] = None
        self.target_encoder:Optional[LabelEncoder]           = None
        self.scaler:        Optional[StandardScaler]          = None
        self.loaded: bool   = False

    def load(self):
        self.model          = joblib.load(os.path.join(self.models_dir, "classifier_model.pkl"))
        self.target_encoder = joblib.load(os.path.join(self.models_dir, "classifier_target_encoder.pkl"))
        self.scaler         = joblib.load(os.path.join(self.models_dir, "classifier_scaler.pkl"))
        self.loaded = True
        print("[WorkloadClassifier] Model + encoders + scaler loaded OK")

    def predict(self, feature_dict: dict) -> dict:
        """
        Classifies system state.

        Args:
            feature_dict: output of SystemSnapshot.to_classifier_input()

        Returns:
            {"workload": str, "confidence": float, "probabilities": dict}
        """
        if not self.loaded:
            raise RuntimeError("Call .load() before .predict()")

        # Narrow optional attributes for static type checkers (pylance)
        assert self.scaler is not None and self.model is not None and self.target_encoder is not None, \
            "Model, encoder or scaler not loaded"

        row = [float(feature_dict.get(col, 0.0)) for col in FEATURE_COLS]
        X = np.array(row, dtype=np.float32).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        pred_idx = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]
        class_name = self.target_encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(proba[pred_idx]), 4)

        probabilities = {
            self.target_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "workload":      class_name,
            "confidence":    confidence,
            "probabilities": probabilities,
        }


# ── Entry (train mode) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    dataset = _sys.argv[1] if len(_sys.argv) > 1 else None

    print("=" * 60)
    print("  SIT-Py | WorkloadClassifier v2 | Training Mode")
    print("=" * 60)

    result = train(dataset_path=dataset)

    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)
    classifier = WorkloadClassifier()
    classifier.load()

    idle_snap = {
        "cpu_pct": 8.0, "ram_pct": 55.0,
        "bandwidth_kbps": 100.0, "packet_rate_pps": 40.0,
        "process_count": 290.0,
        "cpu_rolling_mean": 9.0, "ram_rolling_mean": 55.5,
        "bw_spike": 1.0,
    }
    heavy_snap = {
        "cpu_pct": 85.0, "ram_pct": 88.0,
        "bandwidth_kbps": 8000.0, "packet_rate_pps": 2000.0,
        "process_count": 450.0,
        "cpu_rolling_mean": 75.0, "ram_rolling_mean": 82.0,
        "bw_spike": 8.5,
    }

    r1 = classifier.predict(idle_snap)
    r2 = classifier.predict(heavy_snap)
    print(f"\n  Idle  -> workload: {r1['workload']} | confidence: {r1['confidence']:.1%}")
    print(f"  Heavy -> workload: {r2['workload']} | confidence: {r2['confidence']:.1%}")