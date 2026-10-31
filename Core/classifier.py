"""
Workload classifier.

This module now uses the dedicated classifier dataset by default and can also
consume normalized live/runtime data prepared by the retraining pipeline.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.baselines import evaluate_baselines, evaluate_model_vs_baselines
from Core.schema import (
    CLASSIFIER_FEATURE_COLS,
    CLASSIFIER_TARGET_COL,
    coerce_numeric_columns,
    ensure_columns,
    normalize_columns,
    normalize_workload_label,
)
from Core.base_model import BaseModel
from config import (
    CLASSIFIER_DATASET_CSV,
    CLASSIFIER_MODEL_PARAMS,
    LIVE_CSV,
    MODELS_DIR,
    RANDOM_STATE,
    WORKLOAD_THRESHOLDS,
)


FEATURE_COLS = CLASSIFIER_FEATURE_COLS
TARGET_COL = CLASSIFIER_TARGET_COL
CLASS_NAMES = ["Low", "Medium", "High", "Critical"]


def _rule_based_label(cpu: float, ram: float) -> str:
    """Infer a workload label from CPU and RAM thresholds."""
    thresholds = WORKLOAD_THRESHOLDS
    if cpu > thresholds["Critical"]["cpu"] or ram > thresholds["Critical"]["ram"]:
        return "Critical"
    if cpu > thresholds["High"]["cpu"] or ram > thresholds["High"]["ram"]:
        return "High"
    if cpu > thresholds["Medium"]["cpu"] or ram > thresholds["Medium"]["ram"]:
        return "Medium"
    return "Low"


def _load_training_dataset(path: str) -> Optional[pd.DataFrame]:
    """Load and normalize a classifier dataset."""
    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = normalize_columns(df)

    required_cols = FEATURE_COLS + [TARGET_COL]
    df = ensure_columns(df, required_cols)
    df = coerce_numeric_columns(df, FEATURE_COLS)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map(normalize_workload_label)

    if TARGET_COL not in df.columns or df[TARGET_COL].isna().all():
        df[TARGET_COL] = [
            _rule_based_label(float(cpu or 0.0), float(ram or 0.0))
            for cpu, ram in zip(df["cpu_pct"].fillna(0.0), df["ram_pct"].fillna(0.0))
        ]

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL].isin(CLASS_NAMES)]
    return df if len(df) >= 50 else None


def _stratify_or_none(labels: np.ndarray) -> np.ndarray | None:
    """Stratify only when each class has enough rows."""
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 2 or counts.min() < 2:
        return None
    return labels


def train(dataset_path: str | None = None) -> dict:
    """
    Train the workload classifier.

    Priority:
      1. Explicit dataset_path
      2. Base classifier dataset
      3. Unified live_metrics.csv
    """
    source = "none"
    df = None

    if dataset_path:
        df = _load_training_dataset(dataset_path)
        source = dataset_path
    if df is None:
        df = _load_training_dataset(CLASSIFIER_DATASET_CSV)
        source = CLASSIFIER_DATASET_CSV
    if df is None:
        df = _load_training_dataset(LIVE_CSV)
        source = LIVE_CSV
    if df is None:
        raise FileNotFoundError(
            "No classifier dataset found. Expected one of: "
            f"{dataset_path!r}, {CLASSIFIER_DATASET_CSV!r}, or {LIVE_CSV!r}."
        )

    print(f"[WorkloadClassifier] Dataset source : {source}")
    print(f"[WorkloadClassifier] Shape          : {df.shape}")
    print(f"[WorkloadClassifier] Label counts   : {df[TARGET_COL].value_counts().to_dict()}")

    target_encoder = LabelEncoder()
    target_encoder.fit(CLASS_NAMES)
    y = target_encoder.transform(df[TARGET_COL].astype(str))
    x = df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Chronological split by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        x_train = x_scaled[:split_idx]
        y_train = y[:split_idx]
        x_test = x_scaled[split_idx:]
        y_test = y[split_idx:]
        print(f"[WorkloadClassifier] Chronological split: train={len(x_train)}, test={len(x_test)}")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=_stratify_or_none(y),
        )
        print(f"[WorkloadClassifier] Random split: train={len(x_train)}, test={len(x_test)}")

    model = RandomForestClassifier(**CLASSIFIER_MODEL_PARAMS)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    class_names = list(target_encoder.classes_)
    print("\n" + "=" * 60)
    print("  WorkloadClassifier Evaluation Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances), key=lambda item: -item[1])
    print("[WorkloadClassifier] Feature importances:")
    for feat, importance in ranked:
        print(f"  {feat:22s} {importance:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save and hash model
    model_path = os.path.join(MODELS_DIR, "classifier_model.pkl")
    joblib.dump(model, model_path)
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    hash_path = model_path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(model_hash)
    # Save and hash target encoder
    encoder_path = os.path.join(MODELS_DIR, "classifier_target_encoder.pkl")
    joblib.dump(target_encoder, encoder_path)
    with open(encoder_path, "rb") as f:
        encoder_hash = hashlib.sha256(f.read()).hexdigest()
    hash_path = encoder_path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(encoder_hash)
    # Save and hash scaler
    scaler_path = os.path.join(MODELS_DIR, "classifier_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    with open(scaler_path, "rb") as f:
        scaler_hash = hashlib.sha256(f.read()).hexdigest()
    hash_path = scaler_path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(scaler_hash)
    print("\n[WorkloadClassifier] Model artifacts saved")

    accuracy = float((y_pred == y_test).mean())
    return {
        "source": source,
        "classes": class_names,
        "accuracy": round(accuracy, 4),
    }


class WorkloadClassifier(BaseModel):
    """Load the trained workload classifier and score live telemetry."""

    model_files = [
        "classifier_model.pkl",
        "classifier_target_encoder.pkl",
        "classifier_scaler.pkl",
    ]

    def __init__(self, models_dir: str = MODELS_DIR):
        super().__init__(models_dir)
        self.model: Optional[RandomForestClassifier] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None

    def load(self) -> bool:
        """Load the classifier model and artifacts."""
        if not super().load():
            return False

        self.model = self.get_artifact("classifier_model.pkl")
        self.target_encoder = self.get_artifact("classifier_target_encoder.pkl")
        self.scaler = self.get_artifact("classifier_scaler.pkl")

        if any(artifact is None for artifact in [self.model, self.target_encoder, self.scaler]):
            print("[WorkloadClassifier] One or more model artifacts not found.")
            self.loaded = False
            return False

        expected_features = len(FEATURE_COLS)
        model_features = getattr(self.model, "n_features_in_", None)
        scaler_features = getattr(self.scaler, "n_features_in_", None)
        if model_features != expected_features or scaler_features != expected_features:
            print(
                "[WorkloadClassifier] Classifier artifacts are from an older schema. "
                "Retrain with `python Core/classifier.py` or `python pipeline/retrain.py --force`."
            )
            self.loaded = False
            return False

        self.loaded = True
        print("[WorkloadClassifier] Model loaded")
        return True

    def predict(self, feature_dict: dict[str, float]) -> dict:
        """Predict Low/Medium/High/Critical workload from live features."""
        if not self.loaded:
            raise RuntimeError("Call .load() before .predict()")
        assert self.model is not None and self.scaler is not None and self.target_encoder is not None

        row = [float(feature_dict.get(col, 0.0)) for col in FEATURE_COLS]
        x = np.array(row, dtype=np.float32).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        pred_idx = int(self.model.predict(x_scaled)[0])
        probabilities = self.model.predict_proba(x_scaled)[0]
        workload = str(self.target_encoder.inverse_transform([pred_idx])[0])
        confidence = round(float(probabilities[pred_idx]), 4)

        per_class = {
            str(self.target_encoder.inverse_transform([idx])[0]): round(float(prob), 4)
            for idx, prob in enumerate(probabilities)
        }

        return {
            "workload": workload,
            "confidence": confidence,
            "probabilities": per_class,
        }


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  SIT-Py | WorkloadClassifier | Training Mode")
    print("=" * 60)
    result = train(dataset_path=dataset)
    print(f"\n[WorkloadClassifier] Accuracy: {result['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)
    classifier = WorkloadClassifier()
    classifier.load()
    print(
        classifier.predict(
            {
                "cpu_pct": 42.0,
                "ram_pct": 68.0,
                "bandwidth_kbps": 28.5,
                "packet_rate_pps": 0.7,
            }
        )
    )
