"""
Anomaly detector.

The anomaly model now trains against the dedicated anomaly dataset and keeps
the same runtime output contract used by main.py and the dashboard.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.schema import (
    ANOMALY_CATEGORICAL_COLS,
    ANOMALY_FEATURE_COLS,
    ANOMALY_NUMERIC_COLS,
    ANOMALY_TARGET_COL,
    coerce_numeric_columns,
    ensure_columns,
    normalize_anomaly_label,
    normalize_columns,
)
from config import (
    ANOMALY_ALERT_THRESHOLD,
    ANOMALY_DATASET_CSV,
    MODELS_DIR,
    RANDOM_STATE,
)


FEATURE_COLS = ANOMALY_FEATURE_COLS
NUMERIC_COLS = ANOMALY_NUMERIC_COLS
CATEGORICAL_COLS = ANOMALY_CATEGORICAL_COLS
TARGET_COL = ANOMALY_TARGET_COL
VALID_LABELS = {"Normal", "Alert"}


def _load_training_dataset(path: str) -> Optional[pd.DataFrame]:
    """Load and normalize an anomaly dataset."""
    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = ensure_columns(df, FEATURE_COLS + [TARGET_COL])
    df = coerce_numeric_columns(df, NUMERIC_COLS)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("").astype(str)

    df[TARGET_COL] = df[TARGET_COL].map(normalize_anomaly_label)
    df = df[df[TARGET_COL].isin(VALID_LABELS)]
    return df if len(df) >= 50 else None


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train(dataset_path: str | None = None) -> dict:
    """
    Train the anomaly detector.

    Priority:
      1. Explicit dataset_path
      2. Base anomaly dataset
    """
    source = "none"
    df = None

    if dataset_path:
        df = _load_training_dataset(dataset_path)
        source = dataset_path
    if df is None:
        df = _load_training_dataset(ANOMALY_DATASET_CSV)
        source = ANOMALY_DATASET_CSV
    if df is None:
        raise FileNotFoundError(
            "No anomaly dataset found. Expected one of: "
            f"{dataset_path!r} or {ANOMALY_DATASET_CSV!r}."
        )

    print(f"[AnomalyDetector] Dataset source : {source}")
    print(f"[AnomalyDetector] Shape          : {df.shape}")
    print(f"[AnomalyDetector] Label counts   : {df[TARGET_COL].value_counts().to_dict()}")

    x = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=_stratify_or_none(y),
    )

    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    print("\n" + "=" * 60)
    print("  AnomalyDetector Evaluation Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = float((y_pred == y_test).mean())
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    print(f"[AnomalyDetector] Model saved to {MODELS_DIR}")

    return {
        "source": source,
        "accuracy": round(accuracy, 4),
        "label_counts": df[TARGET_COL].value_counts().to_dict(),
    }


class AnomalyDetector:
    """Load the trained anomaly model and score live telemetry."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model: Optional[Pipeline] = None
        self.loaded = False

    def load(self) -> bool:
        """Load the anomaly model if it exists, else stay in bridge mode."""
        model_path = os.path.join(self.models_dir, "anomaly_model.pkl")
        if not os.path.exists(model_path):
            print("[AnomalyDetector] No anomaly model found, using statistical bridge.")
            self.loaded = False
            return False

        self.model = joblib.load(model_path)
        if not hasattr(self.model, "named_steps") or "model" not in self.model.named_steps:
            print("[AnomalyDetector] Anomaly model is from an older schema; using bridge mode.")
            self.model = None
            self.loaded = False
            return False
        self.loaded = True
        print("[AnomalyDetector] Supervised anomaly model loaded")
        return True

    def _prepare_input_frame(self, feature_dict: dict) -> pd.DataFrame:
        frame = pd.DataFrame([feature_dict])
        frame = normalize_columns(frame)
        frame = ensure_columns(frame, FEATURE_COLS)
        frame = coerce_numeric_columns(frame, NUMERIC_COLS)
        frame[NUMERIC_COLS] = frame[NUMERIC_COLS].fillna(0.0)
        for col in CATEGORICAL_COLS:
            frame[col] = frame[col].fillna("").astype(str)
        return frame

    def predict(self, snap_input: dict) -> dict:
        """Predict whether the current telemetry looks normal or anomalous."""
        if not self.loaded or self.model is None:
            raise RuntimeError(
                "Anomaly model not loaded. Call .load() first or use the statistical bridge."
            )

        frame = self._prepare_input_frame(snap_input)
        classes = list(self.model.named_steps["model"].classes_)
        probabilities = self.model.predict_proba(frame)[0]

        if "Alert" in classes:
            alert_index = classes.index("Alert")
            alert_probability = float(probabilities[alert_index])
        else:
            alert_probability = 0.0

        label = "Alert" if alert_probability >= ANOMALY_ALERT_THRESHOLD else "Normal"
        confidence = alert_probability if label == "Alert" else 1.0 - alert_probability

        return {
            "label": label,
            "anomaly_score": round(alert_probability, 4),
            "is_anomaly": label == "Alert",
            "confidence": round(float(confidence), 4),
            "source": "supervised_rf",
        }

    def predict_batch(self, inputs: list[dict]) -> list[dict]:
        return [self.predict(item) for item in inputs]

    @property
    def mode(self) -> str:
        return "supervised_rf" if self.loaded else "stat_bridge"


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  SIT-Py | AnomalyDetector | Training Mode")
    print("=" * 60)
    result = train(dataset_path=dataset)
    print(f"\n[AnomalyDetector] Accuracy: {result['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)
    detector = AnomalyDetector()
    detector.load()
    print(
        detector.predict(
            {
                "cpu_pct": 12.0,
                "ram_pct": 58.0,
                "bandwidth_kbps": 48.0,
                "packet_rate_pps": 0.8,
                "protocol": "",
                "attack_type": "",
                "severity": "",
                "threat_type": "",
            }
        )
    )
