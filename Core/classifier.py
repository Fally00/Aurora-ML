import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Dict, Any

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset", "dataset.csv")

# --- Robust fallback: try common alternative locations (repo uses top-level `Data/`)
if not os.path.exists(DATASET_PATH):
    alt = os.path.join(os.path.dirname(__file__), "..", "Data", "dataset.csv")
    if os.path.exists(alt):
        DATASET_PATH = alt
    else:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt2 = os.path.join(repo_root, "Data", "dataset.csv")
        if os.path.exists(alt2):
            DATASET_PATH = alt2

# features used to classify workload type
FEATURE_COLS     = [
    "CPU_Usage", "Memory_Usage", "Bandwidth",
    "Packet_Rate", "Failed_Logins", "Malware_Alerts", "Intrusion_Alerts"
]

# categorical feature to encode
CATEGORICAL_COL  = "Traffic_Type"

# target — what kind of workload is happening
TARGET_COL       = "Task_Priority"

# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH):
    """
    Trains Random Forest classifier to predict Task_Priority
    (Low / Medium / High / Critical) from system + network metrics.
    Saves model + encoders + scaler to /models.
    """
    print("[WorkloadClassifier] Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"[WorkloadClassifier] Shape: {df.shape}")
    print(f"[WorkloadClassifier] Workload distribution:\n{df[TARGET_COL].value_counts().to_dict()}")

    # ── Encode Traffic_Type ──
    traffic_encoder = LabelEncoder()
    vals = df[CATEGORICAL_COL].astype(str)
    transformed = traffic_encoder.fit_transform(vals)
    df[CATEGORICAL_COL] = pd.Series(transformed, index=df.index)

    # ── Encode target label ──
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL].astype(str))
    print(f"[WorkloadClassifier] Classes: {list(target_encoder.classes_)}")

    # ── Build feature matrix ──
    all_features = FEATURE_COLS + [CATEGORICAL_COL]
    X = df[all_features].values

    # ── Scale ──
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train Random Forest ──
    print("[WorkloadClassifier] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth    = None,    # let trees grow fully
        random_state = 42,
        n_jobs       = -1       # all cores
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred       = model.predict(X_test)
    class_names  = list(target_encoder.classes_)

    print("\n[WorkloadClassifier] ── Evaluation Report ──")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    header = "".join(f"{c:>10}" for c in class_names)
    print(f"{'':12}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"  {class_names[i]:10}{row_str}")

    # ── Feature Importance ──
    all_feature_names = FEATURE_COLS + [CATEGORICAL_COL]
    importances = model.feature_importances_
    ranked = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\n[WorkloadClassifier] Feature Importances:")
    for feat, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"  {feat:22s} {bar} {imp:.4f}")

    # ── Save ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,          os.path.join(MODELS_DIR, "classifier_model.pkl"))
    joblib.dump(traffic_encoder,os.path.join(MODELS_DIR, "classifier_traffic_encoder.pkl"))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, "classifier_target_encoder.pkl"))
    joblib.dump(scaler,         os.path.join(MODELS_DIR, "classifier_scaler.pkl"))

    print("\n[WorkloadClassifier] Models saved ✅")
    return model, traffic_encoder, target_encoder, scaler


# ── Inference ──────────────────────────────────────────────────────────────────

class WorkloadClassifier:
    """
    Loads trained Random Forest and classifies live system snapshots
    into workload priority levels: Low / Medium / High / Critical.

    Usage:
        classifier = WorkloadClassifier()
        classifier.load()

        snapshot = {
            "CPU_Usage": 85.0, "Memory_Usage": 78.0, "Bandwidth": 800.0,
            "Packet_Rate": 3500.0, "Failed_Logins": 0, "Malware_Alerts": 0,
            "Intrusion_Alerts": 0, "Traffic_Type": "HTTP"
        }
        result = classifier.predict(snapshot)
        # result → {"workload": "High", "confidence": 0.91, "probabilities": {...}}
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir      = models_dir
        self.model           = None
        self.traffic_encoder = None
        self.target_encoder  = None
        self.scaler          = None
        self.loaded          = False

    def load(self):
        """Loads model + all encoders + scaler from disk."""
        self.model           = joblib.load(os.path.join(self.models_dir, "classifier_model.pkl"))
        self.traffic_encoder = joblib.load(os.path.join(self.models_dir, "classifier_traffic_encoder.pkl"))
        self.target_encoder  = joblib.load(os.path.join(self.models_dir, "classifier_target_encoder.pkl"))
        self.scaler          = joblib.load(os.path.join(self.models_dir, "classifier_scaler.pkl"))
        self.loaded          = True
        print("[WorkloadClassifier] Loaded model + encoders + scaler ✅")

    def predict(self, snapshot: dict) -> dict:
        """
        Classifies a single snapshot into a workload priority.

        Returns dict with:
          - workload:      "Low" / "Medium" / "High" / "Critical"
          - confidence:    probability of predicted class (0.0 - 1.0)
          - probabilities: full class probability breakdown
        """
        if not self.loaded:
            # try to auto-load for convenience
            try:
                self.load()
            except Exception as e:
                raise RuntimeError("Models not loaded. Call .load() before .predict()") from e

        # runtime guards
        if self.traffic_encoder is None:
            raise RuntimeError("Traffic encoder not loaded. Call .load() before .predict()")
        if self.target_encoder is None:
            raise RuntimeError("Target encoder not loaded. Call .load() before .predict()")
        if self.scaler is None or not hasattr(self.scaler, "transform"):
            raise RuntimeError("Scaler missing or invalid. Call .load() before .predict()")
        if self.model is None or not hasattr(self.model, "predict"):
            raise RuntimeError("Model missing or invalid. Call .load() before .predict()")

        df = pd.DataFrame([snapshot])

        # encode Traffic_Type — handle unseen labels gracefully
        vals = df[CATEGORICAL_COL].astype(str)
        known = set(getattr(self.traffic_encoder, "classes_", []))
        default_class = getattr(self.traffic_encoder, "classes_", [None])[0]
        mapped = vals.apply(lambda x, d=default_class: x if x in known else d)
        transformed = self.traffic_encoder.transform(mapped)
        df[CATEGORICAL_COL] = pd.Series(transformed, index=df.index)

        all_features = FEATURE_COLS + [CATEGORICAL_COL]
        X = df[all_features].values
        X_scaled = self.scaler.transform(X)

        pred_idx = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]
        class_name = self.target_encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(proba[pred_idx]), 4)

        # full probability breakdown
        probabilities = {
            self.target_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "workload"     : class_name,
            "confidence"   : confidence,
            "probabilities": probabilities
        }


# ── Entry (train mode) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH

    print("=" * 60)
    print("  SIT-Py | WorkloadClassifier | Training Mode")
    print("=" * 60)

    train(dataset_path=dataset)

    print("\n── Quick Inference Test ──")
    classifier = WorkloadClassifier()
    classifier.load()

    # low load scenario
    low_snap = {
        "CPU_Usage": 15.0, "Memory_Usage": 20.0, "Bandwidth": 100.0,
        "Packet_Rate": 500.0, "Failed_Logins": 0, "Malware_Alerts": 0,
        "Intrusion_Alerts": 0, "Traffic_Type": "HTTP"
    }

    # critical load scenario
    critical_snap = {
        "CPU_Usage": 95.0, "Memory_Usage": 90.0, "Bandwidth": 9000.0,
        "Packet_Rate": 8000.0, "Failed_Logins": 0, "Malware_Alerts": 0,
        "Intrusion_Alerts": 0, "Traffic_Type": "DNS"
    }

    r1 = classifier.predict(low_snap)
    r2 = classifier.predict(critical_snap)

    print(f"\n  Low load snapshot      → workload: {r1['workload']} | confidence: {r1['confidence']}")
    print(f"  Critical load snapshot → workload: {r2['workload']} | confidence: {r2['confidence']}")
    print(f"\n  Full probabilities (low):")
    for k, v in r1['probabilities'].items():
        print(f"    {k:10s}: {v:.4f}")