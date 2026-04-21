import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Dict, Tuple, Any

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

# numeric features fed directly to the model
NUMERIC_COLS     = [
    "CPU_Usage", "Memory_Usage", "Bandwidth",
    "Packet_Rate", "Failed_Logins", "Malware_Alerts", "Intrusion_Alerts"
]

# categorical features that need encoding
CATEGORICAL_COLS = ["Task_Priority", "Traffic_Type"]

# all features in order (numeric first, then encoded categoricals)
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# isolation forest contamination = ratio of anomalies in dataset (~20%)
CONTAMINATION = 0.2

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, encoders: Optional[Dict[str, LabelEncoder]] = None, fit: bool = True) -> Tuple[np.ndarray, Dict[str, LabelEncoder], Any]:
    """
    Encodes categorical columns and scales features.

    Args:
        df:       raw dataframe with ALL_FEATURE_COLS present
        encoders: dict of {col_name: LabelEncoder} — pass existing ones at inference
        fit:      if True, fits new encoders (training mode)
                  if False, uses provided encoders (inference mode)

    Returns:
        X_scaled:  numpy array ready for model
        encoders:  dict of fitted LabelEncoders (reuse at inference)
        scaler:    fitted StandardScaler (reuse at inference)
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    # encode categoricals
    for col in CATEGORICAL_COLS:
        vals = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            transformed = le.fit_transform(vals)
            df[col] = pd.Series(transformed, index=df.index)
            encoders[col] = le
        else:
            if col not in encoders or encoders[col] is None:
                raise ValueError(f"Missing encoder for column '{col}' in inference mode")
            le = encoders[col]
            # handle unseen labels gracefully → map to most common class (0)
            known = set(getattr(le, "classes_", []))
            mapped = vals.apply(lambda x: x if x in known else le.classes_[0])
            transformed = le.transform(mapped)
            df[col] = pd.Series(transformed, index=df.index)

    X = df[ALL_FEATURE_COLS].values

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        raise ValueError("Pass scaler explicitly at inference — use AnomalyDetector.predict()")

    return X_scaled, encoders, scaler


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH):
    """
    Trains Isolation Forest on dataset.csv.
    Saves model + encoders + scaler to /models.
    Evaluates against the Label column (Normal / Attack).
    """
    print("[AnomalyDetector] Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"[AnomalyDetector] Shape: {df.shape}")
    print(f"[AnomalyDetector] Label distribution:\n{df['Label'].value_counts().to_dict()}")

    # preprocess (fit mode)
    X_scaled, encoders, scaler = preprocess(df, fit=True)

    print(f"[AnomalyDetector] Training Isolation Forest (contamination={CONTAMINATION})...")
    model = IsolationForest(
        n_estimators  = 200,       # more trees = more stable
        contamination = CONTAMINATION,
        random_state  = 42,
        n_jobs        = -1         # use all CPU cores
    )
    model.fit(X_scaled)

    # ── Evaluation ──
    # isolation forest returns: -1 = anomaly, 1 = normal
    raw_preds = model.predict(X_scaled)

    # map to human labels: -1 → "Attack", 1 → "Normal"
    pred_labels = np.where(raw_preds == -1, "Attack", "Normal")
    # ensure numpy arrays of strings for sklearn metrics
    pred_labels = np.array(pred_labels, dtype=object)
    true_labels = df["Label"].astype(str).to_numpy()

    print("\n[AnomalyDetector] ── Evaluation Report ──")
    print(classification_report(true_labels, pred_labels, target_names=["Attack", "Normal"]))

    cm = confusion_matrix(true_labels, pred_labels, labels=["Attack", "Normal"])
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             Attack  Normal")
    print(f"  Attack   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  Normal   {cm[1][0]:6d}  {cm[1][1]:6d}")

    # ── Save ──
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,    os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "anomaly_encoders.pkl"))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, "anomaly_scaler.pkl"))

    print("\n[AnomalyDetector] Models saved ✅")
    return model, encoders, scaler


# ── Inference ──────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Loads trained Isolation Forest and detects anomalies in live snapshots.

    Usage:
        detector = AnomalyDetector()
        detector.load()

        snapshot = {
            "CPU_Usage": 85.0, "Memory_Usage": 90.0, "Bandwidth": 900.0,
            "Packet_Rate": 4000.0, "Failed_Logins": 5, "Malware_Alerts": 1,
            "Intrusion_Alerts": 1, "Task_Priority": "Critical",
            "Traffic_Type": "DNS"
        }
        result = detector.predict(snapshot)
        # result → {"label": "Attack", "anomaly_score": -0.23, "is_anomaly": True}
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model      = None
        self.encoders   = None
        self.scaler     = None
        self.loaded     = False

    def load(self):
        """Loads model, encoders, scaler from disk."""
        self.model    = joblib.load(os.path.join(self.models_dir, "anomaly_model.pkl"))
        self.encoders = joblib.load(os.path.join(self.models_dir, "anomaly_encoders.pkl"))
        self.scaler   = joblib.load(os.path.join(self.models_dir, "anomaly_scaler.pkl"))
        self.loaded   = True
        print("[AnomalyDetector] Loaded model + encoders + scaler ✅")

    def predict(self, snapshot: dict) -> dict:
        """
        Runs anomaly detection on a single snapshot dict.

        Returns dict with:
          - label:         "Normal" or "Attack"
          - anomaly_score: raw isolation forest score (more negative = more anomalous)
          - is_anomaly:    True if Attack
        """
        if not self.loaded:
            raise RuntimeError("Call .load() before .predict()")

        # runtime guards for static type checkers and clearer errors
        if self.encoders is None:
            raise RuntimeError("Encoders missing. Call .load() before .predict()")
        if self.scaler is None or not hasattr(self.scaler, "transform"):
            raise RuntimeError("Scaler missing or invalid. Call .load() before .predict()")
        if self.model is None or not hasattr(self.model, "predict"):
            raise RuntimeError("Model missing or invalid. Call .load() before .predict()")

        df = pd.DataFrame([snapshot])

        # encode categoricals using saved encoders (safe Series assignments)
        for col in CATEGORICAL_COLS:
            le = self.encoders.get(col)
            if le is None:
                raise RuntimeError(f"Encoder for column '{col}' not found. Re-run training or check saved encoders.")
            vals = df[col].astype(str)
            known = set(getattr(le, "classes_", []))
            mapped = vals.apply(lambda x: x if x in known else le.classes_[0])
            transformed = le.transform(mapped)
            df[col] = pd.Series(transformed, index=df.index)

        X = df[ALL_FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)

        raw_pred = self.model.predict(X_scaled)[0]          # -1 or 1
        score    = self.model.score_samples(X_scaled)[0]    # anomaly score

        label = "Attack" if raw_pred == -1 else "Normal"

        return {
            "label"        : label,
            "anomaly_score": round(float(score), 4),
            "is_anomaly"   : raw_pred == -1
        }

    def predict_batch(self, snapshots: list) -> list:
        """Runs predict() on a list of snapshot dicts. Returns list of result dicts."""
        return [self.predict(s) for s in snapshots]


# ── Entry (train mode) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH

    print("=" * 60)
    print("  SIT-Py | AnomalyDetector | Training Mode")
    print("=" * 60)

    train(dataset_path=dataset)

    print("\n── Quick Inference Test ──")
    detector = AnomalyDetector()
    detector.load()

    # test 1: normal-looking snapshot
    normal_snap = {
        "CPU_Usage": 30.0, "Memory_Usage": 25.0, "Bandwidth": 300.0,
        "Packet_Rate": 1500.0, "Failed_Logins": 0, "Malware_Alerts": 0,
        "Intrusion_Alerts": 0, "Task_Priority": "Low", "Traffic_Type": "HTTP"
    }

    # test 2: sus snapshot — high everything
    attack_snap = {
        "CPU_Usage": 95.0, "Memory_Usage": 92.0, "Bandwidth": 9999.0,
        "Packet_Rate": 9000.0, "Failed_Logins": 10, "Malware_Alerts": 3,
        "Intrusion_Alerts": 2, "Task_Priority": "Critical", "Traffic_Type": "DNS"
    }

    r1 = detector.predict(normal_snap)
    r2 = detector.predict(attack_snap)

    print(f"\n  Normal snapshot  → {r1}")
    print(f"  Attack snapshot  → {r2}")