import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 5       # how many past readings to use as features
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
# Default (legacy) expected dataset location
DATASET_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "dataset",
                             "Laptop_Motherboard_Health_Monitoring_Dataset.csv")

# --- Robust fallback: try common alternative locations (repo uses `Data/` at top-level)
if not os.path.exists(DATASET_PATH):
    alt = os.path.join(os.path.dirname(__file__), "..", "Data",
                       "Laptop_Motherboard_Health_Monitoring_Dataset.csv")
    if os.path.exists(alt):
        DATASET_PATH = alt
    else:
        # last resort: try repository-root `Data` folder (normalized)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt2 = os.path.join(repo_root, "Data", "Laptop_Motherboard_Health_Monitoring_Dataset.csv")
        if os.path.exists(alt2):
            DATASET_PATH = alt2

# columns we use as input features
FEATURE_COLS  = ["CPUUsage", "RAMUsage", "Temperature", "DiskUsage", "FanSpeed"]

# columns we want to predict (one model per target)
TARGET_COLS   = ["CPUUsage", "RAMUsage", "DiskUsage"]

# ── Sliding Window Builder ─────────────────────────────────────────────────────

def build_windows(df: pd.DataFrame, window: int):
    """
    Turns a time-series DataFrame into (X, y) pairs using a sliding window.

    For each position i (starting at `window`):
      X[i] = flattened values of rows [i-window : i]   → shape (window * n_features,)
      y[i] = values of row i for each target column

    Example with window=3, 2 features, targets=[A]:
      rows: [r0, r1, r2, r3, r4]
      X[0] = [r0_f1, r0_f2, r1_f1, r1_f2, r2_f1, r2_f2]  →  y[0] = r3_A
      X[1] = [r1_f1, r1_f2, r2_f1, r2_f2, r3_f1, r3_f2]  →  y[1] = r4_A
    """
    feature_data = df[FEATURE_COLS].values
    target_data  = df[TARGET_COLS].values

    X_list, y_list = [], []

    for i in range(window, len(df)):
        window_slice = feature_data[i - window : i]   # shape: (window, n_features)
        X_list.append(window_slice.flatten())          # flatten → 1D
        y_list.append(target_data[i])

    return np.array(X_list), np.array(y_list)


# ── Training ───────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH, window: int = WINDOW_SIZE):
    """
    Trains one LinearRegression model per target column.
    Saves models + scaler to /models directory.
    Returns dict of evaluation metrics.
    """
    print("[Predictor] Loading dataset...")
    df = pd.read_csv(dataset_path)

    # keep only the columns we need, no duplicates
    all_cols = list(dict.fromkeys(FEATURE_COLS + TARGET_COLS))  # preserves order, dedupes
    df = df[all_cols].dropna()

    print(f"[Predictor] Dataset shape: {df.shape}")
    print(f"[Predictor] Building sliding windows (size={window})...")

    X, y = build_windows(df, window)
    print(f"[Predictor] Window dataset → X: {X.shape}, y: {y.shape}")

    # scale features
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics = {}

    # one model per target
    for idx, target in enumerate(TARGET_COLS):
        print(f"[Predictor] Training model for → {target}")
        model = LinearRegression()
        model.fit(X_train, y_train[:, idx])

        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test[:, idx], y_pred)
        r2     = r2_score(y_test[:, idx], y_pred)

        metrics[target] = {"MAE": round(mae, 4), "R2": round(r2, 4)}
        print(f"  → MAE: {mae:.4f} | R²: {r2:.4f}")

        model_path = os.path.join(MODELS_DIR, f"predictor_{target.lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"  → Saved: {model_path}")

    # save scaler (shared across all predictor models)
    scaler_path = os.path.join(MODELS_DIR, "predictor_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[Predictor] Scaler saved: {scaler_path}")
    print("[Predictor] Training complete ✅")

    return metrics


# ── Inference ──────────────────────────────────────────────────────────────────

class ResourcePredictor:
    """
    Loads trained models and runs prediction on a live window of readings.

    Usage:
        predictor = ResourcePredictor()
        predictor.load()

        # feed it a list of last WINDOW_SIZE dicts (from collect.py output)
        window_data = [
            {"CPUUsage": 45.2, "RAMUsage": 60.1, "Temperature": 55.0, "DiskUsage": 30.0, "FanSpeed": 2800},
            ...  (5 total)
        ]
        result = predictor.predict(window_data)
        # result → {"CPUUsage": 47.3, "RAMUsage": 61.0, "DiskUsage": 30.1}
    """

    def __init__(self, models_dir: str = MODELS_DIR, window: int = WINDOW_SIZE):
        self.models_dir = models_dir
        self.window     = window
        self.models     = {}
        self.scaler     = None
        self.loaded     = False

    def load(self):
        """Loads all models and scaler from disk."""
        scaler_path = os.path.join(self.models_dir, "predictor_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Scaler not found. Run predictor.train() first."
            )

        self.scaler = joblib.load(scaler_path)

        for target in TARGET_COLS:
            model_path = os.path.join(self.models_dir, f"predictor_{target.lower()}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.models[target] = joblib.load(model_path)

        self.loaded = True
        print(f"[Predictor] Loaded {len(self.models)} models + scaler ✅")

    def predict(self, window_data: list) -> dict:
        """
        Predicts next values for CPU, RAM, Disk.

        Args:
            window_data: list of dicts, length must equal WINDOW_SIZE.
                         Each dict must have keys: CPUUsage, RAMUsage,
                         Temperature, DiskUsage, FanSpeed.

        Returns:
            dict with predicted next values for CPUUsage, RAMUsage, DiskUsage.
        """
        if not self.loaded:
            # Try to auto-load for convenience; if it fails, raise a clear error.
            try:
                self.load()
            except Exception as e:
                raise RuntimeError("Models/scaler not loaded. Call .load() before .predict()") from e

        # Ensure scaler is present and valid before calling transform
        if self.scaler is None or not hasattr(self.scaler, "transform"):
            scaler_path = os.path.join(self.models_dir, "predictor_scaler.pkl")
            raise RuntimeError(
                f"Scaler missing or invalid. Expected scaler at: {scaler_path}. "
                "Call .load() or run training to create the scaler."
            )

        if len(window_data) != self.window:
            raise ValueError(
                f"window_data must have exactly {self.window} entries, "
                f"got {len(window_data)}"
            )

        # build flat feature vector
        feature_vector = []
        for row in window_data:
            for col in FEATURE_COLS:
                feature_vector.append(float(row.get(col, 0.0)))

        X = np.array(feature_vector).reshape(1, -1)

        # Sanity-check scaler input size to provide a clearer error message
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"Feature size mismatch: scaler expects {expected} features but input has {X.shape[1]} "
                f"(window={self.window}, features_per_row={len(FEATURE_COLS)})"
            )

        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}") from e

        predictions = {}
        for target, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[target] = round(float(pred[0]), 2)
            except Exception as e:
                raise RuntimeError(f"Model predict failed for target {target}: {e}") from e

        return predictions

    def predict_from_csv(self, csv_path: str) -> dict:
        """
        Reads last WINDOW_SIZE rows from a live_metrics.csv and predicts.
        Convenience method for real-time use with collect.py output.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # map collector column names → motherboard column names
        col_map = {
            "CPU_Usage"    : "CPUUsage",
            "Memory_Usage" : "RAMUsage",
            "DiskUsage"    : "DiskUsage",
        }
        df = df.rename(columns=col_map)

        # fill missing columns with 0 if not present
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        if len(df) < self.window:
            raise ValueError(
                f"Not enough rows in CSV: need {self.window}, got {len(df)}"
            )

        last_window = df[FEATURE_COLS].tail(self.window).to_dict(orient="records")
        return self.predict(last_window)


# ── Entry (train mode) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH

    print("=" * 60)
    print("  SIT-Py | ResourcePredictor | Training Mode")
    print("=" * 60)

    metrics = train(dataset_path=dataset)

    print("\n── Evaluation Summary ──")
    for target, m in metrics.items():
        print(f"  {target:12s} → MAE: {m['MAE']} | R²: {m['R2']}")

    print("\n── Quick Inference Test ──")
    predictor = ResourcePredictor()
    predictor.load()

    dummy_window = [
        {"CPUUsage": 50.0, "RAMUsage": 60.0, "Temperature": 55.0,
         "DiskUsage": 30.0, "FanSpeed": 2800}
        for _ in range(WINDOW_SIZE)
    ]
    result = predictor.predict(dummy_window)
    print(f"  Input:  CPU=50%, RAM=60%, Disk=30%")
    print(f"  Predicted next → {result}")