"""
config.py — SIT-Py Centralized Configuration
Single source of truth for all paths, constants, and tunable parameters.
All modules import from here — no more scattered os.path.join chains.
"""
import os

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Directories ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT_DIR, "Models")
DATA_DIR   = os.path.join(ROOT_DIR, "Data")
CORE_DIR   = os.path.join(ROOT_DIR, "Core")

# ── Data Files ────────────────────────────────────────────────────────────────
LOG_PATH              = os.path.join(DATA_DIR, "sit_log.csv")
LIVE_CSV              = os.path.join(DATA_DIR, "live_metrics.csv")
KAGGLE_SECURITY_CSV   = os.path.join(DATA_DIR, "dataset.csv")
KAGGLE_MOTHERBOARD_CSV= os.path.join(DATA_DIR,
                            "Laptop_Motherboard_Health_Monitoring_Dataset.csv")

# ── Pipeline Tuning ───────────────────────────────────────────────────────────
POLL_INTERVAL   = 2        # seconds between metric snapshots
WINDOW_SIZE     = 15       # rolling context window for predictor + feature eng.
MAX_HISTORY     = 60       # dashboard history ring-buffer size
MAX_CSV_ROWS    = 5_000    # cap on live_metrics.csv before rotation

# ── ML Thresholds ─────────────────────────────────────────────────────────────
ANOMALY_Z_THRESHOLD   = 2.5     # std-deviations for statistical anomaly bridge
RETRAIN_ROW_THRESHOLD = 500     # min live rows before retraining is triggered
CONTAMINATION         = 0.10    # Isolation Forest contamination (lower than 0.2
                                #   since real system anomalies are rarer)

# ── Workload Thresholds (rule-based labels for live training) ─────────────────
WORKLOAD_THRESHOLDS = {
    "Critical": {"cpu": 75, "ram": 85},
    "High":     {"cpu": 50, "ram": 70},
    "Medium":   {"cpu": 20, "ram": 50},
    # else → Low
}
