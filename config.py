"""
Central project configuration for SIT-Py.

All runtime modules should import paths and tunables from here instead of
hard-coding dataset locations or thresholds.

Configuration sections:
- Directories: ROOT_DIR, MODELS_DIR, DATA_DIR, DATASETS_DIR, CORE_DIR, PIPELINE_DIR
- Runtime files: LOG_PATH, LIVE_CSV, RETRAIN_LOG
- Base training datasets: ANOMALY_DATASET_CSV, CLASSIFIER_DATASET_CSV, PREDICTOR_DATASET_CSV
- Pipeline tuning: POLL_INTERVAL, WINDOW_SIZE, MAX_HISTORY, MAX_CSV_ROWS
- Retraining: RETRAIN_ROW_THRESHOLD, LIVE_DATA_WEIGHT, LABELED_LOG_WEIGHT, RANDOM_STATE
- ML thresholds: ANOMALY_Z_THRESHOLD, CONTAMINATION, ANOMALY_ALERT_THRESHOLD
- Workload thresholds: WORKLOAD_THRESHOLDS
- Model hyperparameters: ANOMALY_MODEL_PARAMS, CLASSIFIER_MODEL_PARAMS, PREDICTOR_MODEL_PARAMS
"""

from __future__ import annotations

import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
# Keep executable pickle artifacts outside the source checkout. Override this
# path with SITPY_MODELS_DIR when deploying or sharing a model directory.
MODELS_DIR = os.path.abspath(
    os.path.expanduser(
        os.environ.get("SITPY_MODELS_DIR", os.path.join("~", ".sit-py", "models"))
    )
)
if getattr(sys, "frozen", False):
    # PyInstaller one-file extracts code to a temporary directory. Runtime data
    # must live in a stable, user-writable location instead of that directory.
    DATA_DIR = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Aurora", "Data"
    )
else:
    DATA_DIR = os.path.join(ROOT_DIR, "Data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
CORE_DIR = os.path.join(ROOT_DIR, "Core")
PIPELINE_DIR = os.path.join(ROOT_DIR, "pipeline")

# Runtime files
LOG_PATH = os.path.join(DATA_DIR, "sit_log.csv")
LIVE_CSV = os.path.join(DATA_DIR, "live_metrics.csv")
RETRAIN_LOG = os.path.join(DATA_DIR, "retrain_log.csv")

# Base training datasets
ANOMALY_DATASET_CSV = os.path.join(DATASETS_DIR, "anomaly_FINAL.csv")
CLASSIFIER_DATASET_CSV = os.path.join(DATASETS_DIR, "classifier_FINAL.csv")
PREDICTOR_DATASET_CSV = os.path.join(DATASETS_DIR, "predictor_FINAL.csv")

# Pipeline tuning
POLL_INTERVAL = 2
WINDOW_SIZE = 15
MAX_HISTORY = 60
MAX_CSV_ROWS = 5_000

# Retraining
RETRAIN_ROW_THRESHOLD = 500
LIVE_DATA_WEIGHT = 3
LABELED_LOG_WEIGHT = 2
RANDOM_STATE = 42

# ML thresholds
ANOMALY_Z_THRESHOLD = 2.5
CONTAMINATION = 0.10
ANOMALY_ALERT_THRESHOLD = 0.60

# Workload thresholds used to label live data when no label exists yet.
WORKLOAD_THRESHOLDS = {
    "Critical": {"cpu": 75, "ram": 85},
    "High": {"cpu": 50, "ram": 70},
    "Medium": {"cpu": 20, "ram": 50},
}

# --- Phase 2 Collector v2 ---
# Polling interval in seconds (v2 uses 1s)
POLL_INTERVAL_V2 = 1
ROLLING_WINDOW_V2 = 60
DB_FLUSH_INTERVAL = 1.0
DB_BATCH_SIZE = 10

# Machine identifier: "laptop" | "desktop" | auto-detect via hostname
MACHINE_ID = os.environ.get("AURORA_MACHINE_ID", "laptop")

# Schema version for v2 snapshots
SCHEMA_VERSION = 2

# SQLite database path (WAL mode)
DB_DIR = os.path.join(DATA_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "aurora_metrics.db")

# Data retention (days)
RETENTION_DAYS = 7

# Workload tag file (re-read each cycle)
WORKLOAD_TAG_FILE = os.path.join(DATA_DIR, "workload_tag.txt")

# Valid workload tags
VALID_WORKLOAD_TAGS = {"idle", "gaming", "compile", "browsing", "stress"}

# NVML/LHM graceful degradation
NVML_WARNING_LOGGED = False
LHM_WARNING_LOGGED = False

# --- Model Hyperparameters ---
ANOMALY_MODEL_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "class_weight": "balanced",
}

CLASSIFIER_MODEL_PARAMS = {
    "n_estimators": 250,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "class_weight": "balanced_subsample",
}

PREDICTOR_MODEL_PARAMS = {
    "n_estimators": 250,
    "max_depth": 16,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
