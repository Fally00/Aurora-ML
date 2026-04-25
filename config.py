"""
Central project configuration for SIT-Py.

All runtime modules should import paths and tunables from here instead of
hard-coding dataset locations or thresholds.
"""

from __future__ import annotations

import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
MODELS_DIR = os.path.join(ROOT_DIR, "Models")
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
