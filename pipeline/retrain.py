"""
pipeline/retrain.py - SIT-Py Retraining Pipeline
==================================================
Converts accumulated sit_log.csv / live_metrics.csv into new model artifacts.

Run manually:
    python pipeline/retrain.py

Or integrate with Windows Task Scheduler / cron for periodic retraining.

The pipeline:
  1. Checks if enough live data exists (RETRAIN_ROW_THRESHOLD)
  2. Optionally merges with Kaggle base datasets (weighted)
  3. Retrains all three models: predictor, anomaly detector, classifier
  4. Logs the retrain event to retrain_log.csv
  5. Reports before/after metric comparison
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    LIVE_CSV, LOG_PATH, MODELS_DIR, DATA_DIR,
    KAGGLE_MOTHERBOARD_CSV, KAGGLE_SECURITY_CSV,
    RETRAIN_ROW_THRESHOLD,
)

RETRAIN_LOG = os.path.join(DATA_DIR, "retrain_log.csv")


# ── Data preparation ───────────────────────────────────────────────────────────

def load_live_data() -> pd.DataFrame:
    """Loads and merges live_metrics.csv + sit_log.csv into one DataFrame."""
    frames = []

    if os.path.exists(LIVE_CSV):
        df = pd.read_csv(LIVE_CSV)
        df["_source"] = "live_metrics"
        frames.append(df)
        print(f"  [data] live_metrics.csv     : {len(df):,} rows")

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df["_source"] = "sit_log"
        frames.append(df)
        print(f"  [data] sit_log.csv          : {len(df):,} rows")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    print(f"  [data] Total after dedup     : {len(combined):,} rows")
    return combined


def merge_with_kaggle(live_df: pd.DataFrame, live_weight: int = 3) -> pd.DataFrame:
    """
    Merges live data (repeated live_weight×) with Kaggle base datasets.
    Live data repetition ensures the model prioritizes real-machine behavior.
    """
    frames = [live_df] * live_weight

    if os.path.exists(KAGGLE_MOTHERBOARD_CSV):
        kdf = pd.read_csv(KAGGLE_MOTHERBOARD_CSV)
        kdf["_source"] = "kaggle_motherboard"
        frames.append(kdf)
        print(f"  [merge] kaggle_motherboard  : {len(kdf):,} rows")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  [merge] Final merged dataset: {len(merged):,} rows "
          f"(live ×{live_weight} + kaggle)")
    return merged


def should_retrain(live_df: pd.DataFrame) -> bool:
    """Returns True if we have enough live data to retrain."""
    n = len(live_df)
    print(f"\n[Retrain] Live data rows: {n} / {RETRAIN_ROW_THRESHOLD} required")
    return n >= RETRAIN_ROW_THRESHOLD


# ── Per-model training ─────────────────────────────────────────────────────────

def retrain_predictor(data_path: str) -> dict:
    from Core.predictor import train as train_predictor
    print("\n" + "=" * 60)
    print("  [Retrain] Predictor")
    print("=" * 60)
    return train_predictor(dataset_path=data_path)


def retrain_anomaly(data_path: str) -> dict:
    from Core.anomaly import train as train_anomaly
    print("\n" + "=" * 60)
    print("  [Retrain] Anomaly Detector")
    print("=" * 60)
    train_anomaly(dataset_path=data_path)
    return {"status": "ok"}


def retrain_classifier(data_path: str) -> dict:
    from Core.classifier import train as train_classifier
    print("\n" + "=" * 60)
    print("  [Retrain] Workload Classifier")
    print("=" * 60)
    return train_classifier(dataset_path=data_path)


# ── Retrain log ───────────────────────────────────────────────────────────────

def log_retrain(live_rows: int, merged_rows: int, predictor_metrics: dict):
    """Appends a retrain event to retrain_log.csv."""
    row = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "live_rows":   live_rows,
        "merged_rows": merged_rows,
    }
    for target, m in predictor_metrics.items():
        row[f"{target}_mae"] = m.get("MAE")
        row[f"{target}_r2"]  = m.get("R2")

    df = pd.DataFrame([row])
    exists = os.path.exists(RETRAIN_LOG)
    df.to_csv(RETRAIN_LOG, mode="a", header=not exists, index=False)
    print(f"\n[Retrain] Event logged -> {RETRAIN_LOG}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(force: bool = False, merge_kaggle: bool = True):
    """
    Full retraining pipeline.

    Args:
        force:        Skip row-count check and retrain regardless.
        merge_kaggle: Merge with Kaggle datasets for richer training set.
    """
    print("=" * 60)
    print("  SIT-Py | Retraining Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load live data
    print("\n[Retrain] Loading live data...")
    live_df = load_live_data()

    if live_df.empty:
        print("[Retrain] No live data found. Run Data/collect.py first.")
        return

    # 2. Check threshold
    if not force and not should_retrain(live_df):
        remaining = RETRAIN_ROW_THRESHOLD - len(live_df)
        mins_remaining = remaining * 2 / 60  # at 2s polling
        print(f"[Retrain] Not enough data yet. "
              f"Need {remaining} more rows (~{mins_remaining:.0f} min of collection).")
        return

    # 3. Merge with Kaggle if requested
    if merge_kaggle:
        print("\n[Retrain] Merging with Kaggle base datasets...")
        training_df = merge_with_kaggle(live_df)
    else:
        training_df = live_df

    # 4. Write merged dataset to a temp CSV for training functions to read
    merged_path = os.path.join(DATA_DIR, "_retrain_merged.csv")
    training_df.to_csv(merged_path, index=False)

    t_start = time.time()

    # 5. Retrain all models
    predictor_metrics = retrain_predictor(merged_path)
    retrain_anomaly(merged_path)
    retrain_classifier(merged_path)

    elapsed = time.time() - t_start

    # 6. Cleanup temp file
    try:
        os.remove(merged_path)
    except OSError:
        pass

    # 7. Log the event
    log_retrain(len(live_df), len(training_df), predictor_metrics)

    print(f"\n[Retrain] OK Complete - took {elapsed:.1f}s")
    print("[Retrain] Restart the main.py / dashboard to load new models.")


# ── Entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIT-Py Retraining Pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if below row threshold")
    parser.add_argument("--no-kaggle", action="store_true",
                        help="Train on live data only (no Kaggle merging)")
    args = parser.parse_args()

    run(force=args.force, merge_kaggle=not args.no_kaggle)
