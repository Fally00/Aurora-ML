"""
Retraining pipeline for all SIT-Py models.

The retrainer now prepares a dedicated dataset for each model:
  - predictor  -> raw time-series telemetry
  - classifier -> workload-labelled telemetry
  - anomaly    -> labelled anomaly telemetry
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.schema import (
    ANOMALY_CATEGORICAL_COLS,
    ANOMALY_FEATURE_COLS,
    ANOMALY_TARGET_COL,
    ANOMALY_NUMERIC_COLS,
    CLASSIFIER_FEATURE_COLS,
    CLASSIFIER_TARGET_COL,
    PREDICTOR_RAW_COLS,
    coerce_numeric_columns,
    ensure_columns,
    normalize_anomaly_label,
    normalize_columns,
    normalize_workload_label,
)
from config import (
    ANOMALY_DATASET_CSV,
    CLASSIFIER_DATASET_CSV,
    DATA_DIR,
    LABELED_LOG_WEIGHT,
    LIVE_CSV,
    LIVE_DATA_WEIGHT,
    LOG_PATH,
    PREDICTOR_DATASET_CSV,
    RANDOM_STATE,
    RETRAIN_LOG,
    RETRAIN_ROW_THRESHOLD,
    WINDOW_SIZE,
    WORKLOAD_THRESHOLDS,
)


def _read_csv_safe(path: str) -> pd.DataFrame:
    """Read a CSV, tolerating older malformed log rows."""
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _load_runtime_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load live_metrics.csv and sit_log.csv using the current normalized schema."""
    live_df = _read_csv_safe(LIVE_CSV)
    log_df = _read_csv_safe(LOG_PATH)

    if not live_df.empty:
        live_df = normalize_columns(live_df)
        print(f"  [data] live_metrics.csv : {len(live_df):,} rows")
    if not log_df.empty:
        log_df = normalize_columns(log_df)
        print(f"  [data] sit_log.csv      : {len(log_df):,} rows")

    return live_df, log_df


def _runtime_row_count(live_df: pd.DataFrame, log_df: pd.DataFrame) -> int:
    """Count unique runtime rows across live_metrics and sit_log."""
    frames = []
    for df in (live_df, log_df):
        if not df.empty:
            if "timestamp" in df.columns:
                frames.append(df[["timestamp"]].copy())
            else:
                frames.append(pd.DataFrame({"timestamp": range(len(df))}))

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    return len(combined)


def _rule_based_label(cpu: float, ram: float) -> str:
    thresholds = WORKLOAD_THRESHOLDS
    if cpu > thresholds["Critical"]["cpu"] or ram > thresholds["Critical"]["ram"]:
        return "Critical"
    if cpu > thresholds["High"]["cpu"] or ram > thresholds["High"]["ram"]:
        return "High"
    if cpu > thresholds["Medium"]["cpu"] or ram > thresholds["Medium"]["ram"]:
        return "Medium"
    return "Low"


def _load_base_frame(path: str, columns: list[str]) -> pd.DataFrame:
    df = _read_csv_safe(path)
    if df.empty:
        return df
    df = ensure_columns(df, columns)
    numeric_cols = [col for col in columns if col not in ANOMALY_CATEGORICAL_COLS and col not in {ANOMALY_TARGET_COL, CLASSIFIER_TARGET_COL}]
    df = coerce_numeric_columns(df, numeric_cols)
    return df.dropna(subset=[col for col in numeric_cols if col in df.columns])


def _build_predictor_dataset(live_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    base_df = _load_base_frame(PREDICTOR_DATASET_CSV, PREDICTOR_RAW_COLS)

    runtime_frames = []
    for df in (live_df, log_df):
        if df.empty:
            continue
        cols = ["timestamp"] + PREDICTOR_RAW_COLS
        frame = ensure_columns(df, cols)
        frame = coerce_numeric_columns(frame, PREDICTOR_RAW_COLS)
        frame = frame.dropna(subset=PREDICTOR_RAW_COLS)
        runtime_frames.append(frame)

    runtime_df = pd.DataFrame()
    if runtime_frames:
        runtime_df = pd.concat(runtime_frames, ignore_index=True)
        runtime_df = runtime_df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="last"
        )
        runtime_df = runtime_df[PREDICTOR_RAW_COLS].reset_index(drop=True)

    if not runtime_df.empty and len(runtime_df) > WINDOW_SIZE + 1:
        if not base_df.empty:
            merged = pd.concat([runtime_df, base_df], ignore_index=True)
        else:
            merged = runtime_df
    else:
        merged = base_df

    print(f"  [predictor] runtime rows : {len(runtime_df):,}")
    print(f"  [predictor] base rows    : {len(base_df):,}")
    print(f"  [predictor] final rows   : {len(merged):,}")
    return merged


def _build_classifier_dataset(live_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = CLASSIFIER_FEATURE_COLS + [CLASSIFIER_TARGET_COL]
    base_df = _load_base_frame(CLASSIFIER_DATASET_CSV, base_cols)
    if not base_df.empty:
        base_df[CLASSIFIER_TARGET_COL] = base_df[CLASSIFIER_TARGET_COL].map(normalize_workload_label)
        base_df = base_df.dropna(subset=[CLASSIFIER_TARGET_COL])

    runtime_frames = []
    for df in (live_df, log_df):
        if df.empty:
            continue
        frame = ensure_columns(df, base_cols)
        frame = coerce_numeric_columns(frame, CLASSIFIER_FEATURE_COLS)
        frame[CLASSIFIER_TARGET_COL] = frame[CLASSIFIER_TARGET_COL].map(normalize_workload_label)
        if frame[CLASSIFIER_TARGET_COL].isna().all():
            frame[CLASSIFIER_TARGET_COL] = [
                _rule_based_label(float(cpu or 0.0), float(ram or 0.0))
                for cpu, ram in zip(frame["cpu_pct"].fillna(0.0), frame["ram_pct"].fillna(0.0))
            ]
        else:
            mask = frame[CLASSIFIER_TARGET_COL].isna()
            frame.loc[mask, CLASSIFIER_TARGET_COL] = [
                _rule_based_label(float(cpu or 0.0), float(ram or 0.0))
                for cpu, ram in zip(
                    frame.loc[mask, "cpu_pct"].fillna(0.0),
                    frame.loc[mask, "ram_pct"].fillna(0.0),
                )
            ]
        frame = frame.dropna(subset=CLASSIFIER_FEATURE_COLS + [CLASSIFIER_TARGET_COL])
        runtime_frames.append(frame)

    runtime_df = pd.DataFrame()
    if runtime_frames:
        runtime_df = pd.concat(runtime_frames, ignore_index=True)
        runtime_df = runtime_df[base_cols].reset_index(drop=True)

    parts = []
    if not base_df.empty:
        parts.append(base_df)
    if not runtime_df.empty:
        parts.extend([runtime_df] * LIVE_DATA_WEIGHT)

    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=base_cols)
    if not merged.empty:
        merged = merged.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  [classifier] runtime rows : {len(runtime_df):,}")
    print(f"  [classifier] base rows    : {len(base_df):,}")
    print(f"  [classifier] final rows   : {len(merged):,}")
    return merged


def _build_anomaly_dataset(live_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ANOMALY_FEATURE_COLS + [ANOMALY_TARGET_COL]
    base_df = _load_base_frame(ANOMALY_DATASET_CSV, base_cols)
    if not base_df.empty:
        base_df[ANOMALY_TARGET_COL] = base_df[ANOMALY_TARGET_COL].map(normalize_anomaly_label)
        base_df = base_df.dropna(subset=[ANOMALY_TARGET_COL])

    runtime_df = pd.DataFrame(columns=base_cols)
    if not log_df.empty:
        frame = log_df.copy()
        if "anomaly_label" in frame.columns and ANOMALY_TARGET_COL not in frame.columns:
            frame[ANOMALY_TARGET_COL] = frame["anomaly_label"]
        frame = ensure_columns(frame, base_cols)
        frame = coerce_numeric_columns(frame, ANOMALY_NUMERIC_COLS)
        frame[ANOMALY_NUMERIC_COLS] = frame[ANOMALY_NUMERIC_COLS].fillna(0.0)
        for col in ANOMALY_CATEGORICAL_COLS:
            frame[col] = frame[col].fillna("").astype(str)
        frame[ANOMALY_TARGET_COL] = frame[ANOMALY_TARGET_COL].map(normalize_anomaly_label)
        runtime_df = frame.dropna(subset=[ANOMALY_TARGET_COL])
        runtime_df = runtime_df[base_cols].reset_index(drop=True)
        if runtime_df[ANOMALY_TARGET_COL].nunique() < 2:
            print("  [anomaly] runtime labels are single-class; skipping runtime blend")
            runtime_df = pd.DataFrame(columns=base_cols)

    parts = []
    if not base_df.empty:
        parts.append(base_df)
    if not runtime_df.empty:
        parts.extend([runtime_df] * LABELED_LOG_WEIGHT)

    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=base_cols)
    if not merged.empty:
        merged = merged.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  [anomaly] labeled runtime rows : {len(runtime_df):,}")
    print(f"  [anomaly] base rows            : {len(base_df):,}")
    print(f"  [anomaly] final rows           : {len(merged):,}")
    return merged


def should_retrain(live_df: pd.DataFrame, log_df: pd.DataFrame) -> bool:
    row_count = _runtime_row_count(live_df, log_df)
    print(f"\n[Retrain] Runtime rows: {row_count} / {RETRAIN_ROW_THRESHOLD} required")
    return row_count >= RETRAIN_ROW_THRESHOLD


def _write_temp_csv(df: pd.DataFrame, prefix: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        prefix=f"sitpy_{prefix}_",
        suffix=".csv",
        dir=DATA_DIR,
        delete=False,
    )
    handle.close()
    df.to_csv(handle.name, index=False)
    return handle.name


def _log_retrain(
    runtime_rows: int,
    predictor_rows: int,
    classifier_rows: int,
    anomaly_rows: int,
    predictor_metrics: dict,
    classifier_metrics: dict,
    anomaly_metrics: dict,
) -> None:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_rows": runtime_rows,
        "predictor_rows": predictor_rows,
        "classifier_rows": classifier_rows,
        "anomaly_rows": anomaly_rows,
        "classifier_accuracy": classifier_metrics.get("accuracy"),
        "anomaly_accuracy": anomaly_metrics.get("accuracy"),
    }

    for target, values in predictor_metrics.items():
        row[f"{target}_mae"] = values.get("MAE")
        row[f"{target}_r2"] = values.get("R2")

    exists = os.path.exists(RETRAIN_LOG)
    pd.DataFrame([row]).to_csv(RETRAIN_LOG, mode="a", header=not exists, index=False)
    print(f"[Retrain] Event logged -> {RETRAIN_LOG}")


def run(force: bool = False) -> None:
    print("=" * 60)
    print("  SIT-Py | Retraining Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[Retrain] Loading runtime data...")
    live_df, log_df = _load_runtime_sources()

    if live_df.empty and log_df.empty:
        print("[Retrain] No runtime data found. Run main.py or Data/collect.py first.")
        return

    if not force and not should_retrain(live_df, log_df):
        remaining = RETRAIN_ROW_THRESHOLD - _runtime_row_count(live_df, log_df)
        mins_remaining = remaining * 2 / 60
        print(
            f"[Retrain] Not enough runtime data yet. Need {remaining} more rows "
            f"(about {mins_remaining:.0f} more minutes at 2s polling)."
        )
        return

    print("\n[Retrain] Preparing per-model datasets...")
    predictor_df = _build_predictor_dataset(live_df, log_df)
    classifier_df = _build_classifier_dataset(live_df, log_df)
    anomaly_df = _build_anomaly_dataset(live_df, log_df)

    if predictor_df.empty or classifier_df.empty or anomaly_df.empty:
        print("[Retrain] One or more training datasets are empty; aborting.")
        return

    predictor_path = classifier_path = anomaly_path = ""
    start_time = time.time()

    try:
        predictor_path = _write_temp_csv(predictor_df, "predictor")
        classifier_path = _write_temp_csv(classifier_df, "classifier")
        anomaly_path = _write_temp_csv(anomaly_df, "anomaly")

        from Core.anomaly import train as train_anomaly
        from Core.classifier import train as train_classifier
        from Core.predictor import train as train_predictor

        print("\n" + "=" * 60)
        print("  [Retrain] Predictor")
        print("=" * 60)
        predictor_metrics = train_predictor(dataset_path=predictor_path)

        print("\n" + "=" * 60)
        print("  [Retrain] Workload Classifier")
        print("=" * 60)
        classifier_metrics = train_classifier(dataset_path=classifier_path)

        print("\n" + "=" * 60)
        print("  [Retrain] Anomaly Detector")
        print("=" * 60)
        anomaly_metrics = train_anomaly(dataset_path=anomaly_path)

    finally:
        for path in (predictor_path, classifier_path, anomaly_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    elapsed = time.time() - start_time
    runtime_rows = _runtime_row_count(live_df, log_df)
    _log_retrain(
        runtime_rows=runtime_rows,
        predictor_rows=len(predictor_df),
        classifier_rows=len(classifier_df),
        anomaly_rows=len(anomaly_df),
        predictor_metrics=predictor_metrics,
        classifier_metrics=classifier_metrics,
        anomaly_metrics=anomaly_metrics,
    )

    print(f"\n[Retrain] Complete in {elapsed:.1f}s")
    print("[Retrain] Restart main.py or the dashboard to load the new models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIT-Py Retraining Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if below the runtime row threshold",
    )
    args = parser.parse_args()
    run(force=args.force)
