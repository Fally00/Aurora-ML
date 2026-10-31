#!/usr/bin/env python3
"""
Dataset rebuild script for Aurora-ML.

Recreates the Data/datasets/ folder layout from raw CSVs in Data/raw/.
Runs the preprocessor logic programmatically and outputs:
  - anomaly_FINAL.csv
  - classifier_FINAL.csv
  - predictor_FINAL.csv

Final datasets are git-ignored (see .gitignore).

Usage:
    python Data/build_datasets.py

Raw CSV files should be placed in Data/raw/ with names matching the CONFIGS keys
in Data/preprocess.py, or you can modify the RAW_FILES mapping below.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Data.preprocess import (
    CONFIGS,
    SCHEMAS,
    TEXT_DEFAULTS,
    validate_dataframe,
    cap_class,
    merge_finals,
    load,
    save,
    show_labels,
    schema_defaults,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Raw data directory
RAW_DIR = _ROOT / "Data" / "raw"
OUT_DIR = _ROOT / "Data" / "datasets"
PREPROCESSED_DIR = _ROOT / "Data" / "preprocessed"


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Aurora-ML Dataset Builder")
    print("=" * 60)

    # Check raw directory
    if not RAW_DIR.exists():
        print(f"\n[ERROR] Raw data directory not found: {RAW_DIR}")
        print("Create it and place your raw CSV files there.")
        print("Expected files (per CONFIGS in preprocess.py):")
        for output_name, cfg in CONFIGS.items():
            if cfg["target"] in ("anomaly", "classifier", "predictor"):
                print(f"  {cfg['file']} -> {output_name}")
        return 1

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process each config
    print(f"\n[STEP 1] Processing individual raw files...")
    for output_name, cfg in CONFIGS.items():
        process_config(output_name, cfg)

    # Merge final datasets
    print(f"\n[STEP 2] Merging final datasets...")
    merge_finals()

    print("\n" + "=" * 60)
    print(f"  Done. Final CSVs saved to: {OUT_DIR}")
    print("=" * 60)

    # Verify outputs
    for target in ("anomaly", "classifier", "predictor"):
        final_path = OUT_DIR / f"{target}_FINAL.csv"
        if final_path.exists():
            df = pd.read_csv(final_path)
            print(f"\n{target}_FINAL.csv: {len(df)} rows, {len(df.columns)} cols")
            print(f"  Columns: {list(df.columns)}")
            if target == "anomaly" and "label" in df.columns:
                print(f"  Label dist: {df['label'].value_counts().to_dict()}")
            elif target == "classifier" and "task_priority" in df.columns:
                print(f"  Label dist: {df['task_priority'].value_counts().to_dict()}")
        else:
            print(f"\n[WARN] {target}_FINAL.csv not found")

    return 0


def process_config(output_name: str, cfg: dict):
    """Process a single raw CSV through the preprocessing pipeline."""
    print(f"\n-- {output_name} --")

    # Load raw file
    raw_path = RAW_DIR / cfg["file"]
    if not raw_path.exists():
        print(f"  [SKIP] Raw file not found: {raw_path}")
        return

    df = pd.read_csv(raw_path)
    print(f"  [LOAD] {cfg['file']} -> {df.shape[0]} rows, {df.shape[1]} cols")

    # Step 1: keep + rename only requested columns
    keep = cfg["keep"]
    existing = {k: v for k, v in keep.items() if k in df.columns}
    missing = [k for k in keep if k not in df.columns]

    if missing:
        print(f"  [WARN] Missing columns (will fill with schema defaults): {missing}")

    df = df[list(existing.keys())].rename(columns=existing)

    # Step 2: special case - telesurgery bandwidth conversion (Mbps -> kbps)
    if output_name == "anomaly_telesurgery.csv" and "bandwidth_kbps" in df.columns:
        df["bandwidth_kbps"] = (df["bandwidth_kbps"] * 1000).round(2)

    # Step 3: apply label mapping
    label_col = cfg.get("label_col")
    label_map = cfg.get("label_map", {})
    if label_col and label_map and label_col in df.columns:
        df[label_col] = df[label_col].map(
            lambda x: label_map.get(x, label_map.get(str(x).lower(), x))
        )
        valid_labels = set(label_map.values())
        before = len(df)
        df = df[df[label_col].isin(valid_labels)]
        if len(df) < before:
            print(f"  [DROP] {before - len(df)} rows with unmapped labels")

    # Step 4: enforce unified schema
    target = cfg["target"]
    schema = SCHEMAS[target]
    for col in schema:
        if col not in df.columns:
            df[col] = TEXT_DEFAULTS.get(col, 0.0)

    df = df[schema]  # reorder + drop anything extra
    df = validate_dataframe(df)

    # Step 5: drop nulls
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"  [DROP] {before - len(df)} null rows")

    # Step 6: cap majority classes
    cap = cfg.get("cap")
    if cap:
        cap_col = label_col if label_col in df.columns else None
        if cap_col:
            df = cap_class(df, cap_col, cap)
        else:
            # predictor - just sample down
            if len(df) > cap:
                df = df.sample(n=cap, random_state=42).reset_index(drop=True)

    # Step 7: show label dist + save
    if label_col and label_col in df.columns:
        show_labels(df, label_col)

    save(df, output_name)


if __name__ == "__main__":
    sys.exit(main())