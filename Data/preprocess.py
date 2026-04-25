"""
SIT-Py Dataset Preprocessor
============================
Run this script on any raw Kaggle CSV to filter, rename,
and export clean unified-schema CSVs ready for model training.

Usage:
    python preprocess.py

Just drop your raw CSVs in the same folder as this script,
update the FILE CONFIGS section below with the correct filenames,
then run. Output goes to /preprocessed folder.
"""

import os
import pandas as pd
import numpy as np

# ── Output folder ─────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────

def load(filename: str) -> pd.DataFrame:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {filename}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  [LOAD] {filename} -> {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def save(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  [SAVE] {name} -> {len(df)} rows, cols: {df.columns.tolist()}")


def show_labels(df: pd.DataFrame, col: str):
    if col in df.columns:
        print(f"         label dist -> {df[col].value_counts().to_dict()}")


def cap_class(df: pd.DataFrame, label_col: str, max_per_class: int) -> pd.DataFrame:
    """Caps majority classes to max_per_class rows, keeps all minority rows."""
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) > max_per_class:
            parts.append(group.sample(n=max_per_class, random_state=42))
        else:
            parts.append(group)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)


def schema_defaults(columns: list[str]) -> dict:
    return {col: TEXT_DEFAULTS.get(col, 0.0) for col in columns}


# ════════════════════════════════════════════════════════════════════
# FILE CONFIGS
# Edit the filenames below to match your actual downloaded CSV names.
# ════════════════════════════════════════════════════════════════════

# Each entry is:
#   "output_name.csv" : {
#       "file"    : "raw_filename.csv",
#       "target"  : "anomaly" | "classifier" | "predictor",
#       "keep"    : { "raw_col_name": "unified_col_name", ... },
#       "label_map": { "raw_value": "unified_value", ... },   # optional
#       "label_col": "name_of_label_column_after_rename",     # optional
#       "cap"     : 2000,                                     # optional, per-class cap
#   }

CONFIGS = {

    # ── ANOMALY DATASETS ────────────────────────────────────────────

    "anomaly_cybersecurity.csv": {
        "file"   : "cybersecurity.csv",
        "target" : "anomaly",
        "keep"   : {
            "protocol"    : "protocol",
            "attack_type" : "attack_type",
            "label"       : "label",
        },
        "label_map" : {0: "Normal", 1: "Alert"},
        "label_col" : "label",
        "cap"       : 3000,
    },

    "anomaly_telesurgery.csv": {
        "file"   : "telesurgery_cybersecurity_dataset.csv",
        "target" : "anomaly",
        "keep"   : {
            "Data Transfer Rate (Mbps)" : "bandwidth_kbps",
            "Network Latency (ms)"      : "packet_rate_pps",
            "Threat Detected"           : "label",
            "Threat Severity"           : "severity",
            "Threat Type"               : "threat_type",
        },
        "label_map" : {0: "Normal", 1: "Alert"},
        "label_col" : "label",
    },

    "anomaly_threat_logs.csv": {
        "file"   : "cybersecurity_threat_detection_logs.csv",   # aryan208 dataset
        "target" : "anomaly",
        "keep"   : {
            "protocol"          : "protocol",
            "bytes_transferred" : "bandwidth_kbps",
            "threat_label"      : "label",
        },
        "label_map" : {"benign": "Normal", "malicious": "Alert", "suspicious": "Alert"},
        "label_col" : "label",
    },

    "anomaly_luflow.csv": {
        "file"   : "luflow.csv",                                # mryanm dataset
        "target" : "anomaly",
        "keep"   : {
            # update these to match actual column names
            "proto"    : "packet_rate_pps",
            "bytes"    : "bandwidth_kbps",
            "label"    : "label",
        },
        "label_map" : {"benign": "Normal", "malicious": "Alert"},
        "label_col" : "label",
        "cap"       : 5000,
    },

    # ── CLASSIFIER DATASETS ─────────────────────────────────────────

    "classifier_distributed.csv": {
        "file"   : "distributed_system_architecture_stress_dataset.csv",
        "target" : "classifier",
        "keep"   : {
            "cpu_utilization_percent"    : "cpu_pct",
            "memory_utilization_percent" : "ram_pct",
            "network_latency_ms"         : "bandwidth_kbps",
            "packet_loss_percent"        : "packet_rate_pps",
            "system_state"               : "task_priority",
        },
        "label_map" : {
            "healthy"           : "Low",
            "degraded"          : "Medium",
            "cascading_failure" : "High",
            "total_outage"      : "Critical",
        },
        "label_col" : "task_priority",
        "cap"       : 4000,
    },

    "classifier_threat.csv": {
        "file"   : "cybersecurity_threat_detection.csv",        # ziya07 dataset
        "target" : "classifier",
        "keep"   : {
            # update these to match actual column names
            "cpu_usage"      : "cpu_pct",
            "memory_usage"   : "ram_pct",
            "severity"       : "task_priority",
        },
        "label_map" : {
            "low"      : "Low",
            "medium"   : "Medium",
            "high"     : "High",
            "critical" : "Critical",
        },
        "label_col" : "task_priority",
        "cap"       : 4000,
    },

    # ── PREDICTOR DATASETS ──────────────────────────────────────────

    "predictor_distributed.csv": {
        "file"   : "distributed_system_architecture_stress_dataset.csv",
        "target" : "predictor",
        "keep"   : {
            "cpu_utilization_percent"    : "cpu_pct",
            "memory_utilization_percent" : "ram_pct",
            "network_latency_ms"         : "bandwidth_kbps",
            "packet_loss_percent"        : "packet_rate_pps",
            "error_rate_percent"         : "disk_pct",
        },
        "cap" : 20000,
    },

}


# ════════════════════════════════════════════════════════════════════
# UNIFIED TARGET SCHEMAS
# All output files must have exactly these columns.
# Numeric features default to 0.0; categorical features use empty strings.
# ════════════════════════════════════════════════════════════════════

SCHEMAS = {
    "anomaly"    : [
        "cpu_pct",
        "ram_pct",
        "bandwidth_kbps",
        "packet_rate_pps",
        "protocol",
        "attack_type",
        "severity",
        "threat_type",
        "label",
    ],
    "classifier" : ["cpu_pct", "ram_pct", "bandwidth_kbps", "packet_rate_pps", "task_priority"],
    "predictor"  : ["cpu_pct", "ram_pct", "disk_pct", "bandwidth_kbps", "packet_rate_pps"],
}

TEXT_DEFAULTS = {
    "protocol": "",
    "attack_type": "",
    "severity": "",
    "threat_type": "",
}


# ════════════════════════════════════════════════════════════════════
# PROCESSOR
# ════════════════════════════════════════════════════════════════════

def process(output_name: str, cfg: dict):
    print(f"\n-- {output_name} --")

    df = load(cfg["file"])
    if df.empty:
        return

    # ── Step 1: keep + rename only requested columns ──
    keep = cfg["keep"]
    existing = {k: v for k, v in keep.items() if k in df.columns}
    missing  = [k for k in keep if k not in df.columns]

    if missing:
        print(f"  [WARN] Missing columns (will fill with schema defaults): {missing}")

    df = df[list(existing.keys())].rename(columns=existing)

    # ── Step 2: special case — cybersecurity bandwidth derivation ──
    if "_bytes_recv" in df.columns:
        df["bandwidth_kbps"] = ((df["bandwidth_kbps"] + df["_bytes_recv"]) * 8 / 1000).round(2)
        df = df.drop(columns=["_bytes_recv"])

    # ── Step 3: telesurgery bandwidth conversion (Mbps → kbps) ──
    if output_name == "anomaly_telesurgery.csv" and "bandwidth_kbps" in df.columns:
        df["bandwidth_kbps"] = (df["bandwidth_kbps"] * 1000).round(2)

    # ── Step 4: apply label mapping ──
    if output_name == "anomaly_threat_logs.csv" and "label" in df.columns and "threat_type" not in df.columns:
        df["threat_type"] = df["label"]

    label_col = cfg.get("label_col")
    label_map = cfg.get("label_map", {})
    if label_col and label_map and label_col in df.columns:
        df[label_col] = df[label_col].map(
            lambda x: label_map.get(x, label_map.get(str(x).lower(), x))
        )
        # drop rows with unmapped labels
        valid_labels = set(label_map.values())
        before = len(df)
        df = df[df[label_col].isin(valid_labels)]
        if len(df) < before:
            print(f"  [DROP] {before - len(df)} rows with unmapped labels")

    # ── Step 5: enforce unified schema ──
    target  = cfg["target"]
    schema  = SCHEMAS[target]
    for col in schema:
        if col not in df.columns:
            df[col] = TEXT_DEFAULTS.get(col, 0.0)

    df = df[schema]  # reorder + drop anything extra

    # ── Step 6: drop nulls ──
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"  [DROP] {before - len(df)} null rows")

    # ── Step 7: cap majority classes ──
    cap = cfg.get("cap")
    if cap:
        cap_col = label_col if label_col in df.columns else None
        if cap_col:
            df = cap_class(df, cap_col, cap)
        else:
            # predictor — just sample down
            if len(df) > cap:
                df = df.sample(n=cap, random_state=42).reset_index(drop=True)

    # ── Step 8: show label dist + save ──
    if label_col and label_col in df.columns:
        show_labels(df, label_col)

    save(df, output_name)


# ════════════════════════════════════════════════════════════════════
# MERGE STEP
# After all individual files are processed, merge by target type
# ════════════════════════════════════════════════════════════════════

def merge_finals():
    print("\n\n-- MERGING FINAL DATASETS --")

    for target in ["anomaly", "classifier", "predictor"]:
        parts = []
        label_col = {"anomaly": "label", "classifier": "task_priority", "predictor": None}[target]

        for name in CONFIGS:
            if CONFIGS[name]["target"] == target:
                path = os.path.join(OUT_DIR, name)
                if os.path.exists(path):
                    parts.append(pd.read_csv(path, keep_default_na=False))

        if not parts:
            print(f"  [SKIP] No files found for target: {target}")
            continue

        merged = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
        schema = SCHEMAS[target]
        defaults = schema_defaults(schema)

        for col in schema:
            if col not in merged.columns:
                merged[col] = defaults[col]

        merged = merged[schema].fillna(defaults)
        merged = merged.dropna()

        # final balance pass on merged
        if label_col and label_col in merged.columns:
            merged = cap_class(merged, label_col, max_per_class=3000)
            show_labels(merged, label_col)

        out_name = f"{target}_FINAL.csv"
        save(merged, out_name)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  SIT-Py Dataset Preprocessor")
    print("=" * 55)

    for output_name, cfg in CONFIGS.items():
        process(output_name, cfg)

    merge_finals()

    print("\n" + "=" * 55)
    print(f"  Done. Clean CSVs saved to: {OUT_DIR}")
    print("=" * 55)
