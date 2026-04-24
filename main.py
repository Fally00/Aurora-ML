"""
main.py - SIT-Py Orchestrator (v2)
====================================
Central pipeline coordinator. Runs the three ML modules on a live polling loop.

Changes from v1:
  - Uses SystemSnapshot (unified schema) - no more manual remapping functions
  - FeatureEngineer provides temporal context for all modules
  - AnomalyDetector falls back to statistical bridge if IF model not trained yet
  - Log columns now match unified schema (not the old CPUUsage / CPU_Usage split)
  - Path resolution centralized in config.py
"""
from __future__ import annotations

import os
import sys
import time
import platform
from collections import deque
from datetime import datetime

import psutil
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from config import (
    POLL_INTERVAL, MODELS_DIR, LOG_PATH, ANOMALY_Z_THRESHOLD,
)
from Core.schema            import SystemSnapshot
from Core.feature_engineering import FeatureEngineer
from Core.predictor         import ResourcePredictor
from Core.anomaly           import AnomalyDetector
from Core.classifier        import WorkloadClassifier
from Data.collect           import take_snapshot

IS_LINUX = platform.system() == "Linux"

# ── Color codes ───────────────────────────────────────────────────────────────
class C:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def _color_workload(w: str) -> str:
    return {
        "Critical": f"{C.RED}{C.BOLD}{w}{C.RESET}",
        "High":     f"{C.YELLOW}{w}{C.RESET}",
        "Medium":   f"{C.CYAN}{w}{C.RESET}",
        "Low":      f"{C.GREEN}{w}{C.RESET}",
    }.get(w, w)

def _color_anomaly(is_anomaly: bool, label: str) -> str:
    return f"{C.RED}{C.BOLD}{label}{C.RESET}" if is_anomaly else f"{C.GREEN}{label}{C.RESET}"


# ── Logging ───────────────────────────────────────────────────────────────────

def log_result(snap: SystemSnapshot, prediction: dict, anomaly: dict, workload: dict):
    """Appends one row to sit_log.csv using the unified schema column names."""
    row = snap.to_log_row(prediction=prediction, anomaly=anomaly, workload=workload)
    df  = pd.DataFrame([row])
    exists = os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=not exists, index=False)


# ── Terminal output ───────────────────────────────────────────────────────────

def print_cycle(cycle: int, snap: SystemSnapshot, prediction: dict,
                anomaly: dict, workload: dict, eng: FeatureEngineer):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {C.BOLD}SIT-Py{C.RESET} | Cycle #{cycle} | {snap.timestamp}")
    print(sep)

    print(f"  CURRENT")
    print(f"    CPU    : {snap.cpu_pct:>6.1f}%   RAM  : {snap.ram_pct:>6.1f}%   "
          f"Disk : {snap.disk_pct:>6.1f}%")
    print(f"    BW     : {snap.bandwidth_kbps:>6.1f} kbps  "
          f"Pkts : {snap.packet_rate_pps:>6.1f} pps  "
          f"Procs: {snap.process_count}")

    if eng.is_ready():
        print(f"    ? CPU  : {snap.cpu_delta:>+6.1f}%   "
              f"? CPU : {snap.cpu_rolling_std:>5.2f}")

    if prediction:
        print(f"\n  PREDICTED (next cycle)")
        print(f"    CPU    : {prediction.get('cpu_pct', 'N/A'):>6}%   "
              f"RAM  : {prediction.get('ram_pct', 'N/A'):>6}%   "
              f"Disk : {prediction.get('disk_pct', 'N/A'):>6}%")
    else:
        print(f"\n  PREDICTED  : warming up ({eng.history_len()}/{15} cycles needed)...")

    anom_src = anomaly.get("source", "")
    print(f"\n  STATUS")
    print(f"    Anomaly  : {_color_anomaly(anomaly['is_anomaly'], anomaly['label'])}  "
          f"(score: {anomaly['anomaly_score']}, src: {anom_src})")
    print(f"    Workload : {_color_workload(workload['workload'])}  "
          f"(confidence: {workload['confidence']*100:.1f}%)")
    print(sep)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    print(f"\n{C.BOLD}{'='*62}{C.RESET}")
    print(f"  {C.BOLD}SIT-Py | Intelligent Resource Monitor v2{C.RESET}")
    print(f"  Platform : {platform.system()} | "
          f"Sensors: {'ON' if IS_LINUX else 'OFF (Windows)'}")
    print(f"  Log      : {LOG_PATH}")
    print(f"{'='*62}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[SIT-Py] Loading models...")
    predictor  = ResourcePredictor(MODELS_DIR)
    detector   = AnomalyDetector(MODELS_DIR)
    classifier = WorkloadClassifier(MODELS_DIR)

    try:
        predictor.load()
    except FileNotFoundError:
        print("[SIT-Py] [!]  Predictor model not found - will skip predictions.")
        print("          Run: python Core/predictor.py")
        predictor = None

    # Anomaly detector: graceful fallback to statistical bridge
    if_loaded = detector.load()   # returns False -> stat bridge mode
    if not if_loaded:
        print("[SIT-Py] [i]  Anomaly detector in STATISTICAL BRIDGE mode.")
        print("          Run: python Core/anomaly.py (after collecting live data)")

    try:
        classifier.load()
    except FileNotFoundError:
        print("[SIT-Py] [!]  Classifier model not found - will skip classification.")
        print("          Run: python Core/classifier.py")
        classifier = None

    print("[SIT-Py] Pipeline ready. Starting...\n")

    # ── Feature engineer (stateful rolling-window) ───────────────────────────
    eng      = FeatureEngineer(window_size=15)
    prev_net = psutil.net_io_counters()
    last_t   = time.time()
    cycle    = 0

    try:
        while True:
            cycle += 1
            now     = time.time()
            elapsed = now - last_t
            last_t  = now

            # 1. Collect raw snapshot
            snap, prev_net = take_snapshot(prev_net, elapsed)

            # 2. Enrich with temporal features
            snap = eng.enrich(snap)

            # 3. Anomaly detection
            if detector.loaded:
                # IF model is available - use it
                anomaly = detector.predict(snap.to_anomaly_input())
            else:
                # Fall back to statistical bridge
                anomaly = eng.statistical_anomaly(snap, z_threshold=ANOMALY_Z_THRESHOLD)

            # 4. Workload classification
            if classifier:
                workload = classifier.predict(snap.to_classifier_input())
            else:
                workload = {"workload": "N/A", "confidence": 0.0, "probabilities": {}}

            # 5. Prediction (needs enough history)
            prediction = {}
            if predictor and eng.is_ready():
                prediction = predictor.predict(snap.to_predictor_input())

            # 6. Output + log
            print_cycle(cycle, snap, prediction, anomaly, workload, eng)
            log_result(snap, prediction, anomaly, workload)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n[SIT-Py] Stopped. Log saved -> {LOG_PATH}")


# ── Entry ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()