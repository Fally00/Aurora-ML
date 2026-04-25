"""
Main SIT-Py runtime orchestrator.

Runs the collector + feature engineering + model inference loop and appends
normalized rows to sit_log.csv.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime

import pandas as pd
import psutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import ANOMALY_Z_THRESHOLD, LOG_PATH, MODELS_DIR, POLL_INTERVAL, WINDOW_SIZE
from Core.anomaly import AnomalyDetector
from Core.classifier import WorkloadClassifier
from Core.feature_engineering import FeatureEngineer
from Core.predictor import ResourcePredictor
from Core.schema import SystemSnapshot
from Data.collect import take_snapshot


IS_LINUX = platform.system() == "Linux"


class C:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _color_workload(workload: str) -> str:
    return {
        "Critical": f"{C.RED}{C.BOLD}{workload}{C.RESET}",
        "High": f"{C.YELLOW}{workload}{C.RESET}",
        "Medium": f"{C.CYAN}{workload}{C.RESET}",
        "Low": f"{C.GREEN}{workload}{C.RESET}",
    }.get(workload, workload)


def _color_anomaly(is_anomaly: bool, label: str) -> str:
    if is_anomaly:
        return f"{C.RED}{C.BOLD}{label}{C.RESET}"
    return f"{C.GREEN}{label}{C.RESET}"


def _ensure_log_schema(path: str, columns: list[str]) -> None:
    """Rotate an older log file out of the way if the header no longer matches."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            header = handle.readline().strip()
    except OSError:
        return

    existing_cols = header.split(",") if header else []
    if existing_cols == columns:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.replace(".csv", f"_legacy_{stamp}.csv")
    os.replace(path, backup_path)
    print(f"[SIT-Py] Existing log schema changed, moved old log to {backup_path}")


def log_result(
    snap: SystemSnapshot,
    prediction: dict,
    anomaly: dict,
    workload: dict,
) -> None:
    """Append one normalized row to sit_log.csv."""
    row = snap.to_log_row(prediction=prediction, anomaly=anomaly, workload=workload)
    frame = pd.DataFrame([row])
    _ensure_log_schema(LOG_PATH, list(frame.columns))
    frame.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


def print_cycle(
    cycle: int,
    snap: SystemSnapshot,
    prediction: dict,
    anomaly: dict,
    workload: dict,
    engineer: FeatureEngineer,
) -> None:
    """Render a terminal summary for the current cycle."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {C.BOLD}SIT-Py{C.RESET} | Cycle #{cycle} | {snap.timestamp}")
    print(sep)

    print("  CURRENT")
    print(
        f"    CPU  : {snap.cpu_pct:>6.1f}%   "
        f"RAM : {snap.ram_pct:>6.1f}%   "
        f"Disk: {snap.disk_pct:>6.1f}%"
    )
    print(
        f"    BW   : {snap.bandwidth_kbps:>8.1f} kbps   "
        f"Pkts: {snap.packet_rate_pps:>6.1f} pps   "
        f"Procs: {snap.process_count}"
    )

    if engineer.is_ready():
        print(
            f"    dCPU : {snap.cpu_delta:>+6.1f}%   "
            f"CPU sd: {snap.cpu_rolling_std:>5.2f}   "
            f"BW x: {snap.bw_spike:>5.2f}"
        )

    if prediction:
        print("\n  PREDICTED (next cycle)")
        print(
            f"    CPU  : {prediction.get('cpu_pct', 'N/A'):>6}%   "
            f"RAM : {prediction.get('ram_pct', 'N/A'):>6}%   "
            f"Disk: {prediction.get('disk_pct', 'N/A'):>6}%"
        )
    else:
        print(
            f"\n  PREDICTED : warming up "
            f"({engineer.history_len()}/{WINDOW_SIZE} cycles needed)..."
        )

    print("\n  STATUS")
    print(
        f"    Anomaly  : {_color_anomaly(anomaly['is_anomaly'], anomaly['label'])}   "
        f"(score: {anomaly['anomaly_score']}, src: {anomaly.get('source', '')})"
    )
    print(
        f"    Workload : {_color_workload(workload['workload'])}   "
        f"(confidence: {workload['confidence'] * 100:.1f}%)"
    )
    print(sep)


def run() -> None:
    """Run the live monitoring loop."""
    print(f"\n{C.BOLD}{'=' * 62}{C.RESET}")
    print(f"  {C.BOLD}SIT-Py | Intelligent Resource Monitor{C.RESET}")
    print(
        f"  Platform : {platform.system()} | "
        f"Sensors: {'ON' if IS_LINUX else 'OFF (Windows)'}"
    )
    print(f"  Log      : {LOG_PATH}")
    print(f"{'=' * 62}\n")

    predictor = ResourcePredictor(MODELS_DIR)
    detector = AnomalyDetector(MODELS_DIR)
    classifier = WorkloadClassifier(MODELS_DIR)

    print("[SIT-Py] Loading models...")

    try:
        predictor.load()
    except FileNotFoundError:
        print("[SIT-Py] Predictor model not found; predictions will be skipped.")
        print("          Train with: python Core/predictor.py")
        predictor = None

    anomaly_loaded = detector.load()
    if not anomaly_loaded:
        print("[SIT-Py] Anomaly detector is in statistical bridge mode.")
        print("          Train with: python Core/anomaly.py")

    try:
        classifier.load()
    except FileNotFoundError:
        print("[SIT-Py] Classifier model not found; classification will be skipped.")
        print("          Train with: python Core/classifier.py")
        classifier = None

    print("[SIT-Py] Pipeline ready. Starting...\n")

    engineer = FeatureEngineer(window_size=WINDOW_SIZE)
    prev_net = psutil.net_io_counters()
    last_t = time.time()
    cycle = 0

    try:
        while True:
            cycle += 1
            now = time.time()
            elapsed = now - last_t
            last_t = now

            snap, prev_net = take_snapshot(prev_net, elapsed)
            snap = engineer.enrich(snap)

            if detector.loaded:
                anomaly = detector.predict(snap.to_anomaly_input())
            else:
                anomaly = engineer.statistical_anomaly(
                    snap,
                    z_threshold=ANOMALY_Z_THRESHOLD,
                )

            if classifier:
                workload = classifier.predict(snap.to_classifier_input())
            else:
                workload = {"workload": "N/A", "confidence": 0.0, "probabilities": {}}

            prediction = {}
            if predictor and engineer.is_ready():
                prediction = predictor.predict(snap.to_predictor_input())

            print_cycle(cycle, snap, prediction, anomaly, workload, engineer)
            log_result(snap, prediction, anomaly, workload)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n[SIT-Py] Stopped. Log saved to {LOG_PATH}")


if __name__ == "__main__":
    run()
