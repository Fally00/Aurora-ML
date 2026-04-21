import os
import sys
import time
import platform
import pandas as pd
import psutil
from datetime import datetime
from collections import deque

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from Core.predictor   import ResourcePredictor, WINDOW_SIZE
from Core.anomly      import AnomalyDetector
from Core.classifier  import WorkloadClassifier
from Data.collect     import take_snapshot

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL  = 2          # seconds between cycles
LOG_PATH       = os.path.join(ROOT_DIR, "data", "sit_log.csv")
MODELS_DIR     = os.path.join(ROOT_DIR, "models")
IS_LINUX       = platform.system() == "Linux"

# ── Color codes for terminal output ───────────────────────────────────────────
class C:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def color_workload(w: str) -> str:
    return {
        "Critical": f"{C.RED}{C.BOLD}{w}{C.RESET}",
        "High"    : f"{C.YELLOW}{w}{C.RESET}",
        "Medium"  : f"{C.CYAN}{w}{C.RESET}",
        "Low"     : f"{C.GREEN}{w}{C.RESET}",
    }.get(w, w)

def color_anomaly(is_anomaly: bool, label: str) -> str:
    return f"{C.RED}{C.BOLD}{label}{C.RESET}" if is_anomaly else f"{C.GREEN}{label}{C.RESET}"

# ── Log to CSV ─────────────────────────────────────────────────────────────────
def log_result(snapshot: dict, prediction: dict, anomaly: dict, workload: dict):
    row = {
        "timestamp"      : snapshot.get("timestamp"),
        "CPUUsage"       : snapshot.get("CPUUsage"),
        "RAMUsage"       : snapshot.get("RAMUsage"),
        "DiskUsage"      : snapshot.get("DiskUsage"),
        "Bandwidth"      : snapshot.get("Bandwidth"),
        "Packet_Rate"    : snapshot.get("Packet_Rate"),
        "Task_Priority"  : snapshot.get("Task_Priority"),
        "pred_CPU"       : prediction.get("CPUUsage", "N/A"),
        "pred_RAM"       : prediction.get("RAMUsage", "N/A"),
        "pred_Disk"      : prediction.get("DiskUsage", "N/A"),
        "anomaly_label"  : anomaly.get("label"),
        "anomaly_score"  : anomaly.get("anomaly_score"),
        "workload"       : workload.get("workload"),
        "wl_confidence"  : workload.get("confidence"),
    }
    df          = pd.DataFrame([row])
    file_exists = os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)

# ── Print cycle summary ────────────────────────────────────────────────────────
def print_cycle(cycle: int, snapshot: dict, prediction: dict, anomaly: dict, workload: dict):
    ts  = snapshot.get("timestamp", "")
    sep = "─" * 62

    print(f"\n{sep}")
    print(f"  {C.BOLD}SIT-Py{C.RESET} | Cycle #{cycle} | {ts}")
    print(sep)

    # current metrics
    print(f"  {'CURRENT':}")
    print(f"    CPU    : {snapshot['CPUUsage']:>6.1f}%   RAM  : {snapshot['RAMUsage']:>6.1f}%   Disk : {snapshot['DiskUsage']:>6.1f}%")
    print(f"    BW     : {snapshot['Bandwidth']:>6.1f} kbps   Pkts : {snapshot['Packet_Rate']:>6.1f} pps")

    # predictions
    if prediction:
        print(f"\n  {'PREDICTED (next cycle)':}")
        print(f"    CPU    : {prediction.get('CPUUsage', 'N/A'):>6}%   RAM  : {prediction.get('RAMUsage', 'N/A'):>6}%   Disk : {prediction.get('DiskUsage', 'N/A'):>6}%")
    else:
        print(f"\n  PREDICTED  : warming up ({WINDOW_SIZE} cycles needed)...")

    # anomaly + workload
    print(f"\n  {'STATUS':}")
    print(f"    Anomaly  : {color_anomaly(anomaly['is_anomaly'], anomaly['label'])}  (score: {anomaly['anomaly_score']})")
    print(f"    Workload : {color_workload(workload['workload'])}  (confidence: {workload['confidence']*100:.1f}%)")
    print(sep)

# ── Build anomaly snapshot from live metrics ───────────────────────────────────
def build_anomaly_snap(snapshot: dict) -> dict:
    """Maps collect.py output keys → anomaly.py expected keys."""
    return {
        "CPU_Usage"       : snapshot.get("CPUUsage", 0.0),
        "Memory_Usage"    : snapshot.get("RAMUsage", 0.0),
        "Bandwidth"       : snapshot.get("Bandwidth", 0.0),
        "Packet_Rate"     : snapshot.get("Packet_Rate", 0.0),
        "Failed_Logins"   : 0,    # psutil can't get this — default 0
        "Malware_Alerts"  : 0,    # aurora's job eventually
        "Intrusion_Alerts": 0,
        "Task_Priority"   : snapshot.get("Task_Priority", "Medium"),
        "Traffic_Type"    : "HTTP",   # default — collector doesn't inspect packets
    }

# ── Build classifier snapshot ──────────────────────────────────────────────────
def build_classifier_snap(snapshot: dict) -> dict:
    """Maps collect.py output → classifier.py expected keys."""
    return {
        "CPU_Usage"       : snapshot.get("CPUUsage", 0.0),
        "Memory_Usage"    : snapshot.get("RAMUsage", 0.0),
        "Bandwidth"       : snapshot.get("Bandwidth", 0.0),
        "Packet_Rate"     : snapshot.get("Packet_Rate", 0.0),
        "Failed_Logins"   : 0,
        "Malware_Alerts"  : 0,
        "Intrusion_Alerts": 0,
        "Traffic_Type"    : "HTTP",
    }

# ── Build predictor window from deque ─────────────────────────────────────────
def build_predictor_window(window_deque: deque) -> list:
    """Maps collect.py keys → predictor.py expected keys."""
    result = []
    for snap in window_deque:
        result.append({
            "CPUUsage"   : snap.get("CPUUsage", 0.0),
            "RAMUsage"   : snap.get("RAMUsage", 0.0),
            "Temperature": snap.get("Temperature", 0.0),
            "DiskUsage"  : snap.get("DiskUsage", 0.0),
            "FanSpeed"   : max(snap.get("FanSpeed", 0), 0),  # -1 on Windows → 0
        })
    return result

# ── Main Pipeline ──────────────────────────────────────────────────────────────
def run():
    print(f"\n{C.BOLD}{'='*62}{C.RESET}")
    print(f"  {C.BOLD}SIT-Py | Intelligent Resource Monitor{C.RESET}")
    print(f"  Platform : {platform.system()} | Sensors: {'ON' if IS_LINUX else 'OFF (Windows)'}")
    print(f"  Log      : {LOG_PATH}")
    print(f"{'='*62}\n")

    # ── Load all models ──
    print("[SIT-Py] Loading models...")
    predictor  = ResourcePredictor(MODELS_DIR)
    detector   = AnomalyDetector(MODELS_DIR)
    classifier = WorkloadClassifier(MODELS_DIR)

    predictor.load()
    detector.load()
    classifier.load()
    print("[SIT-Py] All models loaded. Starting pipeline...\n")

    # sliding window buffer for predictor
    window_buffer = deque(maxlen=WINDOW_SIZE)

    prev_net  = psutil.net_io_counters()   # seed for bandwidth delta
    cycle     = 0

    try:
        while True:
            cycle += 1

            # ── 1. Collect live snapshot ──
            snapshot, prev_net = take_snapshot(prev_net)

            # ── 2. Run AnomalyDetector ──
            anomaly_snap   = build_anomaly_snap(snapshot)
            anomaly_result = detector.predict(anomaly_snap)

            # ── 3. Run WorkloadClassifier ──
            classifier_snap    = build_classifier_snap(snapshot)
            classifier_result  = classifier.predict(classifier_snap)

            # ── 4. Run ResourcePredictor (needs full window) ──
            window_buffer.append(snapshot)
            if len(window_buffer) == WINDOW_SIZE:
                predictor_window  = build_predictor_window(window_buffer)
                prediction_result = predictor.predict(predictor_window)
            else:
                prediction_result = {}    # not enough data yet

            # ── 5. Print + Log ──
            print_cycle(cycle, snapshot, prediction_result, anomaly_result, classifier_result)
            log_result(snapshot, prediction_result, anomaly_result, classifier_result)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n[SIT-Py] Stopped. Log saved → {LOG_PATH}")

# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()