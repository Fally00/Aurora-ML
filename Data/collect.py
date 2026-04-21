import psutil
import pandas as pd
import time
import os
import platform
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL   = 2          # seconds between each snapshot
OUTPUT_CSV      = os.path.join(os.path.dirname(__file__), "live_metrics.csv")
MAX_ROWS        = 5000       # cap so CSV doesn't explode overnight
IS_LINUX        = platform.system() == "Linux"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_temperature() -> float:
    """
    Returns average CPU temp in Celsius.
    Only works on Linux with sensor support.
    Returns -1.0 on Windows or unsupported systems.
    """
    if not IS_LINUX:
        return -1.0
    try:
        temps_func = getattr(psutil, "sensors_temperatures", None)
        if not temps_func:
            return -1.0
        temps = temps_func()
        if not temps:
            return -1.0
        # grab first available sensor group (coretemp, k10temp, etc.)
        for key, entries in temps.items():
            if entries:
                values = [getattr(e, "current", None) for e in entries]
                values = [v for v in values if v is not None]
                if values:
                    return round(sum(values) / len(values), 2)
    except Exception:
        return -1.0
    return -1.0


def get_fan_speed() -> int:
    """
    Returns first fan speed in RPM.
    Linux only. Returns -1 on Windows.
    """
    if not IS_LINUX:
        return -1
    try:
        fans_func = getattr(psutil, "sensors_fans", None)
        if not fans_func:
            return -1
        fans = fans_func()
        if not fans:
            return -1
        for entries in fans.values():
            if entries:
                first = entries[0]
                val = getattr(first, "current", None)
                return int(val) if val is not None else -1
    except Exception:
        return -1
    return -1


def get_task_priority() -> str:
    """
    Derives workload priority from running process nice values.
    Maps to: Low / Medium / High / Critical — mirrors dataset labels.
    """
    try:
        nice_values = []
        for proc in psutil.process_iter(['nice']):
            try:
                n = proc.info['nice']
                if n is not None:
                    nice_values.append(n)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not nice_values:
            return "Medium"

        avg_nice = sum(nice_values) / len(nice_values)

        # lower nice = higher priority in Unix; Windows uses different scale
        if avg_nice <= 0:
            return "Critical"
        elif avg_nice <= 5:
            return "High"
        elif avg_nice <= 15:
            return "Medium"
        else:
            return "Low"
    except Exception:
        return "Medium"


def get_bandwidth_and_packets(prev_net) -> Tuple[float, float]:
    """
    Returns (bandwidth_kbps, packet_rate_pps) since last poll.
    Needs previous net_io snapshot to compute delta.
    """
    curr_net = psutil.net_io_counters()

    if prev_net is None:
        return 0.0, 0.0

    bytes_delta   = (curr_net.bytes_sent + curr_net.bytes_recv) - \
                    (prev_net.bytes_sent + prev_net.bytes_recv)
    packets_delta = (curr_net.packets_sent + curr_net.packets_recv) - \
                    (prev_net.packets_sent + prev_net.packets_recv)

    bandwidth_kbps  = round((bytes_delta * 8) / (POLL_INTERVAL * 1000), 2)  # kbps
    packet_rate_pps = round(packets_delta / POLL_INTERVAL, 2)                # pps

    return bandwidth_kbps, packet_rate_pps


# ── Main Snapshot ─────────────────────────────────────────────────────────────

def take_snapshot(prev_net) -> Tuple[Dict[str, Any], Any]:
    """
    Captures a full system snapshot.
    Returns dict of metrics + current net_io for next delta.
    """
    cpu_usage    = psutil.cpu_percent(interval=None)
    ram          = psutil.virtual_memory()
    disk         = psutil.disk_usage('/')
    bandwidth, packet_rate = get_bandwidth_and_packets(prev_net)
    curr_net     = psutil.net_io_counters()

    snapshot = {
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # ── Motherboard dataset columns ──
        "CPUUsage"     : cpu_usage,
        "RAMUsage"     : ram.percent,
        "Temperature"  : get_temperature(),
        "DiskUsage"    : round(disk.percent, 2),
        "FanSpeed"     : get_fan_speed(),

        # ── Security dataset columns ──
        "CPU_Usage"    : cpu_usage,
        "Memory_Usage" : ram.percent,
        "Bandwidth"    : bandwidth,
        "Task_Priority": get_task_priority(),
        "Packet_Rate"  : packet_rate,

        # ── Extra context ──
        "RAM_Available_MB"  : round(ram.available / (1024 ** 2), 2),
        "Disk_Free_GB"      : round(disk.free / (1024 ** 3), 2),
        "Process_Count"     : len(psutil.pids()),
    }

    return snapshot, curr_net


# ── Poller Loop ───────────────────────────────────────────────────────────────

def run_collector(duration_seconds: Optional[int] = None, verbose: bool = True):
    """
    Main polling loop.
    Runs forever if duration_seconds is None, else stops after N seconds.
    Appends rows to OUTPUT_CSV.
    """
    print(f"[SIT-Py Collector] Starting — polling every {POLL_INTERVAL}s")
    print(f"[SIT-Py Collector] Output → {OUTPUT_CSV}")
    print(f"[SIT-Py Collector] Platform: {platform.system()} | Sensors: {'ON' if IS_LINUX else 'OFF (Windows)'}")
    print("-" * 60)

    rows         = []
    prev_net     = psutil.net_io_counters()   # seed for delta calc
    start_time   = time.time()
    snapshot_num = 0

    try:
        while True:
            snapshot, prev_net = take_snapshot(prev_net)
            rows.append(snapshot)
            snapshot_num += 1

            if verbose:
                print(f"[{snapshot['timestamp']}] "
                      f"CPU: {snapshot['CPUUsage']}% | "
                      f"RAM: {snapshot['RAMUsage']}% | "
                      f"Disk: {snapshot['DiskUsage']}% | "
                      f"BW: {snapshot['Bandwidth']} kbps | "
                      f"Priority: {snapshot['Task_Priority']}")

            # flush to CSV every 10 snapshots
            if snapshot_num % 10 == 0:
                _flush_to_csv(rows)
                rows = []
                print(f"  → flushed {snapshot_num} rows to CSV")

            # cap check
            if snapshot_num >= MAX_ROWS:
                print(f"[SIT-Py Collector] Hit MAX_ROWS ({MAX_ROWS}), stopping.")
                break

            # duration check
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                print(f"[SIT-Py Collector] Duration reached ({duration_seconds}s), stopping.")
                break

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[SIT-Py Collector] Stopped by user.")

    finally:
        if rows:
            _flush_to_csv(rows)
        print(f"[SIT-Py Collector] Done. Total snapshots: {snapshot_num}")


def _flush_to_csv(rows: list):
    """Appends rows to CSV, writes header only on first write."""
    df          = pd.DataFrame(rows)
    file_exists = os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_collector()