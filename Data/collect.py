"""
Data/collect.py - SIT-Py Live System Metric Collector
======================================================
Captures real-time system metrics via psutil and returns a SystemSnapshot.

Key changes from v1:
  - Returns SystemSnapshot (schema-aligned dataclass) instead of raw dict
  - No more dual CPUUsage / CPU_Usage keys - schema is now unified
  - Writes live_metrics.csv using unified column names
  - Task priority now derived from both nice values AND resource usage
"""
from __future__ import annotations

import os
import sys
import platform
import time
from datetime import datetime
from typing import Any, Optional, Tuple

import psutil
import pandas as pd

# ── Path bootstrap (allow running as script or imported from parent) ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Core.schema import SystemSnapshot
from config import (
    POLL_INTERVAL, LIVE_CSV, MAX_CSV_ROWS,
    WORKLOAD_THRESHOLDS,
)

IS_LINUX = platform.system() == "Linux"


# ── Hardware sensor helpers ─────────────────────────────────────────────────

def _get_temperature() -> Optional[float]:
    """Average CPU temperature in °C. None on Windows / unsupported systems."""
    if not IS_LINUX:
        return None
    try:
        temps_fn = getattr(psutil, "sensors_temperatures", None)
        if not temps_fn:
            return None
        temps = temps_fn()
        if not temps:
            return None
        for entries in temps.values():
            vals = [getattr(e, "current", None) for e in entries]
            vals = [v for v in vals if v is not None]
            if vals:
                return round(sum(vals) / len(vals), 2)
    except Exception:
        return None
    return None


def _get_fan_speed() -> Optional[int]:
    """First available fan speed in RPM. None on Windows / unsupported."""
    if not IS_LINUX:
        return None
    try:
        fans_fn = getattr(psutil, "sensors_fans", None)
        if not fans_fn:
            return None
        fans = fans_fn()
        if not fans:
            return None
        for entries in fans.values():
            if entries:
                val = getattr(entries[0], "current", None)
                return int(val) if val is not None else None
    except Exception:
        return None
    return None


# ── Derived metrics ─────────────────────────────────────────────────────────

def _get_task_priority(cpu_pct: float, ram_pct: float) -> str:
    """
    Derives workload priority from BOTH process nice values AND resource usage.
    Falls back to resource thresholds if nice-value collection fails.
    This gives a more accurate label for live training data.
    """
    try:
        nice_values = []
        for proc in psutil.process_iter(["nice"]):
            try:
                n = proc.info["nice"]
                if n is not None:
                    nice_values.append(n)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if nice_values:
            avg_nice = sum(nice_values) / len(nice_values)
            # Unix: lower nice = higher priority; Windows uses a different scale
            if avg_nice <= 0:
                return "Critical"
            elif avg_nice <= 5:
                return "High"
            elif avg_nice <= 15:
                return "Medium"
            else:
                return "Low"
    except Exception:
        pass

    # Fallback: resource-threshold labels
    t = WORKLOAD_THRESHOLDS
    if cpu_pct > t["Critical"]["cpu"] or ram_pct > t["Critical"]["ram"]:
        return "Critical"
    elif cpu_pct > t["High"]["cpu"] or ram_pct > t["High"]["ram"]:
        return "High"
    elif cpu_pct > t["Medium"]["cpu"] or ram_pct > t["Medium"]["ram"]:
        return "Medium"
    return "Low"


def _get_bandwidth_and_packets(
    prev_net, elapsed: float = POLL_INTERVAL
) -> Tuple[float, float]:
    """Returns (bandwidth_kbps, packet_rate_pps) since last poll."""
    curr_net = psutil.net_io_counters()
    if prev_net is None:
        return 0.0, 0.0

    bytes_delta   = ((curr_net.bytes_sent   + curr_net.bytes_recv) -
                     (prev_net.bytes_sent   + prev_net.bytes_recv))
    packets_delta = ((curr_net.packets_sent + curr_net.packets_recv) -
                     (prev_net.packets_sent + prev_net.packets_recv))

    t = max(elapsed, 0.001)   # guard against div/0
    bandwidth_kbps  = round((bytes_delta * 8) / (t * 1000), 2)
    packet_rate_pps = round(packets_delta / t, 2)
    return bandwidth_kbps, packet_rate_pps


# ── Main Snapshot ───────────────────────────────────────────────────────────

def take_snapshot(prev_net, elapsed: float = POLL_INTERVAL) -> Tuple[SystemSnapshot, Any]:
    """
    Captures a full system snapshot.

    Args:
        prev_net: previous psutil.net_io_counters() for delta calculation.
        elapsed:  seconds since last snapshot (for accurate rate calculation).

    Returns:
        (SystemSnapshot, current net_io_counters for next call)
    """
    cpu_pct   = psutil.cpu_percent(interval=None)
    ram       = psutil.virtual_memory()
    disk      = psutil.disk_usage("/")
    bw, pkts  = _get_bandwidth_and_packets(prev_net, elapsed)
    curr_net  = psutil.net_io_counters()

    snap = SystemSnapshot(
        timestamp        = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cpu_pct          = cpu_pct,
        ram_pct          = ram.percent,
        disk_pct         = round(disk.percent, 2),
        bandwidth_kbps   = bw,
        packet_rate_pps  = pkts,
        temperature_c    = _get_temperature(),
        fan_rpm          = _get_fan_speed(),
        process_count    = len(psutil.pids()),
        task_priority    = _get_task_priority(cpu_pct, ram.percent),
        ram_available_mb = round(ram.available / (1024 ** 2), 2),
        disk_free_gb     = round(disk.free / (1024 ** 3), 2),
    )
    return snap, curr_net


# ── Collector Loop ───────────────────────────────────────────────────────────

def run_collector(duration_seconds: Optional[int] = None, verbose: bool = True):
    """
    Main polling loop. Writes to live_metrics.csv using unified schema columns.
    Runs indefinitely if duration_seconds is None, else stops after N seconds.
    """
    print(f"[Collector] Starting - polling every {POLL_INTERVAL}s")
    print(f"[Collector] Output  -> {LIVE_CSV}")
    print(f"[Collector] Platform: {platform.system()} | Sensors: {'ON' if IS_LINUX else 'OFF'}")
    print("-" * 60)

    rows:      list  = []
    prev_net         = psutil.net_io_counters()
    start_time       = time.time()
    last_poll        = start_time
    snapshot_num: int = 0

    try:
        while True:
            now     = time.time()
            elapsed = now - last_poll
            last_poll = now

            snap, prev_net = take_snapshot(prev_net, elapsed)
            rows.append(snap.to_live_csv_row())
            snapshot_num += 1

            if verbose:
                print(
                    f"[{snap.timestamp}]  "
                    f"CPU: {snap.cpu_pct:5.1f}%  "
                    f"RAM: {snap.ram_pct:5.1f}%  "
                    f"Disk: {snap.disk_pct:5.1f}%  "
                    f"BW: {snap.bandwidth_kbps:8.1f} kbps  "
                    f"Priority: {snap.task_priority}"
                )

            # Flush every 10 rows
            if snapshot_num % 10 == 0:
                _flush_to_csv(rows)
                rows = []
                print(f"  -> flushed {snapshot_num} rows to {LIVE_CSV}")

            if snapshot_num >= MAX_CSV_ROWS:
                print(f"[Collector] Hit MAX_ROWS ({MAX_CSV_ROWS}), stopping.")
                break

            if duration_seconds and (now - start_time) >= duration_seconds:
                print(f"[Collector] Duration reached ({duration_seconds}s), stopping.")
                break

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Collector] Stopped by user.")
    finally:
        if rows:
            _flush_to_csv(rows)
        print(f"[Collector] Done. Total snapshots: {snapshot_num}")


def _flush_to_csv(rows: list):
    """Appends rows to LIVE_CSV; writes header only on first write."""
    df          = pd.DataFrame(rows)
    file_exists = os.path.exists(LIVE_CSV)
    df.to_csv(LIVE_CSV, mode="a", header=not file_exists, index=False)


# ── Entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIT-Py Live Metric Collector")
    parser.add_argument("--duration", type=int, default=None,
                        help="Stop after N seconds (default: run forever)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-snapshot output")
    args = parser.parse_args()
    run_collector(duration_seconds=args.duration, verbose=not args.quiet)