"""Command-line entry point for Aurora telemetry and file monitoring."""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
import threading
from typing import Any, Optional

from config import DB_PATH, MACHINE_ID, VALID_WORKLOAD_TAGS
from Core.schema_v2 import SystemSnapshotV2
from Data import collector_v2
from Data.collector_v2 import run_collector, set_workload_tag
from Data.file_monitor import FileMonitor
from Data.storage import MetricsDB, get_db


LOGGER = logging.getLogger("aurora")


def _print_alert(alert, *, stream=None) -> None:
    print(f"{alert.severity} {alert.timestamp} {alert.title}", file=stream, flush=True)


def _run_collector(
    stop_event: threading.Event,
    quiet: bool,
    failure_event: threading.Event,
) -> None:
    try:
        run_collector(verbose=not quiet, stop_event=stop_event)
    except Exception:
        LOGGER.exception("Collector stopped after an error")
        failure_event.set()
        stop_event.set()


def _start_monitor(monitor: FileMonitor, stop_event: threading.Event) -> None:
    try:
        if not stop_event.is_set():
            monitor.start()
    except Exception as exc:
        LOGGER.warning("Module B could not start; telemetry collection will continue: %s", exc)


def run(tag: str | None = None, quiet: bool = False) -> int:
    """Run telemetry and Module B until interrupted."""
    storage = get_db()
    if tag and not set_workload_tag(tag):
        LOGGER.error("Unsupported workload tag: %s", tag)
        return 2

    console = sys.stdout
    seen_ids = {alert.id for alert in storage.get_alerts(limit=500)}
    stop_event = threading.Event()
    collector_failed = threading.Event()
    monitor = FileMonitor(storage=storage)
    collector_thread = threading.Thread(
        target=_run_collector,
        args=(stop_event, quiet, collector_failed),
        name="aurora-collector",
        daemon=True,
    )
    monitor_thread = threading.Thread(
        target=_start_monitor,
        args=(monitor, stop_event),
        name="aurora-monitor-bootstrap",
        daemon=True,
    )
    collector_thread.start()
    monitor_thread.start()

    try:
        while not collector_failed.is_set():
            for alert in reversed(storage.get_alerts(limit=500)):
                if alert.id not in seen_ids:
                    _print_alert(alert, stream=console)
                    seen_ids.add(alert.id)
            stop_event.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            monitor.stop()
        except Exception as exc:
            LOGGER.warning("Error while stopping Module B: %s", exc)
        monitor_thread.join(timeout=3.0)
        collector_thread.join(timeout=3.0)
        storage.close()
    return 1 if collector_failed.is_set() else 0


def show_alerts(limit: int) -> int:
    """Print recent stored alerts, newest first."""
    for alert in get_db().get_alerts(limit=limit):
        _print_alert(alert)
    return 0


# Number of recent snapshots used to average the rolling bandwidth metrics.
BANDWIDTH_AVERAGE_WINDOW = 60


def _round_optional(value: Optional[float], ndigits: int = 2) -> Optional[float]:
    """Round a nullable float, leaving ``None`` untouched."""
    if value is None:
        return None
    return round(float(value), ndigits)


def _top_gpu_memory_process(snap: SystemSnapshotV2) -> dict[str, Any] | None:
    """Aggregate the top GPU-memory process: pid, cpu %, and GPU memory MB."""
    best = None
    for proc in snap.top_gpu_processes:
        if proc.gpu_memory_mb is None:
            continue
        if best is None or proc.gpu_memory_mb > best.gpu_memory_mb:
            best = proc
    if best is None:
        return None
    return {
        "pid": best.pid,
        "name": best.name,
        "cpu-percent": _round_optional(best.cpu_percent, 2),
        "gpu-memory-mb": _round_optional(best.gpu_memory_mb, 2),
    }


def _bandwidth_averages(
    storage: MetricsDB, machine_id: str
) -> tuple[Optional[float], Optional[float], int]:
    """Average bw_rolling_mean and bw_spike over recent snapshots."""
    recent = storage.query(machine_id=machine_id, limit=BANDWIDTH_AVERAGE_WINDOW)
    means = [s.bw_rolling_mean for s in recent if s.bw_rolling_mean is not None]
    spikes = [s.bw_spike for s in recent if s.bw_spike is not None]
    avg_mean = sum(means) / len(means) if means else None
    avg_spike = sum(spikes) / len(spikes) if spikes else None
    return (
        _round_optional(avg_mean, 2),
        _round_optional(avg_spike, 2),
        len(recent),
    )


def _display_value(value: Optional[float], suffix: str = "", digits: int = 1) -> str:
    """Format a numeric reading or show N/A when that sensor is unavailable."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _database_counts() -> dict[str, int]:
    """Return total telemetry samples, alerts, and file observations."""
    with sqlite3.connect(DB_PATH) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'metrics_[0-9]*'"
            )
            if re.fullmatch(r"metrics_\d{8}", row[0])
        ]
        samples = sum(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables)
        alerts = connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        observations = connection.execute("SELECT COUNT(*) FROM file_observations").fetchone()[0]
    return {"snapshots": samples, "alerts": alerts, "file_observations": observations}


def show_status() -> int:
    """Print the latest snapshot and SQLite counts as a readable console report."""
    storage = get_db()
    latest = storage.get_latest(MACHINE_ID)
    counts = _database_counts()

    print("Aurora status")
    print("=============")
    if latest is None:
        print("No telemetry samples found.")
        print("\nDatabase")
        print(f"  Snapshots:          {counts['snapshots']}")
        print(f"  Alerts recorded:    {counts['alerts']}")
        print(f"  File observations:  {counts['file_observations']}")
        return 0

    bw_rolling_avg, bw_spike_avg, bw_samples = _bandwidth_averages(storage, MACHINE_ID)
    cores = ", ".join(f"{core:.1f}%" for core in latest.cpu_per_core_pct) or "N/A"
    top_process = _top_gpu_memory_process(latest)
    if top_process:
        top_process_text = (
            f"{top_process['name']} (PID {top_process['pid']}; "
            f"{_display_value(top_process['gpu-memory-mb'], ' MB')} GPU memory, "
            f"{_display_value(top_process['cpu-percent'], '%')} CPU)"
        )
    else:
        top_process_text = "N/A"

    print(f"Sample: {latest.timestamp}  |  Machine: {latest.machine_id}  |  Workload: {latest.workload_tag}")
    print("\nCPU")
    print(f"  Usage:              {_display_value(latest.cpu_pct, '%')}")
    print(f"  Per core:           {cores}")
    print(f"  Change:             {_display_value(latest.cpu_delta, '%', 2)}")
    print(
        f"  Rolling average:    {_display_value(latest.cpu_rolling_mean, '%', 2)} "
        f"(std {_display_value(latest.cpu_rolling_std, '%', 2)})"
    )

    print("\nMemory")
    print(f"  Usage:              {_display_value(latest.ram_pct, '%')}")
    print(
        f"  Used / total:       {_display_value(latest.ram_used_mb, ' MB', 2)} / "
        f"{_display_value(latest.ram_total_mb, ' MB', 2)}"
    )
    print(f"  Available:          {_display_value(latest.ram_available_mb, ' MB', 2)}")
    print(f"  Change:             {_display_value(latest.ram_delta, '%', 2)}")
    print(
        f"  Rolling average:    {_display_value(latest.ram_rolling_mean, '%', 2)} "
        f"(std {_display_value(latest.ram_rolling_std, '%', 2)})"
    )

    print("\nDisk")
    print(f"  Usage:              {_display_value(latest.disk_pct, '%')}")
    print(
        f"  Free / total:       {_display_value(latest.disk_free_gb, ' GB', 2)} / "
        f"{_display_value(latest.disk_total_gb, ' GB', 2)}"
    )

    print("\nNetwork")
    print(f"  Bandwidth:          {_display_value(latest.bandwidth_kbps, ' kbps', 2)}")
    print(f"  Packet rate:        {_display_value(latest.packet_rate_pps, ' packets/s', 2)}")
    print(
        f"  Rolling bandwidth: {_display_value(bw_rolling_avg, ' kbps', 2)} "
        f"(average over {bw_samples} samples)"
    )
    print(f"  Average spike:      {_display_value(bw_spike_avg, 'x', 2)}")

    print("\nGPU")
    print(f"  Usage:              {_display_value(latest.gpu_util_pct, '%')}")
    print(
        f"  Memory:             {_display_value(latest.gpu_mem_used_mb, ' MB', 2)} / "
        f"{_display_value(latest.gpu_mem_total_mb, ' MB', 2)}"
    )
    print(f"  Memory usage:       {_display_value(latest.gpu_mem_pct, '%')}")
    print(f"  Temperature:        {_display_value(latest.gpu_temp_c, ' deg C')}")
    print(f"  Fan:                {_display_value(latest.gpu_fan_pct, '%')}")
    print(
        f"  Core / memory clock: {_display_value(latest.gpu_core_clock_mhz, ' MHz', 0)} / "
        f"{_display_value(latest.gpu_mem_clock_mhz, ' MHz', 0)}"
    )
    print(f"  Power limit:        {_display_value(latest.gpu_power_limit_w, ' W')}")
    print(f"  Top memory process: {top_process_text}")

    print("\nCollector")
    print(f"  CPU:                {_display_value(latest.collector_cpu_pct, '%', 2)}")
    print(f"  Memory:             {_display_value(latest.collector_rss_mb, ' MB', 2)}")

    print("\nDatabase")
    print(f"  Snapshots:          {counts['snapshots']}")
    print(f"  Alerts recorded:    {counts['alerts']}")
    print(f"  File observations:  {counts['file_observations']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aurora telemetry and file monitor")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="collect telemetry and monitor files (default)")
    run_parser.add_argument("--tag", choices=sorted(VALID_WORKLOAD_TAGS), help="workload tag")
    run_parser.add_argument("--quiet", action="store_true", help="suppress collector sample output")

    alerts_parser = subparsers.add_parser("alerts", help="print recent alerts from SQLite")
    alerts_parser.add_argument("--limit", type=int, default=20, help="number of recent alerts")
    status_parser = subparsers.add_parser("status", help="print latest snapshot and database counts")

    command_args = list(sys.argv[1:] if argv is None else argv)
    if not command_args or command_args[0].startswith("-"):
        command_args.insert(0, "run")
    args = parser.parse_args(command_args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command in (None, "run"):
        return run(tag=getattr(args, "tag", None), quiet=getattr(args, "quiet", False))
    if args.command == "alerts":
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        return show_alerts(args.limit)
    if args.command == "status":
        return show_status()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
