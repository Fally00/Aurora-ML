"""Command-line entry point for Aurora telemetry and file monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import re
import sqlite3
import sys
import threading
import time
from collections import deque
from typing import Any, Optional

try:
    import msvcrt  # Windows non-blocking keyboard input

    _HAS_MSVCRT = True
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None
    _HAS_MSVCRT = False

from rich.console import Console
from rich.layout import Layout
from rich.live import Live

import aurora_ui

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


def run(
    tag: str | None = None,
    quiet: bool = False,
    *,
    plain: bool = True,
    refresh: float = 1.0,
) -> int:
    """Run telemetry and Module B until interrupted.

    When ``plain`` is False (the default) a btop-style Rich dashboard is
    shown; when True, the legacy headless alert stream is used (tests and
    background use rely on this).
    """
    storage = get_db()
    if tag and not set_workload_tag(tag):
        LOGGER.error("Unsupported workload tag: %s", tag)
        return 2

    if plain:
        return _run_plain(storage, quiet=quiet)

    return _run_dashboard(storage, refresh=refresh)


def _run_plain(storage: MetricsDB, quiet: bool = False) -> int:
    """Legacy headless mode: print one alert per line to the console."""
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


# ── Dashboard logging ───────────────────────────────────────────────────

class _LogCaptureHandler(logging.Handler):
    """Captures recent log records so WARNING+ lines can be shown in the UI."""

    def __init__(self, capacity: int = 80):
        super().__init__(level=logging.WARNING)
        self.records: deque[logging.LogRecord] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


_LOG_CAPTURE: Optional[_LogCaptureHandler] = None


def _setup_dashboard_logging() -> None:
    """Route logging to a rotating file in the data dir (never the console).

    Configures the root logger so every module (aurora, collector, Module B,
    event log) writes to the file; the capture handler keeps recent WARNING+
    lines for the dashboard strip.
    """
    global _LOG_CAPTURE
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    log_path = os.path.join(os.path.dirname(DB_PATH), "aurora.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    capture = _LogCaptureHandler()
    root.addHandler(capture)
    _LOG_CAPTURE = capture
    logging.getLogger("aurora").setLevel(logging.WARNING)


def _recent_warnings() -> list[str]:
    """Return the most recent captured WARNING+ log lines."""
    if _LOG_CAPTURE is None:
        return []
    lines = []
    for record in list(_LOG_CAPTURE.records)[-6:]:
        lines.append(f"{record.levelname[0]} {record.getMessage()[:60]}")
    return lines


# ── Keyboard input ──────────────────────────────────────────────────────

def _get_key() -> Optional[str]:
    """Non-blocking single-key read. Returns 'UP'/'DOWN' for arrows, else the char."""
    if not _HAS_MSVCRT or msvcrt is None:
        return None
    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getch()
    if isinstance(ch, bytes):
        ch = ch.decode("utf-8", errors="replace")
    if ch in ("\x00", "\xe0"):  # special key prefix (arrow / function keys)
        if msvcrt.kbhit():
            ch2 = msvcrt.getch()
            if isinstance(ch2, bytes):
                ch2 = ch2.decode("utf-8", errors="replace")
            return {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}.get(ch2)
        return None
    return ch


def _resolve_selected(
    alerts: list[Any], scroll: int, selected: Optional[int]
) -> Optional[Any]:
    """Resolve the currently selected alert (newest-first order, scroll-adjusted)."""
    if selected is None:
        return None
    ordered = list(reversed(alerts))
    visible = ordered[scroll:]
    if selected < 0 or selected >= len(visible):
        return None
    return visible[selected]


# ── Dashboard ───────────────────────────────────────────────────────────

def _run_dashboard(storage: MetricsDB, refresh: float) -> int:
    """btop-style Rich Live dashboard. Reads from SQLite; collector/Module B
    threads keep running as in plain mode."""
    console = Console()
    layout = _build_layout(console)

    stop_event = threading.Event()
    collector_failed = threading.Event()
    monitor = FileMonitor(storage=storage)
    start_time = time.monotonic()

    collector_thread = threading.Thread(
        target=_run_collector,
        args=(stop_event, True, collector_failed),  # dashboard: quiet collector
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

    # Rolling history (oldest first) for sparklines.
    history: deque[SystemSnapshotV2] = deque(maxlen=60)
    scroll = 0
    selected: Optional[int] = None

    live = Live(
        layout,
        console=console,
        screen=True,
        auto_refresh=False,
        refresh_per_second=4,
        redirect_stdout=False,
        redirect_stderr=False,
    )

    def render() -> None:
        nonlocal scroll, selected
        latest = storage.get_latest(MACHINE_ID)
        if latest and (not history or history[-1].timestamp_ns != latest.timestamp_ns):
            history.append(latest)

        alerts = list(storage.get_alerts(limit=200))
        ordered = list(reversed(alerts))
        max_scroll = max(0, len(ordered) - 1)
        scroll = max(0, min(scroll, max_scroll))
        if selected is not None and selected >= max(0, len(ordered) - scroll):
            selected = None

        panel_w = max(20, console.width // 2 - 2)
        warnings = _recent_warnings()
        uptime_seconds = int(time.monotonic() - start_time)
        uptime = f"{uptime_seconds // 3600}:{(uptime_seconds % 3600) // 60:02d}:{uptime_seconds % 60:02d}"
        workload = latest.workload_tag if latest else "idle"
        timestamp = latest.timestamp if latest else "--"

        layout["header"].update(
            aurora_ui.header_panel(
                MACHINE_ID,
                workload,
                timestamp,
                uptime,
                monitor.watched_root_count,
                monitor.event_log_active,
                latest.collector_cpu_pct if latest else None,
                latest.collector_rss_mb if latest else None,
                width=console.width,
            )
        )
        layout["cpu"].update(aurora_ui.cpu_panel(latest, history, panel_w))
        layout["gpu"].update(aurora_ui.gpu_panel(latest, panel_w))
        layout["mem"].update(aurora_ui.memory_panel(latest, panel_w))
        layout["disk"].update(aurora_ui.disk_panel(latest, panel_w))
        layout["net"].update(aurora_ui.network_panel(latest, history, panel_w))
        layout["processes"].update(aurora_ui.processes_panel(latest, panel_w))
        layout["alerts"].update(
            aurora_ui.alerts_panel(
                alerts,
                panel_w,
                max(6, console.height - 10),
                scroll=scroll,
                selected=selected,
            )
        )
        layout["footer"].update(aurora_ui.footer_panel())

    render()
    live.update(layout)

    try:
        with live:
            last_update = 0.0
            while not stop_event.is_set() and not collector_failed.is_set():
                now = time.monotonic()
                if now - last_update >= max(0.1, refresh):
                    render()
                    live.update(layout)
                    last_update = now

                key = _get_key()
                if key is not None:
                    if key == "q":
                        break
                    elif key == "\t":  # tab: cycle focus
                        selected = None
                    elif key == "UP":
                        scroll = max(0, scroll - 1)
                    elif key == "DOWN":
                        scroll += 1
                    elif key in ("n", "d"):
                        alert = _resolve_selected(list(storage.get_alerts(limit=200)), scroll, selected)
                        if alert is None:
                            continue
                        if key == "n":
                            storage.mark_alert_normal(alert.id)
                        else:
                            storage.dismiss_alert(alert.id)
                        selected = None

                time.sleep(0.05)
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


def _build_layout(console: Console) -> Layout:
    """Build the responsive dashboard layout (degrades under 80x24)."""
    root = Layout(name="root")
    root.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=1),
    )
    root["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    root["left"].split(
        Layout(name="cpu", ratio=1),
        Layout(name="gpu", ratio=1),
        Layout(name="mem", ratio=1),
        Layout(name="disk", ratio=1),
        Layout(name="net", ratio=1),
    )
    root["right"].split(
        Layout(name="processes", ratio=1),
        Layout(name="alerts", ratio=2),
    )
    return root
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
    run_parser.add_argument(
        "--plain",
        action="store_true",
        help="headless mode: print one alert per line (no dashboard)",
    )
    run_parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="dashboard refresh interval in seconds (default: 1)",
    )

    alerts_parser = subparsers.add_parser("alerts", help="print recent alerts from SQLite")
    alerts_parser.add_argument("--limit", type=int, default=20, help="number of recent alerts")
    status_parser = subparsers.add_parser("status", help="print latest snapshot and database counts")

    command_args = list(sys.argv[1:] if argv is None else argv)
    if not command_args or command_args[0].startswith("-"):
        command_args.insert(0, "run")
    args = parser.parse_args(command_args)
    if args.command in (None, "run") and not args.plain:
        # Dashboard mode: logging goes to a rotating file, never the console.
        _setup_dashboard_logging()
        return run(
            tag=getattr(args, "tag", None),
            quiet=True,
            plain=False,
            refresh=args.refresh,
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command in (None, "run"):
        return run(
            tag=getattr(args, "tag", None),
            quiet=getattr(args, "quiet", False),
            plain=getattr(args, "plain", False),
            refresh=args.refresh,
        )
    if args.command == "alerts":
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        return show_alerts(args.limit)
    if args.command == "status":
        return show_status()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
