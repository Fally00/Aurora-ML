"""Command-line entry point for Aurora telemetry and file monitoring."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
from dataclasses import asdict

from config import DB_PATH, MACHINE_ID, VALID_WORKLOAD_TAGS
from Core.schema_v2 import SystemSnapshotV2
from Data import collector_v2
from Data.collector_v2 import run_collector, set_workload_tag
from Data.file_monitor import FileMonitor
from Data.storage import get_db


LOGGER = logging.getLogger("aurora")
# The protected legacy collector's startup message reads DB_PATH as a module
# global even though it is not imported there. Supply the configured path here.
collector_v2.DB_PATH = DB_PATH


def _print_alert(alert, *, stream=None) -> None:
    print(f"{alert.severity} {alert.timestamp} {alert.title}", file=stream, flush=True)


def _run_collector(
    stop_event: threading.Event,
    quiet: bool,
    failure_event: threading.Event,
) -> None:
    try:
        if quiet:
            with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
                run_collector(verbose=False, stop_event=stop_event)
        else:
            run_collector(verbose=True, stop_event=stop_event)
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
    """Print the latest snapshot and current SQLite record counts."""
    latest = get_db().get_latest(MACHINE_ID)
    snapshot = asdict(latest) if isinstance(latest, SystemSnapshotV2) else None
    print(json.dumps({"latest_snapshot": snapshot, "db_counts": _database_counts()}, indent=2))
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
