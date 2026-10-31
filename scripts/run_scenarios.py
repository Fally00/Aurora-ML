"""Run safe, isolated Module B scenarios using real filesystem notifications."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Core.alerts import Alert  # noqa: E402
from Data.eventlog_monitor import LogClearEvent  # noqa: E402
from Data.file_monitor import FileMonitor, MonitorConfig, Observer  # noqa: E402
from Data.storage import MetricsDB  # noqa: E402

ALL_SCENARIOS = ("watched-edit", "startup-executable", "bulk-copy", "log-clear")
LOG_CLEAR_SCENARIO = "log-clear"


def validate_scenario_request(scenarios: list[str], is_test_vm: bool) -> None:
    """Guard the privileged log-clear scenario; the runner only simulates it."""
    if LOG_CLEAR_SCENARIO in scenarios and not is_test_vm:
        raise PermissionError(
            "log-clear scenario refused: pass --i-am-on-a-test-vm to explicitly enable "
            "the simulated Security 1102 event"
        )


def _new_alerts(storage: MetricsDB, before_ids: set[str]) -> list[Alert]:
    return [alert for alert in storage.get_alerts(limit=1000) if alert.id not in before_ids]


def _wait_for_scenario(
    storage: MetricsDB,
    before_ids: set[str],
    *,
    severity: str,
    title_prefix: str,
    timeout_seconds: float,
    quiet_seconds: float,
) -> list[Alert]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        alerts = _new_alerts(storage, before_ids)
        if any(a.severity == severity and a.title.startswith(title_prefix) for a in alerts):
            # Let the OS observer deliver the remainder of a burst before checking extras.
            time.sleep(quiet_seconds)
            return _new_alerts(storage, before_ids)
        time.sleep(0.1)
    return _new_alerts(storage, before_ids)


def _scenario_record(
    name: str,
    before_ids: set[str],
    alerts: list[Alert],
    expected_severity: str,
    title_prefix: str,
) -> dict:
    matched = [
        a.id for a in alerts
        if a.module == "module_b" and a.severity == expected_severity and a.title.startswith(title_prefix)
    ]
    return {
        "scenario": name,
        "expected_count": 1,
        "expected_severity": expected_severity,
        "expected_title_prefix": title_prefix,
        "alert_ids": [a.id for a in alerts],
        "matched_alert_ids": matched[:1],
        "started_with_alert_ids": sorted(before_ids),
        "passed": len(matched) == 1 and len(alerts) == 1,
        "alerts": [
            {"id": a.id, "severity": a.severity, "title": a.title, "grouped_count": a.grouped_count}
            for a in alerts
        ],
    }


def run_scenarios(
    output_root: Path,
    *,
    scenarios: list[str] | None = None,
    timeout_seconds: float = 15.0,
    include_log_clear: bool = False,
) -> tuple[Path, dict]:
    """Run selected scenarios in a disposable watched tree and a new SQLite DB."""
    selected = scenarios or list(ALL_SCENARIOS)
    validate_scenario_request(selected, include_log_clear)
    if Observer is None:
        raise RuntimeError("watchdog is required; install the dependencies from requirements.txt")

    run_id = uuid4().hex[:12]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    watched_root = run_dir / "watched"
    startup_root = run_dir / "startup"
    downloads = watched_root / "Downloads"
    watched_root.mkdir()
    startup_root.mkdir()
    downloads.mkdir()
    (watched_root / "edit-me.txt").write_text("before\n", encoding="utf-8")

    db_path = run_dir / "alerts.sqlite"
    storage = MetricsDB(str(db_path))
    config = MonitorConfig(
        watch_roots=[str(watched_root)],
        startup_roots=[str(startup_root)],
        temp_roots=[],
        critical_paths=[],
        executable_extensions={".exe", ".dll", ".sys", ".scr", ".com", ".msi"},
        excluded_directory_names={"cache", "node_modules", "build", "dist", ".git"},
        excluded_path_fragments=[],
        burst_window_seconds=1.0,
        burst_threshold=25,
        max_batch_events=2048,
        event_queue_size=4096,
        cooldown_seconds=5,
    )
    monitor = FileMonitor(config, storage=storage)
    started = time.monotonic()
    run_started_at = datetime.now(timezone.utc).isoformat()
    scenario_records: list[dict] = []
    try:
        monitor.start(watch_event_logs=False)
        time.sleep(0.25)  # Allow the native observer to finish scheduling the sandbox roots.

        if "watched-edit" in selected:
            before = {a.id for a in storage.get_alerts(limit=1000)}
            edit_file = watched_root / "edit-me.txt"
            edit_file.write_text("after\n", encoding="utf-8")
            alerts = _wait_for_scenario(
                storage, before, severity="warning", title_prefix="Watched file modified",
                timeout_seconds=timeout_seconds, quiet_seconds=config.burst_window_seconds + 0.1,
            )
            scenario_records.append(_scenario_record("watched-edit", before, alerts, "warning", "Watched file modified"))

        if "startup-executable" in selected:
            before = {a.id for a in storage.get_alerts(limit=1000)}
            # Harmless data stub: Module B only hashes it and never executes it.
            (startup_root / "aurora-harmless-test.exe").write_bytes(b"Aurora scenario stub; do not execute\n")
            alerts = _wait_for_scenario(
                storage, before, severity="warning", title_prefix="Protected or startup path changed",
                timeout_seconds=timeout_seconds, quiet_seconds=config.burst_window_seconds + 0.1,
            )
            scenario_records.append(_scenario_record("startup-executable", before, alerts, "warning", "Protected or startup path changed"))

        if "bulk-copy" in selected:
            source = run_dir / "copy-source"
            source.mkdir()
            for index in range(100):
                (source / f"harmless-{index:03d}.txt").write_text(f"scenario row {index}\n", encoding="utf-8")
            before = {a.id for a in storage.get_alerts(limit=1000)}
            destination = downloads / "bulk-test"
            shutil.copytree(source, destination)
            alerts = _wait_for_scenario(
                storage, before, severity="info", title_prefix="Grouped file activity",
                timeout_seconds=timeout_seconds, quiet_seconds=config.burst_window_seconds + 0.1,
            )
            grouped = [a for a in alerts if a.title.startswith("Grouped file activity")]
            expected = len(grouped) == 1 and grouped[0].grouped_count >= config.burst_threshold
            record = _scenario_record("bulk-copy", before, alerts, "info", "Grouped file activity")
            record["passed"] = expected and len(alerts) == 1 and all(a.severity == "info" for a in alerts)
            record["expected_grouped_count_min"] = config.burst_threshold
            scenario_records.append(record)

        if LOG_CLEAR_SCENARIO in selected:
            before = {a.id for a in storage.get_alerts(limit=1000)}
            # Never clear an operating-system log; exercise the same queued critical path synthetically.
            monitor.enqueue(LogClearEvent(channel="Security", event_id=1102))
            alerts = _wait_for_scenario(
                storage, before, severity="critical", title_prefix="Windows Security event log cleared",
                timeout_seconds=timeout_seconds, quiet_seconds=0.1,
            )
            scenario_records.append(_scenario_record("log-clear", before, alerts, "critical", "Windows Security event log cleared"))
    finally:
        monitor.stop()
        duration_seconds = max(time.monotonic() - started, 0.001)
        storage.close()

    manifest = {
        "run_id": run_id,
        "started_at": run_started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "simulated_log_clear": LOG_CLEAR_SCENARIO in selected,
        "scenarios": scenario_records,
        "passed": all(row["passed"] for row in scenario_records) and len(scenario_records) == len(selected),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=ALL_SCENARIOS, help="Scenario to run; repeat to select several (default: all).")
    parser.add_argument("--output-root", type=Path, default=_ROOT / "Data" / "scenario_runs")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for each expected alert.")
    parser.add_argument(
        "--i-am-on-a-test-vm", action="store_true",
        help="Explicitly enable the simulated log-clear test scenario; no real logs are cleared.",
    )
    args = parser.parse_args(argv)
    selected = args.scenario or list(ALL_SCENARIOS)
    if LOG_CLEAR_SCENARIO in selected:
        print(
            "WARNING: this run includes a simulated Security event 1102 scenario. "
            "It is refused without --i-am-on-a-test-vm and never clears an OS event log.",
            file=sys.stderr,
            flush=True,
        )
    try:
        validate_scenario_request(selected, args.i_am_on_a_test_vm)
        run_dir, result = run_scenarios(
            args.output_root,
            scenarios=selected,
            timeout_seconds=max(args.timeout, 1.0),
            include_log_clear=args.i_am_on_a_test_vm,
        )
    except (PermissionError, RuntimeError, OSError) as exc:
        print(f"Scenario run refused/failed: {exc}", file=sys.stderr)
        return 2
    print(f"Scenario output: {run_dir}")
    for row in result["scenarios"]:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['scenario']}: {row['alerts']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
