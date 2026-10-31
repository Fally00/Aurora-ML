"""Rule, exclusion, burst, and file hashing tests for Module B."""

import hashlib
from pathlib import Path

from Data.file_monitor import (
    FileChange,
    FileMonitor,
    MonitorConfig,
    classify_file_event,
    is_excluded_path,
)
from Data.eventlog_monitor import LogClearEvent


class MemoryStorage:
    def __init__(self):
        self.alerts = []
        self.observations = []

    def record_alert(self, alert, **kwargs):
        self.alerts.append((alert, kwargs))
        return alert

    def record_file_observation(self, **kwargs):
        self.observations.append(kwargs)
        return len(self.observations)


def _config(root: Path, *, threshold: int = 20) -> MonitorConfig:
    temp_root = root / "Temp"
    startup_root = root / "Startup"
    return MonitorConfig(
        watch_roots=[str(root)],
        startup_roots=[str(startup_root)],
        temp_roots=[str(temp_root)],
        critical_paths=[str(root / "critical.ini")],
        executable_extensions={".exe", ".dll"},
        excluded_directory_names={"cache", "node_modules", "build", "dist", ".git", "steamapps"},
        excluded_path_fragments=[],
        burst_window_seconds=0.05,
        burst_threshold=threshold,
        max_batch_events=100,
        event_queue_size=200,
        cooldown_seconds=60,
    )


def test_exclusion_rules_include_temp_executable_exception(tmp_path):
    config = _config(tmp_path)
    assert is_excluded_path(str(tmp_path / "Cache" / "icon.png"), config)
    assert is_excluded_path(str(tmp_path / "node_modules" / "pkg" / "tool.exe"), config)
    assert is_excluded_path(str(tmp_path / "Temp" / "scratch.txt"), config)
    assert not is_excluded_path(str(tmp_path / "Temp" / "sample.exe"), config)


def test_severity_assignment():
    assert classify_file_event("created", False, False) == "info"
    assert classify_file_event("modified", False, False) == "warning"
    assert classify_file_event("created", True, False) == "warning"
    assert classify_file_event("created", False, True) == "warning"


def test_single_watched_file_edit_is_warning(tmp_path):
    storage = MemoryStorage()
    monitor = FileMonitor(_config(tmp_path), storage=storage, observer_factory=None)
    file_path = tmp_path / "notes.txt"
    file_path.write_text("edited", encoding="utf-8")

    monitor.process_batch([FileChange(str(file_path), "modified", 1.0)])

    assert len(storage.alerts) == 1
    alert = storage.alerts[0][0]
    assert alert.severity == "warning"
    assert alert.signals["path"] == str(file_path)


def test_large_copy_is_one_grouped_info_event(tmp_path):
    storage = MemoryStorage()
    monitor = FileMonitor(_config(tmp_path), storage=storage, observer_factory=None)
    events = [
        FileChange(str(tmp_path / "Downloads" / f"copy-{i}.txt"), "created", float(i))
        for i in range(30)
    ]

    monitor.process_batch(events)

    assert len(storage.alerts) == 1
    alert = storage.alerts[0][0]
    assert alert.severity == "info"
    assert alert.grouped_count == 30


def test_bulk_info_does_not_suppress_startup_warning_or_hash(tmp_path):
    storage = MemoryStorage()
    config = _config(tmp_path)
    monitor = FileMonitor(config, storage=storage, observer_factory=None)
    startup_exe = Path(config.startup_roots[0]) / "harmless.exe"
    startup_exe.parent.mkdir(parents=True)
    content = b"harmless test executable bytes"
    startup_exe.write_bytes(content)
    events = [FileChange(str(startup_exe), "created", 0.0)]
    events.extend(
        FileChange(str(tmp_path / "Downloads" / f"copy-{i}.txt"), "created", float(i + 1))
        for i in range(30)
    )

    monitor.process_batch(events)

    assert sorted(alert.severity for alert, _ in storage.alerts) == ["info", "warning"]
    assert len(storage.observations) == 1
    assert storage.observations[0]["sha256"] == hashlib.sha256(content).hexdigest()


def test_critical_file_baseline_detects_change(tmp_path):
    storage = MemoryStorage()
    config = _config(tmp_path)
    critical = Path(config.critical_paths[0])
    critical.write_text("trusted", encoding="utf-8")
    monitor = FileMonitor(config, storage=storage, observer_factory=None)
    monitor.initialize_critical_baselines()
    baseline = hashlib.sha256(b"trusted").hexdigest()
    critical.write_text("changed", encoding="utf-8")

    monitor.process_batch([FileChange(str(critical), "modified", 2.0)])

    alert = storage.alerts[0][0]
    assert alert.severity == "warning"
    assert alert.signals["critical_baseline_sha256"] == baseline
    assert alert.signals["critical_hash_changed"] is True
    assert storage.observations[0]["event_kind"] == "baseline"


def test_event_queue_is_bounded_and_drops_without_blocking(tmp_path, caplog):
    config = _config(tmp_path)
    config.event_queue_size = 1
    monitor = FileMonitor(config, storage=MemoryStorage(), observer_factory=None)
    event = FileChange(str(tmp_path / "one.txt"), "created", 1.0)

    assert monitor.enqueue(event) is True
    assert monitor.enqueue(event) is False
    assert monitor.dropped_events == 1
    assert sum("event queue is full" in record.message for record in caplog.records) == 1


def test_log_clear_is_critical(tmp_path):
    storage = MemoryStorage()
    monitor = FileMonitor(_config(tmp_path), storage=storage, observer_factory=None)

    monitor._process_log_clear(LogClearEvent(channel="Security", event_id=1102))

    alert, options = storage.alerts[0]
    assert alert.severity == "critical"
    assert alert.signals == {"channel": "Security", "event_id": 1102}
    assert options["dedupe_key"] == "event-log-clear:Security:1102"
