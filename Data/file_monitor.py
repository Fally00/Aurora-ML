"""Passive OS-notification watcher for user files and startup locations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from Core.alerts import Alert
from Data.storage import MetricsDB, get_db
from Data.eventlog_monitor import LogClearEvent, WindowsEventLogListener

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # Allows config/rule tests to run before optional deps are installed.
    Observer = None

    class FileSystemEventHandler:  # type: ignore[no-redef]
        pass


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("watcher_config.json")


@dataclass(frozen=True, slots=True)
class FileChange:
    """Small immutable event copied out of the OS callback."""

    path: str
    kind: str
    timestamp: float


@dataclass(slots=True)
class MonitorConfig:
    """Resolved watcher paths and bounded event-processing settings."""

    watch_roots: list[str]
    startup_roots: list[str]
    temp_roots: list[str]
    critical_paths: list[str]
    executable_extensions: set[str]
    excluded_directory_names: set[str]
    excluded_path_fragments: list[str]
    burst_window_seconds: float = 2.0
    burst_threshold: int = 25
    max_batch_events: int = 2048
    event_queue_size: int = 4096
    cooldown_seconds: int = 60

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "MonitorConfig":
        """Load JSON settings and expand user/environment path variables."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        resolve = _resolve_path
        return cls(
            watch_roots=_resolve_list(raw.get("watch_roots", []), resolve),
            startup_roots=_resolve_list(raw.get("startup_roots", []), resolve),
            temp_roots=_resolve_list(raw.get("temp_roots", []), resolve),
            critical_paths=_resolve_list(raw.get("critical_paths", []), resolve),
            executable_extensions={str(x).lower() for x in raw.get("executable_extensions", [])},
            excluded_directory_names={str(x).strip().lower() for x in raw.get("excluded_directory_names", [])},
            excluded_path_fragments=[os.path.normcase(str(x)).lower() for x in raw.get("excluded_path_fragments", [])],
            burst_window_seconds=max(float(raw.get("burst_window_seconds", 2.0)), 0.05),
            burst_threshold=max(int(raw.get("burst_threshold", 25)), 2),
            max_batch_events=max(int(raw.get("max_batch_events", 2048)), 1),
            event_queue_size=max(int(raw.get("event_queue_size", 4096)), 1),
            cooldown_seconds=max(int(raw.get("cooldown_seconds", 60)), 0),
        )

    @property
    def all_roots(self) -> list[str]:
        """Unique observation roots, including critical-file parents and temp."""
        roots = self.watch_roots + self.startup_roots + self.temp_roots
        roots += [os.path.dirname(path) for path in self.critical_paths]
        return list(dict.fromkeys(os.path.abspath(path) for path in roots if path))


def _resolve_path(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    # Windows-style %VARS% remain unexpanded on non-Windows test hosts.
    if os.name != "nt":
        import re
        expanded = re.sub(r"%([^%]+)%", lambda m: os.environ.get(m.group(1), m.group(0)), expanded)
    return os.path.abspath(expanded)


def _resolve_list(values: list[str], resolver: Callable[[str], str]) -> list[str]:
    return [resolver(str(value)) for value in values if str(value).strip()]


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
    except (ValueError, OSError):
        return False


def is_executable_path(path: str, config: MonitorConfig) -> bool:
    """Check configured executable suffixes without touching disk."""
    return Path(path).suffix.lower() in config.executable_extensions


def is_excluded_path(path: str, config: MonitorConfig) -> bool:
    """Apply configured cache/build/game exclusions, except temp executables."""
    normalized = os.path.normcase(os.path.abspath(path)).lower()
    in_temp = any(_within(path, root) for root in config.temp_roots)
    if in_temp:
        return not is_executable_path(path, config)

    if any(fragment and fragment in normalized for fragment in config.excluded_path_fragments):
        return True
    components = {part.strip().lower() for part in Path(normalized).parts}
    return bool(components & config.excluded_directory_names)


def is_startup_path(path: str, config: MonitorConfig) -> bool:
    return any(_within(path, root) for root in config.startup_roots)


def is_critical_path(path: str, config: MonitorConfig) -> bool:
    normalized = os.path.normcase(os.path.abspath(path)).lower()
    return any(normalized == os.path.normcase(os.path.abspath(item)).lower() for item in config.critical_paths)


class _WatchHandler(FileSystemEventHandler):
    """OS callback adapter; only normalizes paths and enqueues tiny records."""

    def __init__(self, enqueue: Callable[[FileChange], bool]):
        self._enqueue = enqueue

    def on_created(self, event: Any) -> None:
        if not event.is_directory:
            self._enqueue(FileChange(event.src_path, "created", time.time()))

    def on_modified(self, event: Any) -> None:
        if not event.is_directory:
            self._enqueue(FileChange(event.src_path, "modified", time.time()))

    def on_deleted(self, event: Any) -> None:
        if not event.is_directory:
            self._enqueue(FileChange(event.src_path, "deleted", time.time()))

    def on_moved(self, event: Any) -> None:
        if not event.is_directory:
            self._enqueue(FileChange(event.dest_path, "moved", time.time()))


class FileMonitor:
    """Watch configured paths using OS change notifications and persist alerts."""

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        *,
        storage: Optional[MetricsDB] = None,
        observer_factory: Any = None,
        event_log_listener: Any = None,
    ):
        self.config = config or MonitorConfig.from_file()
        self.storage = storage or get_db()
        self._observer_factory = observer_factory if observer_factory is not None else Observer
        self._event_log_listener = event_log_listener
        self._events: queue.Queue[FileChange | LogClearEvent] = queue.Queue(
            maxsize=self.config.event_queue_size
        )
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._observer: Any = None
        self._dropped_events = 0
        self._warned_queue_full = False
        self._critical_baselines: dict[str, Optional[str]] = {}

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def watched_root_count(self) -> int:
        """Number of configured watch roots (filesystem locations observed)."""
        return len(self.config.all_roots)

    @property
    def event_log_active(self) -> bool:
        """Whether Windows Event Log subscriptions are running (needs admin)."""
        listener = self._event_log_listener
        return listener is not None and bool(getattr(listener, "_threads", None))

    def enqueue(self, event: FileChange | LogClearEvent) -> bool:
        """Queue an event without blocking the OS callback thread."""
        try:
            self._events.put_nowait(event)
            return True
        except queue.Full:
            self._dropped_events += 1
            if not self._warned_queue_full:
                LOGGER.warning("Module B event queue is full; some file events are being dropped")
                self._warned_queue_full = True
            return False

    def start(self, *, watch_files: bool = True, watch_event_logs: bool = True) -> None:
        """Start bounded processing, filesystem notifications, and log subscriptions."""
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="aurora-file-monitor", daemon=True)
        self._worker.start()

        if watch_files:
            self.initialize_critical_baselines()
            if self._observer_factory is None:
                LOGGER.error("watchdog is not installed; Module B file monitoring is disabled")
            else:
                self._observer = self._observer_factory()
                handler = _WatchHandler(self.enqueue)
                scheduled = set()
                for root in self.config.all_roots:
                    if root in scheduled or not os.path.isdir(root):
                        continue
                    scheduled.add(root)
                    self._observer.schedule(handler, root, recursive=True)
                self._observer.start()
                LOGGER.info("Module B watching %d filesystem roots", len(scheduled))

        if watch_event_logs:
            if self._event_log_listener is None:
                self._event_log_listener = WindowsEventLogListener(self.enqueue)
            self._event_log_listener.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watchers and drain queued work for up to timeout seconds."""
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=timeout)
            self._observer = None
        if self._event_log_listener is not None:
            self._event_log_listener.stop(timeout=timeout)
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def _worker_loop(self) -> None:
        while not self._stop.is_set() or not self._events.empty():
            try:
                first = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(first, LogClearEvent):
                self._process_log_clear(first)
                continue

            batch: list[FileChange] = [first]
            deadline = time.monotonic() + self.config.burst_window_seconds
            while len(batch) < self.config.max_batch_events:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_event = self._events.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(next_event, LogClearEvent):
                    self._process_batch(batch)
                    batch = []
                    self._process_log_clear(next_event)
                    deadline = time.monotonic() + self.config.burst_window_seconds
                    continue
                batch.append(next_event)
            if batch:
                self._process_batch(batch)

    def process_batch(self, events: list[FileChange]) -> None:
        """Public synchronous entry point for deterministic tests and replay."""
        self._process_batch(events)

    def initialize_critical_baselines(self) -> None:
        """Record configured critical-file hashes once before watching changes."""
        for path in self.config.critical_paths:
            digest, size, error = _hash_stable_file(path)
            self._critical_baselines[os.path.normcase(path).lower()] = digest
            if digest is not None or error is not None:
                self.storage.record_file_observation(
                    file_path=path,
                    event_kind="baseline",
                    sha256=digest,
                    file_size=size,
                    error=error,
                )

    def _process_log_clear(self, event: LogClearEvent) -> None:
        self.storage.record_alert(
            Alert(
                module="module_b",
                severity="critical",
                title=f"Windows {event.channel} event log cleared",
                signals={"channel": event.channel, "event_id": event.event_id},
            ),
            dedupe_key=f"event-log-clear:{event.channel}:{event.event_id}",
            cooldown_seconds=self.config.cooldown_seconds,
        )

    def _process_batch(self, events: list[FileChange]) -> None:
        by_path: dict[str, set[str]] = {}
        for event in events:
            path = os.path.abspath(event.path)
            if is_excluded_path(path, self.config):
                continue
            by_path.setdefault(path, set()).add(event.kind)
        if not by_path:
            return

        observations: dict[str, dict[str, Any]] = {}
        for path, kinds in by_path.items():
            if is_executable_path(path, self.config) or is_critical_path(path, self.config):
                kind = _preferred_kind(kinds)
                digest, size, error = _hash_stable_file(path)
                critical = is_critical_path(path, self.config)
                baseline = self._critical_baselines.get(os.path.normcase(path).lower()) if critical else None
                observations[path] = {
                    "sha256": digest,
                    "file_size": size,
                    "error": error,
                    "critical_baseline_sha256": baseline,
                    "critical_hash_changed": critical and baseline != digest,
                }
                self.storage.record_file_observation(
                    file_path=path,
                    event_kind=kind,
                    sha256=digest,
                    file_size=size,
                    error=error,
                )

        protected_paths = {
            path for path in by_path
            if is_startup_path(path, self.config) or is_critical_path(path, self.config)
        }
        ordinary_paths = set(by_path) - protected_paths
        bulk_kinds_only = all(
            by_path[path] <= {"created", "modified", "moved"} for path in ordinary_paths
        )
        is_burst = (
            len(ordinary_paths) >= self.config.burst_threshold
            and bulk_kinds_only
            and bool(ordinary_paths)
        )

        if is_burst:
            self.storage.record_alert(
                Alert(
                    module="module_b",
                    severity="info",
                    title=f"Grouped file activity ({len(ordinary_paths)} files)",
                    grouped_count=len(ordinary_paths),
                    signals={
                        "event_count": len(events),
                        "unique_paths": len(ordinary_paths),
                        "paths_sample": sorted(ordinary_paths)[:10],
                        "executable_hash_count": sum(path in observations for path in ordinary_paths),
                    },
                ),
                dedupe_key="file-activity-burst",
                cooldown_seconds=self.config.cooldown_seconds,
            )
            by_path = {path: kinds for path, kinds in by_path.items() if path in protected_paths}

        for path, kinds in by_path.items():
            kind = _preferred_kind(kinds)
            is_exec = is_executable_path(path, self.config)
            is_protected = path in protected_paths
            severity = classify_file_event(kind, is_exec, is_protected)
            observation = observations.get(path, {})
            details: dict[str, Any] = {
                "path": path,
                "event_kinds": sorted(kinds),
                "executable": is_exec,
                "critical_path": is_critical_path(path, self.config),
                "startup_path": is_startup_path(path, self.config),
            }
            details.update(observation)
            key_kind = "modified" if kind in {"modified", "deleted", "moved"} else kind
            self.storage.record_alert(
                Alert(
                    module="module_b",
                    severity=severity,
                    title=_file_event_title(kind, path, is_exec, is_protected),
                    signals=details,
                ),
                dedupe_key=f"file:{os.path.normcase(path).lower()}:{key_kind}",
                cooldown_seconds=self.config.cooldown_seconds,
            )


def classify_file_event(kind: str, executable: bool, protected: bool) -> str:
    """Map passive file signals to info/warning; file changes never trigger action."""
    if protected or executable or kind in {"modified", "deleted", "moved"}:
        return "warning"
    return "info"


def _preferred_kind(kinds: set[str]) -> str:
    if "deleted" in kinds:
        return "deleted"
    if "moved" in kinds:
        return "moved"
    if "created" in kinds:
        return "created"
    return "modified"


def _file_event_title(kind: str, path: str, executable: bool, protected: bool) -> str:
    if protected:
        return "Protected or startup path changed"
    if executable:
        return "Executable file changed"
    if kind == "modified":
        return "Watched file modified"
    if kind == "deleted":
        return "Watched file deleted"
    if kind == "moved":
        return "Watched file moved"
    return "Watched file created"


def _hash_stable_file(path: str, *, retries: int = 2) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Hash in chunks on the worker thread and reject files changing mid-read."""
    for attempt in range(retries):
        try:
            if os.path.islink(path) or not os.path.isfile(path):
                return None, None, "file is unavailable or is a symbolic link"
            before = os.stat(path)
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = os.stat(path)
            if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
                return digest.hexdigest(), after.st_size, None
        except OSError as exc:
            return None, None, str(exc)
        if attempt + 1 < retries:
            time.sleep(0.05)
    return None, None, "file changed while hashing"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    monitor = FileMonitor()
    monitor.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
