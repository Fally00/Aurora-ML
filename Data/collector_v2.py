"""
Aurora-ML Collector v2 — 1s sampling, GPU (NVML), CPU temp (LHM), SQLite storage.

Signals:
- CPU: util%, per-core%, package temp (LHM), core clocks (LHM)
- RAM: used/avail/total MB, %
- Disk: free/total GB, %
- Net: bandwidth kbps, packet rate pps
- GPU (NVML): util%, VRAM used/total/%, temp, power, clocks, fan, throttle reasons
- Top processes by CPU% and GPU%
- Self-overhead: collector CPU%, RSS MB
- Workload tag (from file or CLI)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import psutil

# Project imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    POLL_INTERVAL_V2,
    ROLLING_WINDOW_V2,
    DB_FLUSH_INTERVAL,
    DB_BATCH_SIZE,
    DB_PATH,
    MACHINE_ID,
    SCHEMA_VERSION,
    WORKLOAD_TAG_FILE,
    VALID_WORKLOAD_TAGS,
    NVML_WARNING_LOGGED,
    LHM_WARNING_LOGGED,
)
from Core.schema_v2 import SystemSnapshotV2, ProcessInfo
from Data.storage import get_db

LOGGER = logging.getLogger(__name__)

# ── Global state for sensors ────────────────────────────────────────────────

_nvml_handle = None
_nvml_initialized = False
_nvml_warning_logged = False

_lhm_handle = None
_lhm_initialized = False
_lhm_warning_logged = False

_workload_tag = "idle"
_workload_tag_lock = threading.Lock()


# ── NVML (GPU) ──────────────────────────────────────────────────────────────

def _init_nvml() -> bool:
    """Initialize NVML. Returns True on success."""
    global _nvml_handle, _nvml_initialized, _nvml_warning_logged
    if _nvml_initialized:
        return _nvml_handle is not None
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        _nvml_handle = nvml
        _nvml_initialized = True
        return True
    except Exception as e:
        if not _nvml_warning_logged:
            print(f"[Collector] NVML unavailable: {e}", file=sys.stderr)
            _nvml_warning_logged = True
        _nvml_initialized = True
        return False


def _get_gpu_metrics() -> Dict[str, Any]:
    """Collect all GPU metrics via NVML. Returns dict with NaN/None on failure."""
    if not _init_nvml() or _nvml_handle is None:
        return _empty_gpu_metrics()

    nvml = _nvml_handle
    try:
        device_count = nvml.nvmlDeviceGetCount()
        if device_count == 0:
            return _empty_gpu_metrics()

        # Use first GPU (index 0)
        handle = nvml.nvmlDeviceGetHandleByIndex(0)

        # Utilization
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu

        # Memory
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = mem.used / (1024 ** 2)
        mem_total_mb = mem.total / (1024 ** 2)
        mem_pct = (mem.used / mem.total) * 100 if mem.total > 0 else 0.0

        # Temperature
        try:
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None

        # Power
        try:
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
        except Exception:
            power = None
        try:
            power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
        except Exception:
            power_limit = None

        # Clocks
        try:
            core_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            core_clock = None
        try:
            mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
        except Exception:
            mem_clock = None

        # Fan
        try:
            fan = nvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            fan = None

        # Throttle reasons
        throttle_reasons = []
        try:
            throttle = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_GPU_IDLE:
                throttle_reasons.append("gpu_idle")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_APPLICATIONS_CLOCKS_SETTING:
                throttle_reasons.append("app_clock_setting")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP:
                throttle_reasons.append("sw_power_cap")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_CAP:
                throttle_reasons.append("hw_power_cap")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL:
                throttle_reasons.append("sw_thermal")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL:
                throttle_reasons.append("hw_thermal")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE:
                throttle_reasons.append("hw_power_brake")
            if throttle & nvml.NVML_CLOCKS_THROTTLE_REASON_SYNC_BOOST:
                throttle_reasons.append("sync_boost")
        except Exception:
            pass

        return {
            "gpu_util_pct": float(gpu_util),
            "gpu_mem_used_mb": mem_used_mb,
            "gpu_mem_total_mb": mem_total_mb,
            "gpu_mem_pct": mem_pct,
            "gpu_temp_c": float(temp) if temp is not None else None,
            "gpu_power_w": power,
            "gpu_power_limit_w": power_limit,
            "gpu_core_clock_mhz": float(core_clock) if core_clock is not None else None,
            "gpu_mem_clock_mhz": float(mem_clock) if mem_clock is not None else None,
            "gpu_fan_pct": float(fan) if fan is not None else None,
            "gpu_throttle_reasons": throttle_reasons,
        }
    except Exception as e:
        if not _nvml_warning_logged:
            print(f"[Collector] NVML error: {e}", file=sys.stderr)
            _nvml_warning_logged = True
        return _empty_gpu_metrics()


def _empty_gpu_metrics() -> Dict[str, Any]:
    return {
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_mem_pct": None,
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "gpu_power_limit_w": None,
        "gpu_core_clock_mhz": None,
        "gpu_mem_clock_mhz": None,
        "gpu_fan_pct": None,
        "gpu_throttle_reasons": [],
    }


# ── LibreHardwareMonitor (CPU Package Temp) ─────────────────────────────────

def _init_lhm() -> bool:
    """Initialize LibreHardwareMonitor WMI. Returns True on success (Windows only)."""
    global _lhm_handle, _lhm_initialized, _lhm_warning_logged
    if platform.system() != "Windows":
        _warn_lhm_unavailable()
        _lhm_initialized = True
        return False
    if _lhm_initialized:
        return _lhm_handle is not None
    try:
        import wmi
        _lhm_handle = wmi.WMI(namespace="root\\LibreHardwareMonitor")
        _lhm_initialized = True
        return True
    except Exception as e:
        LOGGER.debug("LibreHardwareMonitor initialization failed: %s", e)
        _warn_lhm_unavailable()
        _lhm_initialized = True
        return False


def _warn_lhm_unavailable() -> None:
    global _lhm_warning_logged
    if not _lhm_warning_logged:
        LOGGER.warning("CPU temp: N/A (LibreHardwareMonitor not running)")
        _lhm_warning_logged = True


def _get_cpu_package_temp_and_clocks() -> Tuple[Optional[float], List[float]]:
    """Get CPU package temp and per-core clocks via LHM. Returns (temp_c, core_clocks_mhz)."""
    global _lhm_warning_logged
    if not _init_lhm() or _lhm_handle is None:
        return None, []

    try:
        sensors = _lhm_handle.Sensor()
        package_temp = None
        core_clocks = []

        for s in sensors:
            if s.SensorType == "Temperature" and "package" in s.Name.lower():
                package_temp = float(s.Value)
            elif s.SensorType == "Clock" and "core" in s.Name.lower():
                core_clocks.append(float(s.Value))

        if package_temp is None and not core_clocks:
            _warn_lhm_unavailable()
        return package_temp, core_clocks
    except Exception as e:
        LOGGER.debug("LibreHardwareMonitor sensor read failed: %s", e)
        _warn_lhm_unavailable()
        return None, []


# ── psutil helpers ──────────────────────────────────────────────────────────

def _get_top_cpu_processes(n: int = 5) -> List[ProcessInfo]:
    """Get top N processes by CPU, normalized to total logical CPU capacity."""
    procs = []
    logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        try:
            info = proc.info
            pid = info["pid"]
            name = info["name"] or "unknown"
            if pid == 0 or name.casefold() == "system idle process":
                continue
            cpu_percent = (info["cpu_percent"] or 0.0) / logical_cpus
            if cpu_percent > 0:
                procs.append(ProcessInfo(
                    pid=pid,
                    name=name,
                    cpu_percent=min(cpu_percent, 100.0),
                ))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda p: p.cpu_percent, reverse=True)
    return procs[:n]


def _get_top_gpu_processes(n: int = 5) -> List[ProcessInfo]:
    """Get top N processes by GPU memory (via NVML)."""
    if not _init_nvml() or _nvml_handle is None:
        return []

    nvml = _nvml_handle
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        compute_procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        graphics_procs = nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        all_procs = compute_procs + graphics_procs

        procs = []
        logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
        for p in all_procs:
            try:
                ps = psutil.Process(p.pid)
                procs.append(ProcessInfo(
                    pid=p.pid,
                    name=ps.name(),
                    cpu_percent=min(ps.cpu_percent() / logical_cpus, 100.0),
                    gpu_memory_mb=p.usedGpuMemory / (1024 ** 2) if p.usedGpuMemory else 0.0,
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda p: p.gpu_memory_mb or 0, reverse=True)
        return procs[:n]
    except Exception:
        return []


# ── Workload Tag ────────────────────────────────────────────────────────────

def get_workload_tag() -> str:
    """Get current workload tag (reads file each call)."""
    global _workload_tag
    with _workload_tag_lock:
        try:
            if Path(WORKLOAD_TAG_FILE).exists():
                tag = Path(WORKLOAD_TAG_FILE).read_text().strip().lower()
                if tag in VALID_WORKLOAD_TAGS:
                    _workload_tag = tag
        except Exception:
            pass
        return _workload_tag


def set_workload_tag(tag: str) -> bool:
    """Set workload tag (writes to file)."""
    global _workload_tag
    tag = tag.strip().lower()
    if tag not in VALID_WORKLOAD_TAGS:
        return False
    with _workload_tag_lock:
        _workload_tag = tag
        try:
            Path(WORKLOAD_TAG_FILE).write_text(tag)
        except Exception:
            pass
    return True


# ── Main Snapshot ───────────────────────────────────────────────────────────

_prev_net: Optional[psutil._common.snicstats] = None
_prev_net_time: Optional[float] = None
_cpu_history = deque(maxlen=ROLLING_WINDOW_V2)
_ram_history = deque(maxlen=ROLLING_WINDOW_V2)
_bw_history = deque(maxlen=ROLLING_WINDOW_V2)


def take_snapshot() -> SystemSnapshotV2:
    """Take a single system snapshot (v2 schema)."""
    global _prev_net, _prev_net_time

    now = time.time()
    timestamp = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    timestamp_ns = time.time_ns()

    # CPU
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

    # RAM
    ram = psutil.virtual_memory()
    ram_pct = ram.percent
    ram_used_mb = ram.used / (1024 ** 2)
    ram_available_mb = ram.available / (1024 ** 2)
    ram_total_mb = ram.total / (1024 ** 2)

    # Disk
    disk = psutil.disk_usage("/")
    disk_pct = disk.percent
    disk_free_gb = disk.free / (1024 ** 3)
    disk_total_gb = disk.total / (1024 ** 3)

    # Network
    net = psutil.net_io_counters()
    bandwidth_kbps = 0.0
    packet_rate_pps = 0.0
    if _prev_net is not None and _prev_net_time is not None:
        elapsed = now - _prev_net_time
        bytes_delta = (net.bytes_sent + net.bytes_recv) - (_prev_net.bytes_sent + _prev_net.bytes_recv)
        pkts_delta = (net.packets_sent + net.packets_recv) - (_prev_net.packets_sent + _prev_net.packets_recv)
        if elapsed > 0:
            bandwidth_kbps = (max(bytes_delta, 0) * 8) / (elapsed * 1000)
            packet_rate_pps = max(pkts_delta, 0) / elapsed
    _prev_net = net
    _prev_net_time = now

    # GPU
    gpu_metrics = _get_gpu_metrics()

    # CPU package temp + clocks (LHM)
    cpu_pkg_temp, cpu_core_clocks = _get_cpu_package_temp_and_clocks()

    # Top processes
    top_cpu = _get_top_cpu_processes(5)
    top_gpu = _get_top_gpu_processes(5)

    # Collector self-overhead
    self_proc = psutil.Process()
    collector_cpu = self_proc.cpu_percent(interval=None)
    collector_rss_mb = self_proc.memory_info().rss / (1024 ** 2)

    # Workload tag
    workload_tag = get_workload_tag()

    snap = SystemSnapshotV2(
        timestamp=timestamp,
        timestamp_ns=timestamp_ns,
        machine_id=MACHINE_ID,
        schema_version=SCHEMA_VERSION,
        workload_tag=workload_tag,

        cpu_pct=cpu_pct,
        cpu_per_core_pct=cpu_per_core,
        cpu_package_temp_c=cpu_pkg_temp,
        cpu_core_clocks_mhz=cpu_core_clocks,

        ram_pct=ram_pct,
        ram_used_mb=ram_used_mb,
        ram_available_mb=ram_available_mb,
        ram_total_mb=ram_total_mb,

        disk_pct=disk_pct,
        disk_free_gb=disk_free_gb,
        disk_total_gb=disk_total_gb,

        bandwidth_kbps=bandwidth_kbps,
        packet_rate_pps=packet_rate_pps,

        **gpu_metrics,

        top_cpu_processes=top_cpu,
        top_gpu_processes=top_gpu,

        collector_cpu_pct=collector_cpu,
        collector_rss_mb=collector_rss_mb,

        cpu_delta=0.0,
        ram_delta=0.0,
        cpu_rolling_mean=0.0,
        ram_rolling_mean=0.0,
        cpu_rolling_std=0.0,
        ram_rolling_std=0.0,
        bw_rolling_mean=0.0,
        bw_spike=1.0,
    )
    _apply_rolling_features(snap)
    return snap


def _apply_rolling_features(snap: SystemSnapshotV2) -> None:
    """Update fixed-size telemetry rings and populate rolling features in place."""
    snap.cpu_delta = snap.cpu_pct - _cpu_history[-1] if _cpu_history else 0.0
    snap.ram_delta = snap.ram_pct - _ram_history[-1] if _ram_history else 0.0
    _cpu_history.append(float(snap.cpu_pct))
    _ram_history.append(float(snap.ram_pct))
    _bw_history.append(float(snap.bandwidth_kbps))

    def mean(values) -> float:
        return sum(values) / len(values) if values else 0.0

    def sample_std(values) -> float:
        if len(values) < 2:
            return 0.0
        average = mean(values)
        return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))

    snap.cpu_rolling_mean = mean(_cpu_history)
    snap.ram_rolling_mean = mean(_ram_history)
    snap.cpu_rolling_std = sample_std(_cpu_history)
    snap.ram_rolling_std = sample_std(_ram_history)
    snap.bw_rolling_mean = mean(_bw_history)
    snap.bw_spike = snap.bandwidth_kbps / snap.bw_rolling_mean if snap.bw_rolling_mean > 0 else 1.0


# ── Collector Loop ──────────────────────────────────────────────────────────

def run_collector(
    duration_seconds: Optional[int] = None,
    verbose: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Main polling loop. Writes to SQLite until duration or stop event."""
    if verbose:
        print(f"[Collector v2] Starting - polling every {POLL_INTERVAL_V2}s")
        print(f"[Collector v2] Machine: {MACHINE_ID} | DB: {DB_PATH}")
        print(f"[Collector v2] Workload tag: {get_workload_tag()} (edit {WORKLOAD_TAG_FILE} to change)")
        print("-" * 60)

    db = get_db()
    db.prune_old_data()
    start_time = time.time()
    snapshot_num = 0
    buffer: List[SystemSnapshotV2] = []
    last_flush = time.monotonic()

    try:
        while stop_event is None or not stop_event.is_set():
            loop_start = time.time()

            snap = take_snapshot()
            buffer.append(snap)
            snapshot_num += 1

            if verbose:
                gpu_str = f"GPU: {snap.gpu_util_pct:.1f}% {snap.gpu_temp_c:.1f}°C" if snap.gpu_util_pct is not None else "GPU: N/A"
                cpu_temp_str = f"CPU pkg: {snap.cpu_package_temp_c:.1f}°C" if snap.cpu_package_temp_c is not None else "CPU pkg: N/A"
                print(
                    f"[{snap.timestamp}] "
                    f"CPU: {snap.cpu_pct:5.1f}% "
                    f"RAM: {snap.ram_pct:5.1f}% "
                    f"{cpu_temp_str} "
                    f"{gpu_str} "
                    f"BW: {snap.bandwidth_kbps:8.1f} kbps "
                    f"Tag: {snap.workload_tag} "
                    f"Self: {snap.collector_cpu_pct:.1f}% {snap.collector_rss_mb:.1f}MB"
                )

            # Flush every 10 rows
            flush_due = time.monotonic() - last_flush >= DB_FLUSH_INTERVAL
            if len(buffer) >= DB_BATCH_SIZE or flush_due:
                db.insert_batch(buffer)
                buffer = []
                last_flush = time.monotonic()
                if verbose:
                    print(f"  -> flushed {snapshot_num} rows to SQLite")

            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                if verbose:
                    print(f"[Collector] Duration reached ({duration_seconds}s), stopping.")
                break
            if stop_event is not None and stop_event.is_set():
                break

            # Sleep to maintain 1s interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, POLL_INTERVAL_V2 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        if verbose:
            print("\n[Collector] Stopped by user.")
    finally:
        if buffer:
            db.insert_batch(buffer)
        if verbose:
            print(f"[Collector] Done. Total snapshots: {snapshot_num}")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aurora-ML Collector v2")
    parser.add_argument("--duration", type=int, default=None,
                        help="Stop after N seconds (default: run forever)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-snapshot output")
    parser.add_argument("--tag", type=str, choices=sorted(VALID_WORKLOAD_TAGS),
                        help="Set workload tag for this run")
    args = parser.parse_args()

    if args.tag:
        set_workload_tag(args.tag)
        print(f"[Collector] Workload tag set to: {args.tag}")

    run_collector(duration_seconds=args.duration, verbose=not args.quiet)
