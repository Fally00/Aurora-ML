"""
Aurora-ML Schema v2 — Extended system snapshot for Collector v2.

Adds GPU (NVML), CPU package temp (LHM), top processes, machine identity,
workload tagging, and collector self-overhead metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import time


@dataclass
class ProcessInfo:
    """Lightweight process info for top-N lists."""
    pid: int
    name: str
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_util_percent: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "cpu_percent": round(self.cpu_percent, 2),
            "gpu_memory_mb": round(self.gpu_memory_mb, 2) if self.gpu_memory_mb is not None else None,
            "gpu_util_percent": round(self.gpu_util_percent, 2) if self.gpu_util_percent is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProcessInfo":
        return cls(
            pid=d["pid"],
            name=d["name"],
            cpu_percent=d["cpu_percent"],
            gpu_memory_mb=d.get("gpu_memory_mb"),
            gpu_util_percent=d.get("gpu_util_percent"),
        )


@dataclass
class SystemSnapshotV2:
    """Extended snapshot with GPU, CPU temp, top processes, and self-overhead."""

    # Core identity
    timestamp: str = ""
    timestamp_ns: int = 0
    machine_id: str = "laptop"
    schema_version: int = 2

    # Workload tag (user-set)
    workload_tag: str = "idle"

    # CPU
    cpu_pct: float = 0.0
    cpu_per_core_pct: list[float] = field(default_factory=list)
    cpu_package_temp_c: Optional[float] = None
    cpu_core_clocks_mhz: list[float] = field(default_factory=list)

    # RAM
    ram_pct: float = 0.0
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    ram_total_mb: float = 0.0

    # Disk
    disk_pct: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0

    # Network
    bandwidth_kbps: float = 0.0
    packet_rate_pps: float = 0.0

    # GPU (NVML) - all Optional, NaN if unavailable
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_mem_pct: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    gpu_power_w: Optional[float] = None
    gpu_power_limit_w: Optional[float] = None
    gpu_core_clock_mhz: Optional[float] = None
    gpu_mem_clock_mhz: Optional[float] = None
    gpu_fan_pct: Optional[float] = None
    gpu_throttle_reasons: list[str] = field(default_factory=list)

    # Top processes (JSON-serialized lists)
    top_cpu_processes: list[ProcessInfo] = field(default_factory=list)
    top_gpu_processes: list[ProcessInfo] = field(default_factory=list)

    # Collector self-overhead
    collector_cpu_pct: float = 0.0
    collector_rss_mb: float = 0.0

    # Derived / rolling (populated by FeatureEngineer)
    cpu_delta: float = 0.0
    ram_delta: float = 0.0
    cpu_rolling_mean: float = 0.0
    ram_rolling_mean: float = 0.0
    cpu_rolling_std: float = 0.0
    ram_rolling_std: float = 0.0
    bw_rolling_mean: float = 0.0
    bw_spike: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.timestamp_ns:
            self.timestamp_ns = time.time_ns()

    def to_sqlite_row(self) -> dict[str, Any]:
        """Convert to flat dict for SQLite INSERT."""
        row = asdict(self)
        # Serialize lists to JSON
        row["cpu_per_core_pct"] = json.dumps(self.cpu_per_core_pct)
        row["cpu_core_clocks_mhz"] = json.dumps(self.cpu_core_clocks_mhz)
        row["gpu_throttle_reasons"] = json.dumps(self.gpu_throttle_reasons)
        row["top_cpu_processes"] = json.dumps([p.to_dict() for p in self.top_cpu_processes])
        row["top_gpu_processes"] = json.dumps([p.to_dict() for p in self.top_gpu_processes])
        # Convert None to NULL for SQL
        for k, v in row.items():
            if v is None:
                row[k] = None
        return row

    @classmethod
    def from_sqlite_row(cls, row: dict[str, Any]) -> "SystemSnapshotV2":
        """Reconstruct from SQLite row."""
        data = dict(row)
        # Deserialize JSON fields
        for field_name in ("cpu_per_core_pct", "cpu_core_clocks_mhz", "gpu_throttle_reasons"):
            if data.get(field_name):
                data[field_name] = json.loads(data[field_name])
            else:
                data[field_name] = []
        for field_name in ("top_cpu_processes", "top_gpu_processes"):
            if data.get(field_name):
                data[field_name] = [ProcessInfo.from_dict(p) for p in json.loads(data[field_name])]
            else:
                data[field_name] = []
        return cls(**data)

    def to_anomaly_input(self) -> dict[str, Any]:
        """Compatibility with AnomalyDetector (uses v1 feature set)."""
        return {
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "protocol": "",  # not collected in v2
        }

    def to_classifier_input(self) -> dict[str, float]:
        """Compatibility with WorkloadClassifier."""
        return {
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
        }

    def to_predictor_input(self) -> dict[str, float]:
        """Compatibility with ResourcePredictor."""
        return {
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "disk_pct": self.disk_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "cpu_delta": self.cpu_delta,
            "ram_delta": self.ram_delta,
            "cpu_rolling_mean": self.cpu_rolling_mean,
            "ram_rolling_mean": self.ram_rolling_mean,
            "cpu_rolling_std": self.cpu_rolling_std,
            "ram_rolling_std": self.ram_rolling_std,
            "bw_rolling_mean": self.bw_rolling_mean,
            "bw_spike": self.bw_spike,
        }


# SQLite schema
SCHEMA_V2_SQL = """
-- Schema migrations table. Metric rows live in per-day tables, created lazily.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);
"""


# Daily table name format
def daily_table_name(date_str: str) -> str:
    """Return daily table name for given date string (YYYY-MM-DD)."""
    return f"metrics_{date_str.replace('-', '')}"


# Column list for INSERT
INSERT_COLUMNS = [
    "timestamp", "timestamp_ns", "machine_id", "schema_version", "workload_tag",
    "cpu_pct", "cpu_per_core_pct", "cpu_package_temp_c", "cpu_core_clocks_mhz",
    "ram_pct", "ram_used_mb", "ram_available_mb", "ram_total_mb",
    "disk_pct", "disk_free_gb", "disk_total_gb",
    "bandwidth_kbps", "packet_rate_pps",
    "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_pct",
    "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_core_clock_mhz",
    "gpu_mem_clock_mhz", "gpu_fan_pct", "gpu_throttle_reasons",
    "top_cpu_processes", "top_gpu_processes",
    "collector_cpu_pct", "collector_rss_mb",
    "cpu_delta", "ram_delta", "cpu_rolling_mean", "ram_rolling_mean",
    "cpu_rolling_std", "ram_rolling_std", "bw_rolling_mean", "bw_spike",
]

PLACEHOLDERS = ", ".join(["?"] * len(INSERT_COLUMNS))
INSERT_SQL = f"INSERT INTO {{table}} ({', '.join(INSERT_COLUMNS)}) VALUES ({PLACEHOLDERS})"
