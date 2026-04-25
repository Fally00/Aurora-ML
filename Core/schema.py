"""
Shared runtime schema and dataframe normalization helpers.

SystemSnapshot is the canonical live object used across collection, inference,
logging, and retraining. The dataframe helpers normalize older CSV exports and
the new per-agent dataset files into a consistent shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


COLUMN_ALIASES = {
    "CPUUsage": "cpu_pct",
    "CPU_Usage": "cpu_pct",
    "RAMUsage": "ram_pct",
    "Memory_Usage": "ram_pct",
    "DiskUsage": "disk_pct",
    "Bandwidth": "bandwidth_kbps",
    "Packet_Rate": "packet_rate_pps",
    "Task_Priority": "task_priority",
    "Process_Count": "process_count",
    "Temperature": "temperature_c",
    "FanSpeed": "fan_rpm",
    "RAM_Available_MB": "ram_available_mb",
    "Disk_Free_GB": "disk_free_gb",
    "pred_CPU": "pred_cpu",
    "pred_RAM": "pred_ram",
    "pred_Disk": "pred_disk",
    "wl_confidence": "wl_conf",
}

TEXT_DEFAULTS = {
    "protocol": "",
    "attack_type": "",
    "severity": "",
    "threat_type": "",
    "task_priority": "",
    "label": "",
}

ANOMALY_LABEL_ALIASES = {
    "0": "Normal",
    "1": "Alert",
    "false": "Normal",
    "true": "Alert",
    "normal": "Normal",
    "benign": "Normal",
    "alert": "Alert",
    "attack": "Alert",
    "malicious": "Alert",
    "suspicious": "Alert",
}

WORKLOAD_LABEL_ALIASES = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
    "healthy": "Low",
    "degraded": "Medium",
    "cascading_failure": "High",
    "total_outage": "Critical",
}

SNAPSHOT_NUMERIC_COLS = [
    "cpu_pct",
    "ram_pct",
    "disk_pct",
    "bandwidth_kbps",
    "packet_rate_pps",
    "temperature_c",
    "fan_rpm",
    "process_count",
    "cpu_delta",
    "ram_delta",
    "cpu_rolling_mean",
    "ram_rolling_mean",
    "cpu_rolling_std",
    "ram_rolling_std",
    "bw_rolling_mean",
    "bw_spike",
    "ram_available_mb",
    "disk_free_gb",
]

ANOMALY_NUMERIC_COLS = [
    "cpu_pct",
    "ram_pct",
    "bandwidth_kbps",
    "packet_rate_pps",
]

ANOMALY_CATEGORICAL_COLS = [
    "protocol",
    "attack_type",
    "severity",
    "threat_type",
]

ANOMALY_FEATURE_COLS = ANOMALY_NUMERIC_COLS + ANOMALY_CATEGORICAL_COLS
ANOMALY_TARGET_COL = "label"

CLASSIFIER_FEATURE_COLS = [
    "cpu_pct",
    "ram_pct",
    "bandwidth_kbps",
    "packet_rate_pps",
]
CLASSIFIER_TARGET_COL = "task_priority"

PREDICTOR_RAW_COLS = [
    "cpu_pct",
    "ram_pct",
    "disk_pct",
    "bandwidth_kbps",
    "packet_rate_pps",
]

PREDICTOR_FEATURE_COLS = [
    "cpu_pct",
    "ram_pct",
    "disk_pct",
    "bandwidth_kbps",
    "packet_rate_pps",
    "cpu_delta",
    "ram_delta",
    "cpu_rolling_mean",
    "ram_rolling_mean",
    "cpu_rolling_std",
    "ram_rolling_std",
    "bw_rolling_mean",
    "bw_spike",
]
PREDICTOR_TARGET_COLS = ["cpu_pct", "ram_pct", "disk_pct"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known legacy columns to the current unified names."""
    out = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    if not out.columns.duplicated().any():
        return out

    collapsed = pd.DataFrame(index=out.index)
    ordered_cols = list(dict.fromkeys(out.columns))
    for col in ordered_cols:
        matching = out.loc[:, out.columns == col]
        if isinstance(matching, pd.Series):
            collapsed[col] = matching
        else:
            collapsed[col] = matching.bfill(axis=1).iloc[:, 0]
    return collapsed


def column_defaults(columns: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return sensible fill defaults for a schema."""
    return {col: TEXT_DEFAULTS.get(col, 0.0) for col in columns}


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    defaults: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Ensure the dataframe has the requested columns and keep only them."""
    defaults = dict(defaults or column_defaults(columns))
    out = normalize_columns(df.copy())
    for col in columns:
        if col not in out.columns:
            out[col] = defaults.get(col, TEXT_DEFAULTS.get(col, 0.0))
    return out[list(columns)]


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Coerce a list of columns to numeric values."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_anomaly_label(value: Any) -> str | None:
    """Map label variants such as Attack/benign/1/0 into Alert/Normal."""
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key in ANOMALY_LABEL_ALIASES:
        return ANOMALY_LABEL_ALIASES[key]
    return str(value).strip().title()


def normalize_workload_label(value: Any) -> str | None:
    """Map workload label variants into Low/Medium/High/Critical."""
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key in WORKLOAD_LABEL_ALIASES:
        return WORKLOAD_LABEL_ALIASES[key]
    return str(value).strip().title()


@dataclass
class SystemSnapshot:
    """Canonical snapshot of the live system state."""

    timestamp: str = ""

    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    disk_pct: float = 0.0

    bandwidth_kbps: float = 0.0
    packet_rate_pps: float = 0.0

    temperature_c: float | None = None
    fan_rpm: int | None = None

    process_count: int = 0
    task_priority: str = "Low"

    protocol: str = ""
    attack_type: str = ""
    severity: str = ""
    threat_type: str = ""

    cpu_delta: float = 0.0
    ram_delta: float = 0.0
    cpu_rolling_mean: float = 0.0
    ram_rolling_mean: float = 0.0
    cpu_rolling_std: float = 0.0
    ram_rolling_std: float = 0.0
    bw_rolling_mean: float = 0.0
    bw_spike: float = 0.0

    ram_available_mb: float = 0.0
    disk_free_gb: float = 0.0

    def to_predictor_input(self) -> dict[str, float]:
        """Return the exact live feature dict expected by ResourcePredictor."""
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

    def to_anomaly_input(self) -> dict[str, Any]:
        """Return the anomaly feature dict expected by AnomalyDetector."""
        return {
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "protocol": self.protocol,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "threat_type": self.threat_type,
        }

    def to_classifier_input(self) -> dict[str, float]:
        """Return the classifier feature dict expected by WorkloadClassifier."""
        return {
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
        }

    def to_log_row(
        self,
        prediction: dict[str, Any] | None = None,
        anomaly: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a flat row for sit_log.csv."""
        row = {
            "timestamp": self.timestamp,
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "disk_pct": self.disk_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "temperature_c": self.temperature_c if self.temperature_c is not None else "",
            "fan_rpm": self.fan_rpm if self.fan_rpm is not None else "",
            "process_count": self.process_count,
            "task_priority": self.task_priority,
            "protocol": self.protocol,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "threat_type": self.threat_type,
            "cpu_delta": self.cpu_delta,
            "ram_delta": self.ram_delta,
            "cpu_rolling_mean": self.cpu_rolling_mean,
            "ram_rolling_mean": self.ram_rolling_mean,
            "cpu_rolling_std": self.cpu_rolling_std,
            "ram_rolling_std": self.ram_rolling_std,
            "bw_rolling_mean": self.bw_rolling_mean,
            "bw_spike": self.bw_spike,
            "ram_available_mb": self.ram_available_mb,
            "disk_free_gb": self.disk_free_gb,
        }
        if prediction:
            row["pred_cpu"] = prediction.get("cpu_pct")
            row["pred_ram"] = prediction.get("ram_pct")
            row["pred_disk"] = prediction.get("disk_pct")
        if anomaly:
            row["anomaly_label"] = anomaly.get("label")
            row["anomaly_score"] = anomaly.get("anomaly_score")
            row["is_anomaly"] = anomaly.get("is_anomaly")
            row["anomaly_source"] = anomaly.get("source")
        if workload:
            row["workload"] = workload.get("workload")
            row["wl_conf"] = workload.get("confidence")
        return row

    def to_live_csv_row(self) -> dict[str, Any]:
        """Return a flat row for live_metrics.csv."""
        return {
            "timestamp": self.timestamp,
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "disk_pct": self.disk_pct,
            "bandwidth_kbps": self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "temperature_c": self.temperature_c if self.temperature_c is not None else "",
            "fan_rpm": self.fan_rpm if self.fan_rpm is not None else "",
            "process_count": self.process_count,
            "task_priority": self.task_priority,
            "protocol": self.protocol,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "threat_type": self.threat_type,
            "ram_available_mb": self.ram_available_mb,
            "disk_free_gb": self.disk_free_gb,
        }
