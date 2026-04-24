"""
Core/schema.py - SIT-Py Unified Feature Schema
===============================================
SystemSnapshot is the canonical real-time system state object.
It is the single source of truth across all three ML modules.

This permanently resolves the CPUUsage / CPU_Usage naming split and
eliminates the three manual remapping functions in main.py and dashboard/app.py.

Design contract:
  - collect.py         produces  SystemSnapshot
  - feature_engineering.py enriches  SystemSnapshot (in-place)
  - predictor.py       consumes  snapshot.to_predictor_input()
  - anomaly.py         consumes  snapshot.to_anomaly_input()
  - classifier.py      consumes  snapshot.to_classifier_input()
  - main.py / dashboard call snapshot.to_log_row() for CSV writing
"""
from __future__ import annotations

import os
import sys
import platform
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── SystemSnapshot ───────────────────────────────────────────────────────────

@dataclass
class SystemSnapshot:
    """
    Canonical snapshot of real-time system state.

    Fields are populated in two stages:
      Stage 1 (collect.py):         hardware + OS metrics
      Stage 2 (feature_engineering.py):  derived/temporal features
    """

    # ── Identity ───────────────────────────────────────────────────────────
    timestamp: str = ""

    # ── Primary Resource Metrics ───────────────────────────────────────────
    cpu_pct:   float = 0.0   # CPU utilization  0–100
    ram_pct:   float = 0.0   # RAM utilization  0–100
    disk_pct:  float = 0.0   # Disk utilization 0–100

    # ── Network ─────────────────────────────────────────────────────────────
    bandwidth_kbps:  float = 0.0   # kilobits per second
    packet_rate_pps: float = 0.0   # packets per second

    # ── Hardware Sensors (Linux only; None on Windows) ─────────────────────
    temperature_c: Optional[float] = None   # degrees Celsius
    fan_rpm:       Optional[int]   = None   # revolutions per minute

    # ── Process Context ───────────────────────────────────────────────────
    process_count: int  = 0
    task_priority: str  = "Medium"   # Low / Medium / High / Critical

    # ── Derived - temporal features (populated by feature_engineering.py) ─
    cpu_delta:        float = 0.0   # cpu_pct ? previous cpu_pct
    ram_delta:        float = 0.0   # ram_pct ? previous ram_pct
    cpu_rolling_mean: float = 0.0   # mean over WINDOW_SIZE history
    ram_rolling_mean: float = 0.0
    cpu_rolling_std:  float = 0.0   # std dev over WINDOW_SIZE history
    ram_rolling_std:  float = 0.0
    bw_rolling_mean:  float = 0.0   # bandwidth rolling mean
    bw_spike:         float = 0.0   # bandwidth / bw_rolling_mean (spike ratio)

    # ── RAM / Disk absolute (useful for monitoring, not ML features) ──────
    ram_available_mb: float = 0.0
    disk_free_gb:     float = 0.0

    # ── View helpers ─────────────────────────────────────────────────────────

    def to_predictor_input(self) -> dict:
        """
        Returns the exact feature dict ResourcePredictor expects.
        Temporal features must be enriched first (feature_engineering.py).
        """
        return {
            "cpu_pct":           self.cpu_pct,
            "ram_pct":           self.ram_pct,
            "disk_pct":          self.disk_pct,
            "bandwidth_kbps":    self.bandwidth_kbps,
            "packet_rate_pps":   self.packet_rate_pps,
            "cpu_delta":         self.cpu_delta,
            "ram_delta":         self.ram_delta,
            "cpu_rolling_mean":  self.cpu_rolling_mean,
            "ram_rolling_mean":  self.ram_rolling_mean,
            "cpu_rolling_std":   self.cpu_rolling_std,
            "ram_rolling_std":   self.ram_rolling_std,
            "bw_rolling_mean":   self.bw_rolling_mean,
            "bw_spike":          self.bw_spike,
        }

    def to_anomaly_input(self) -> dict:
        """Returns the exact feature dict AnomalyDetector expects."""
        return {
            "cpu_pct":           self.cpu_pct,
            "ram_pct":           self.ram_pct,
            "bandwidth_kbps":    self.bandwidth_kbps,
            "packet_rate_pps":   self.packet_rate_pps,
            "task_priority":     self.task_priority,
            "cpu_delta":         self.cpu_delta,
            "ram_delta":         self.ram_delta,
            "cpu_rolling_mean":  self.cpu_rolling_mean,
            "cpu_rolling_std":   self.cpu_rolling_std,
            "bw_rolling_mean":   self.bw_rolling_mean,
            "bw_spike":          self.bw_spike,
        }

    def to_classifier_input(self) -> dict:
        """Returns the exact feature dict WorkloadClassifier expects."""
        return {
            "cpu_pct":           self.cpu_pct,
            "ram_pct":           self.ram_pct,
            "bandwidth_kbps":    self.bandwidth_kbps,
            "packet_rate_pps":   self.packet_rate_pps,
            "process_count":     float(self.process_count),
            "cpu_rolling_mean":  self.cpu_rolling_mean,
            "ram_rolling_mean":  self.ram_rolling_mean,
            "bw_spike":          self.bw_spike,
        }

    def to_log_row(self, prediction: dict = None, anomaly: dict = None,
                   workload: dict = None) -> dict:
        """
        Returns a flat dict for CSV logging.
        Merges prediction/anomaly/workload results if provided.
        """
        row = {
            "timestamp":       self.timestamp,
            "cpu_pct":         self.cpu_pct,
            "ram_pct":         self.ram_pct,
            "disk_pct":        self.disk_pct,
            "bandwidth_kbps":  self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "task_priority":   self.task_priority,
            "process_count":   self.process_count,
            "cpu_delta":       self.cpu_delta,
            "ram_delta":       self.ram_delta,
            "cpu_rolling_mean":self.cpu_rolling_mean,
            "cpu_rolling_std": self.cpu_rolling_std,
        }
        if prediction:
            row["pred_cpu"]  = prediction.get("cpu_pct",  "N/A")
            row["pred_ram"]  = prediction.get("ram_pct",  "N/A")
            row["pred_disk"] = prediction.get("disk_pct", "N/A")
        if anomaly:
            row["anomaly_label"] = anomaly.get("label")
            row["anomaly_score"] = anomaly.get("anomaly_score")
            row["is_anomaly"]    = anomaly.get("is_anomaly")
        if workload:
            row["workload"]    = workload.get("workload")
            row["wl_conf"]     = workload.get("confidence")
        return row

    def to_live_csv_row(self) -> dict:
        """
        Flat dict for appending to live_metrics.csv.
        Used by collect.py for raw metric accumulation.
        """
        return {
            "timestamp":       self.timestamp,
            "cpu_pct":         self.cpu_pct,
            "ram_pct":         self.ram_pct,
            "disk_pct":        self.disk_pct,
            "bandwidth_kbps":  self.bandwidth_kbps,
            "packet_rate_pps": self.packet_rate_pps,
            "temperature_c":   self.temperature_c if self.temperature_c is not None else "",
            "fan_rpm":         self.fan_rpm if self.fan_rpm is not None else "",
            "process_count":   self.process_count,
            "task_priority":   self.task_priority,
            "ram_available_mb":self.ram_available_mb,
            "disk_free_gb":    self.disk_free_gb,
        }


# ── Feature column registries (used by ML modules) ───────────────────────────

PREDICTOR_FEATURE_COLS = list(SystemSnapshot().to_predictor_input().keys())
ANOMALY_FEATURE_COLS   = list(SystemSnapshot().to_anomaly_input().keys())
CLASSIFIER_FEATURE_COLS= list(SystemSnapshot().to_classifier_input().keys())

PREDICTOR_TARGET_COLS  = ["cpu_pct", "ram_pct", "disk_pct"]
