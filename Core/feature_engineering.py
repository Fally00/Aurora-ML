"""
Temporal feature enrichment for live SIT-Py snapshots.

FeatureEngineer is the shared runtime state that adds rolling statistics and
simple drift signals to each SystemSnapshot before it reaches the models.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from typing import Deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import WORKLOAD_THRESHOLDS
from Core.schema import SystemSnapshot


class FeatureEngineer:
    """
    Stateful rolling-window enricher for live telemetry.

    The history stores recent snapshots so the current reading can be turned
    into deltas, rolling means/std, and bridge-mode anomaly signals.
    """

    def __init__(self, window_size: int = 15, bridge_min_history: int = 5):
        self.window_size = max(int(window_size), 1)
        self.bridge_min_history = max(int(bridge_min_history), 2)
        self._history: Deque[SystemSnapshot] = deque(maxlen=self.window_size)

    def enrich(self, snap: SystemSnapshot) -> SystemSnapshot:
        """
        Mutate and return the current snapshot with temporal features added.
        """
        if not self._history:
            self._apply_cold_start_defaults(snap)
            self._history.append(snap)
            return snap

        cpu_vals = self._history_values("cpu_pct")
        ram_vals = self._history_values("ram_pct")
        bw_vals = self._history_values("bandwidth_kbps")

        snap.cpu_rolling_mean = _mean(cpu_vals)
        snap.ram_rolling_mean = _mean(ram_vals)
        snap.bw_rolling_mean = _mean(bw_vals)

        snap.cpu_rolling_std = _std(cpu_vals)
        snap.ram_rolling_std = _std(ram_vals)

        prev = self._history[-1]
        snap.cpu_delta = snap.cpu_pct - prev.cpu_pct
        snap.ram_delta = snap.ram_pct - prev.ram_pct

        if snap.bw_rolling_mean > 0:
            snap.bw_spike = snap.bandwidth_kbps / snap.bw_rolling_mean
        else:
            snap.bw_spike = 1.0

        self._history.append(snap)
        return snap

    def is_ready(self) -> bool:
        """True once enough history exists for full predictor context."""
        return len(self._history) >= self.window_size

    def history_len(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        """Clear all rolling history."""
        self._history.clear()

    def statistical_anomaly(
        self,
        snap: SystemSnapshot,
        z_threshold: float = 2.5,
    ) -> dict:
        """
        Lightweight bridge-mode anomaly detection while no trained model exists.

        The returned shape intentionally mirrors the supervised anomaly model so
        main.py and the dashboard do not need special handling.
        """
        if len(self._history) < self.bridge_min_history:
            return {
                "label": "Normal",
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "confidence": 1.0,
                "source": "stat_bridge_warmup",
            }

        cpu_z = _z_score(snap.cpu_pct, self._history_values("cpu_pct"))
        ram_z = _z_score(snap.ram_pct, self._history_values("ram_pct"))
        bw_z = _z_score(snap.bandwidth_kbps, self._history_values("bandwidth_kbps"))
        pkt_z = _z_score(snap.packet_rate_pps, self._history_values("packet_rate_pps"))

        z_scores = {
            "cpu": round(cpu_z, 2),
            "ram": round(ram_z, 2),
            "bw": round(bw_z, 2),
            "pkt": round(pkt_z, 2),
        }

        max_z = max(abs(value) for value in (cpu_z, ram_z, bw_z, pkt_z))
        anomaly_score = min(max_z / max(z_threshold, 1e-6), 1.0)
        is_anomaly = max_z >= z_threshold
        confidence = anomaly_score if is_anomaly else 1.0 - anomaly_score

        return {
            "label": "Alert" if is_anomaly else "Normal",
            "anomaly_score": round(float(anomaly_score), 4),
            "is_anomaly": is_anomaly,
            "confidence": round(float(confidence), 4),
            "source": "stat_bridge",
            "z_scores": z_scores,
        }

    @staticmethod
    def rule_based_workload(snap: SystemSnapshot) -> str:
        """
        Derive a workload label using the centralized config thresholds.
        """
        thresholds = WORKLOAD_THRESHOLDS
        if snap.cpu_pct > thresholds["Critical"]["cpu"] or snap.ram_pct > thresholds["Critical"]["ram"]:
            return "Critical"
        if snap.cpu_pct > thresholds["High"]["cpu"] or snap.ram_pct > thresholds["High"]["ram"]:
            return "High"
        if snap.cpu_pct > thresholds["Medium"]["cpu"] or snap.ram_pct > thresholds["Medium"]["ram"]:
            return "Medium"
        return "Low"

    def _apply_cold_start_defaults(self, snap: SystemSnapshot) -> None:
        snap.cpu_rolling_mean = snap.cpu_pct
        snap.ram_rolling_mean = snap.ram_pct
        snap.bw_rolling_mean = snap.bandwidth_kbps
        snap.cpu_rolling_std = 0.0
        snap.ram_rolling_std = 0.0
        snap.cpu_delta = 0.0
        snap.ram_delta = 0.0
        snap.bw_spike = 1.0

    def _history_values(self, field_name: str) -> list[float]:
        return [float(getattr(snap, field_name, 0.0) or 0.0) for snap in self._history]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _z_score(value: float, history: list[float]) -> float:
    """Return the z-score of value relative to the provided history."""
    if len(history) < 2:
        return 0.0
    avg = _mean(history)
    std = _std(history)
    if std < 1e-9:
        return 0.0
    return (value - avg) / std
