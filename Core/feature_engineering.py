"""
Core/feature_engineering.py - SIT-Py Temporal Feature Enrichment
=================================================================
Takes a raw SystemSnapshot and enriches it with rolling statistics,
temporal deltas, and derived ratios using a history ring-buffer.

This is the component that gives all three ML models temporal awareness -
the ability to detect trends, spikes, and drift, not just point-in-time values.

Usage (in main.py / dashboard/app.py):
    from collections import deque
    from Core.feature_engineering import FeatureEngineer
    from config import WINDOW_SIZE

    eng = FeatureEngineer(window_size=WINDOW_SIZE)
    # each cycle:
    enriched_snap = eng.enrich(raw_snapshot)
    # then pass enriched_snap to ML modules
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque

from Core.schema import SystemSnapshot


class FeatureEngineer:
    """
    Stateful rolling-window feature enricher.

    Maintains an internal deque of past SystemSnapshots and
    computes derived temporal features for each new snapshot.

    Attributes:
        window_size: number of past readings used for rolling stats.
        _history:    ring-buffer of the last `window_size` snapshots
                     (before enrichment, to avoid circular dependency).
    """

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self._history: Deque[SystemSnapshot] = deque(maxlen=window_size)

    # ── Public API ───────────────────────────────────────────────────────────

    def enrich(self, snap: SystemSnapshot) -> SystemSnapshot:
        """
        Enriches `snap` with temporal features computed from internal history.
        Appends the raw snap to history AFTER enrichment to avoid self-reference.

        Returns the same snapshot object (mutated in-place) for convenience.
        """
        if len(self._history) >= 2:
            cpu_vals = [s.cpu_pct for s in self._history]
            ram_vals = [s.ram_pct for s in self._history]
            bw_vals  = [s.bandwidth_kbps for s in self._history]

            # Rolling mean
            snap.cpu_rolling_mean = _mean(cpu_vals)
            snap.ram_rolling_mean = _mean(ram_vals)
            snap.bw_rolling_mean  = _mean(bw_vals)

            # Rolling std (volatility signal)
            snap.cpu_rolling_std  = _std(cpu_vals)
            snap.ram_rolling_std  = _std(ram_vals)

            # Temporal delta (direction and magnitude of change)
            prev = self._history[-1]
            snap.cpu_delta = snap.cpu_pct  - prev.cpu_pct
            snap.ram_delta = snap.ram_pct  - prev.ram_pct

            # Bandwidth spike ratio: current / rolling mean
            # Values > 3.0 signal burst events (downloads, sync, etc.)
            if snap.bw_rolling_mean > 0:
                snap.bw_spike = snap.bandwidth_kbps / snap.bw_rolling_mean
            else:
                snap.bw_spike = 1.0

        else:
            # Not enough history yet - set safe defaults
            snap.cpu_rolling_mean = snap.cpu_pct
            snap.ram_rolling_mean = snap.ram_pct
            snap.bw_rolling_mean  = snap.bandwidth_kbps
            snap.cpu_rolling_std  = 0.0
            snap.ram_rolling_std  = 0.0
            snap.cpu_delta        = 0.0
            snap.ram_delta        = 0.0
            snap.bw_spike         = 1.0

        # Append raw (unenriched) snapshot to history for next cycle
        self._history.append(snap)
        return snap

    def is_ready(self) -> bool:
        """True once history has enough readings for meaningful statistics."""
        return len(self._history) >= self.window_size

    def history_len(self) -> int:
        return len(self._history)

    def reset(self):
        """Clears history (e.g., after retraining or session restart)."""
        self._history.clear()

    # ── Statistical anomaly bridge ───────────────────────────────────────────

    def statistical_anomaly(self, snap: SystemSnapshot,
                            z_threshold: float = 2.5) -> dict:
        """
        Rule-based anomaly detection using z-scores over rolling window.
        Used as a bridge while Isolation Forest is being tuned to real data.

        Returns same schema as AnomalyDetector.predict():
            {"label": str, "anomaly_score": float, "is_anomaly": bool, "source": str}
        """
        if len(self._history) < 5:
            # Too early to judge - return "Normal" with low confidence score
            return {
                "label":         "Normal",
                "anomaly_score": 0.0,
                "is_anomaly":    False,
                "source":        "stat_bridge_warmup",
            }

        cpu_vals = [s.cpu_pct for s in self._history]
        ram_vals = [s.ram_pct for s in self._history]
        bw_vals  = [s.bandwidth_kbps for s in self._history]

        cpu_z = _z_score(snap.cpu_pct, cpu_vals)
        ram_z = _z_score(snap.ram_pct, ram_vals)
        bw_z  = _z_score(snap.bandwidth_kbps, bw_vals)

        # Composite score: max z across metrics, normalized to [-1, 0] range
        # (mimics Isolation Forest sign convention - more negative = more anomalous)
        max_z    = max(abs(cpu_z), abs(ram_z), abs(bw_z))
        score    = -min(max_z / 10.0, 1.0)   # scale z to approx IF score range
        is_anom  = max_z >= z_threshold

        return {
            "label":         "Alert" if is_anom else "Normal",
            "anomaly_score": round(score, 4),
            "is_anomaly":    is_anom,
            "source":        "stat_bridge",
            "z_scores":      {
                "cpu": round(cpu_z, 2),
                "ram": round(ram_z, 2),
                "bw":  round(bw_z, 2),
            },
        }

    # ── Workload rule-based label ─────────────────────────────────────────────

    @staticmethod
    def rule_based_workload(snap: SystemSnapshot) -> str:
        """
        Derives workload label from thresholds.
        Used to auto-label live_metrics.csv for retraining.
        """
        if snap.cpu_pct > 75 or snap.ram_pct > 85:
            return "Critical"
        elif snap.cpu_pct > 50 or snap.ram_pct > 70:
            return "High"
        elif snap.cpu_pct > 20 or snap.ram_pct > 50:
            return "Medium"
        else:
            return "Low"


# ── Math helpers ───────────────────────────────────────────────────────────

def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)

def _z_score(value: float, history: list) -> float:
    """Z-score of `value` relative to `history`."""
    if len(history) < 2:
        return 0.0
    m = _mean(history)
    s = _std(history)
    if s < 1e-9:
        return 0.0
    return (value - m) / s
