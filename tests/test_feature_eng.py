"""
Tests for feature_engineering.py functions.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from Core.feature_engineering import FeatureEngineer, _apply_cold_start_defaults
from Core.schema import SystemSnapshot


def test_apply_cold_start_defaults():
    """Test cold start defaults application."""
    snap = SystemSnapshot(
        timestamp="2026-05-07T12:00:00",
        cpu_pct=50.0,
        ram_pct=60.0,
        disk_pct=30.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    _apply_cold_start_defaults(snap)
    assert snap.cpu_rolling_mean == 50.0
    assert snap.ram_rolling_mean == 60.0
    assert snap.bw_rolling_mean == 100.0
    assert snap.cpu_rolling_std == 0.0
    assert snap.ram_rolling_std == 0.0
    assert snap.cpu_delta == 0.0
    assert snap.ram_delta == 0.0
    assert snap.bw_spike == 1.0


def test_enrich_first_call():
    """Test enrich on first call (no history)."""
    engineer = FeatureEngineer()
    snap = SystemSnapshot(
        timestamp="2026-05-07T12:00:00",
        cpu_pct=50.0,
        ram_pct=60.0,
        disk_pct=30.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    result = engineer.enrich(snap)
    # Should have applied cold start defaults
    assert result.cpu_rolling_mean == 50.0
    assert result.ram_rolling_mean == 60.0
    assert result.bw_rolling_mean == 100.0
    assert result.cpu_rolling_std == 0.0
    assert result.ram_rolling_std == 0.0
    assert result.cpu_delta == 0.0
    assert result.ram_delta == 0.0
    assert result.bw_spike == 1.0
    # History should have one entry
    assert engineer.history_len() == 1


def test_enrich_subsequent_calls():
    """Test enrich with existing history."""
    engineer = FeatureEngineer(window_size=3)
    # Add some history
    snap1 = SystemSnapshot(
        timestamp="2026-05-07T12:00:00",
        cpu_pct=40.0,
        ram_pct=50.0,
        bandwidth_kbps=80.0,
        packet_rate_pps=5.0,
    )
    snap2 = SystemSnapshot(
        timestamp="2026-05-07T12:00:01",
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=90.0,
        packet_rate_pps=6.0,
    )
    engineer.enrich(snap1)
    engineer.enrich(snap2)

    snap3 = SystemSnapshot(
        timestamp="2026-05-07T12:00:02",
        cpu_pct=60.0,
        ram_pct=70.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=7.0,
    )
    result = engineer.enrich(snap3)

    # Check rolling means: history has snap1, snap2, snap3
    # cpu: 40, 50, 60 -> mean = 50.0
    assert result.cpu_rolling_mean == 50.0
    # ram: 50, 60, 70 -> mean = 60.0
    assert result.ram_rolling_mean == 60.0
    # bw: 80, 90, 100 -> mean = 90.0
    assert result.bw_rolling_mean == 90.0
    # Check deltas: snap3 - snap2
    assert result.cpu_delta == 10.0  # 60 - 50
    assert result.ram_delta == 10.0  # 70 - 60
    # Check bw_spike: bw_rolling_mean = 90, current bw = 100 -> spike = 100/90
    assert result.bw_spike == 100.0 / 90.0
    # History should have 3 entries
    assert engineer.history_len() == 3


def test_statistical_anomaly_warmup():
    """Test statistical_anomaly when not enough history."""
    engineer = FeatureEngineer(bridge_min_history=5)
    snap = SystemSnapshot(
        timestamp="2026-05-07T12:00:00",
        cpu_pct=50.0,
        ram_pct=60.0,
        disk_pct=30.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    # Add only 2 history entries (less than bridge_min_history=5)
    engineer.enrich(SystemSnapshot(cpu_pct=40.0, ram_pct=50.0, bandwidth_kbps=80.0, packet_rate_pps=5.0))
    engineer.enrich(SystemSnapshot(cpu_pct=50.0, ram_pct=60.0, bandwidth_kbps=90.0, packet_rate_pps=6.0))
    result = engineer.statistical_anomaly(snap)
    assert result["source"] == "stat_bridge_warmup"
    assert result["is_anomaly"] == False
    assert result["anomaly_score"] == 0.0


def test_statistical_anomaly_detection():
    """Test statistical_anomaly detects an anomaly."""
    engineer = FeatureEngineer(bridge_min_history=3)
    # Add history with stable values
    for i in range(3):
        engineer.enrich(SystemSnapshot(
            cpu_pct=50.0, ram_pct=50.0, bandwidth_kbps=100.0, packet_rate_pps=10.0
        ))
    # Now test with an outlier
    snap = SystemSnapshot(
        cpu_pct=200.0,  # extreme outlier
        ram_pct=50.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    result = engineer.statistical_anomaly(snap)
    # Should detect anomaly based on cpu z-score
    assert result["is_anomaly"] == True
    assert result["label"] == "Alert"
    assert result["anomaly_score"] > 0.0
    assert "z_scores" in result
    assert result["z_scores"]["cpu"] > 2.0  # should be high