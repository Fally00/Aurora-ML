"""
Tests for schema.py functions.
"""

import pandas as pd
import pytest

from Core.schema import (
    normalize_columns,
    ensure_columns,
    coerce_numeric_columns,
    normalize_anomaly_label,
    normalize_workload_label,
    SystemSnapshot,
    ANOMALY_FEATURE_COLS,
    CLASSIFIER_FEATURE_COLS,
    PREDICTOR_FEATURE_COLS,
)


def test_normalize_columns():
    """Test column name normalization."""
    df = pd.DataFrame({"CPUUsage": [1.0], "RAMUsage": [2.0], "Unknown": [3.0]})
    result = normalize_columns(df)
    assert "cpu_pct" in result.columns
    assert "ram_pct" in result.columns
    assert "Unknown" in result.columns
    assert result["cpu_pct"].iloc[0] == 1.0
    assert result["ram_pct"].iloc[0] == 2.0


def test_ensure_columns():
    """Test ensuring columns exist."""
    df = pd.DataFrame({"a": [1.0]})
    result = ensure_columns(df, ["a", "b", "c"], defaults={"b": 2.0, "c": 3.0})
    assert list(result.columns) == ["a", "b", "c"]
    assert result["a"].iloc[0] == 1.0
    assert result["b"].iloc[0] == 2.0
    assert result["c"].iloc[0] == 3.0


def test_coerce_numeric_columns():
    """Test coercing columns to numeric."""
    df = pd.DataFrame({"a": ["1.0", "2.0", "invalid"], "b": ["x", "y", "z"]})
    result = coerce_numeric_columns(df, ["a", "b"])
    assert pd.api.types.is_numeric_dtype(result["a"])
    # In pandas 3.x, all-NaN columns become float dtype
    assert pd.api.types.is_numeric_dtype(result["b"])
    assert result["a"].iloc[0] == 1.0
    assert result["a"].iloc[1] == 2.0
    assert pd.isna(result["a"].iloc[2])  # invalid becomes NaN
    assert pd.isna(result["b"].iloc[0])  # all non-numeric become NaN


def test_normalize_anomaly_label():
    """Test anomaly label normalization."""
    assert normalize_anomaly_label("normal") == "Normal"
    assert normalize_anomaly_label("alert") == "Alert"
    assert normalize_anomaly_label("1") == "Alert"
    assert normalize_anomaly_label("0") == "Normal"
    assert normalize_anomaly_label("true") == "Alert"
    assert normalize_anomaly_label("false") == "Normal"
    assert normalize_anomaly_label("benign") == "Normal"
    assert normalize_anomaly_label("malicious") == "Alert"
    assert normalize_anomaly_label("unknown") == "Unknown"  # falls back to title


def test_normalize_workload_label():
    """Test workload label normalization."""
    assert normalize_workload_label("low") == "Low"
    assert normalize_workload_label("medium") == "Medium"
    assert normalize_workload_label("high") == "High"
    assert normalize_workload_label("critical") == "Critical"
    assert normalize_workload_label("healthy") == "Low"
    assert normalize_workload_label("degraded") == "Medium"
    assert normalize_workload_label("cascading_failure") == "High"
    assert normalize_workload_label("total_outage") == "Critical"
    assert normalize_workload_label("unknown") == "Unknown"  # falls back to title


# ── Schema round-trip tests ──

def test_schema_roundtrip_anomaly():
    """Test anomaly schema: normalize -> ensure -> coerce preserves data."""
    raw = pd.DataFrame({
        "CPUUsage": [12.5, 45.0],
        "RAMUsage": [60.0, 70.0],
        "Bandwidth": [100.0, 200.0],
        "Packet_Rate": [10.0, 20.0],
        "protocol": ["TCP", "UDP"],
        "label": ["Normal", "Alert"],
    })
    # Round-trip
    df = normalize_columns(raw)
    df = ensure_columns(df, ANOMALY_FEATURE_COLS + ["label"])
    df = coerce_numeric_columns(df, ["cpu_pct", "ram_pct", "bandwidth_kbps", "packet_rate_pps"])

    assert list(df.columns) == ANOMALY_FEATURE_COLS + ["label"]
    assert len(df) == 2
    assert df["cpu_pct"].tolist() == [12.5, 45.0]
    assert df["protocol"].tolist() == ["TCP", "UDP"]
    assert df["label"].tolist() == ["Normal", "Alert"]


def test_schema_roundtrip_classifier():
    """Test classifier schema round-trip."""
    raw = pd.DataFrame({
        "CPUUsage": [12.5, 45.0, 80.0],
        "RAMUsage": [60.0, 70.0, 90.0],
        "Bandwidth": [100.0, 200.0, 300.0],
        "Packet_Rate": [10.0, 20.0, 30.0],
        "Task_Priority": ["Low", "Medium", "Critical"],
    })
    df = normalize_columns(raw)
    df = ensure_columns(df, CLASSIFIER_FEATURE_COLS + ["task_priority"])
    df = coerce_numeric_columns(df, CLASSIFIER_FEATURE_COLS)

    assert list(df.columns) == CLASSIFIER_FEATURE_COLS + ["task_priority"]
    assert len(df) == 3
    assert df["task_priority"].tolist() == ["Low", "Medium", "Critical"]


def test_schema_roundtrip_predictor():
    """Test predictor schema round-trip."""
    raw = pd.DataFrame({
        "CPUUsage": [12.5, 45.0],
        "RAMUsage": [60.0, 70.0],
        "DiskUsage": [30.0, 40.0],
        "Bandwidth": [100.0, 200.0],
        "Packet_Rate": [10.0, 20.0],
    })
    df = normalize_columns(raw)
    df = ensure_columns(df, PREDICTOR_FEATURE_COLS)
    df = coerce_numeric_columns(df, PREDICTOR_FEATURE_COLS)

    assert list(df.columns) == PREDICTOR_FEATURE_COLS
    assert len(df) == 2


def test_system_snapshot_to_anomaly_input():
    """Test SystemSnapshot.to_anomaly_input() matches ANOMALY_FEATURE_COLS."""
    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
        protocol="TCP",
    )
    inp = snap.to_anomaly_input()
    assert set(inp.keys()) == set(ANOMALY_FEATURE_COLS)
    assert inp["cpu_pct"] == 50.0
    assert inp["protocol"] == "TCP"


def test_system_snapshot_to_classifier_input():
    """Test SystemSnapshot.to_classifier_input() matches CLASSIFIER_FEATURE_COLS."""
    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    inp = snap.to_classifier_input()
    assert set(inp.keys()) == set(CLASSIFIER_FEATURE_COLS)


def test_system_snapshot_to_predictor_input():
    """Test SystemSnapshot.to_predictor_input() matches PREDICTOR_FEATURE_COLS."""
    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        disk_pct=30.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
        cpu_delta=1.0,
        ram_delta=0.5,
        cpu_rolling_mean=48.0,
        ram_rolling_mean=59.0,
        cpu_rolling_std=2.0,
        ram_rolling_std=1.0,
        bw_rolling_mean=95.0,
        bw_spike=1.05,
    )
    inp = snap.to_predictor_input()
    assert set(inp.keys()) == set(PREDICTOR_FEATURE_COLS)
    assert inp["cpu_delta"] == 1.0
    assert inp["bw_spike"] == 1.05