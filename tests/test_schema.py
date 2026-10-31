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
)


def test_normalize_columns():
    """Test column name normalization."""
    df = pd.DataFrame({"CPUUsage": [1.0], "RAM_Usage": [2.0], "Unknown": [3.0]})
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
    assert not pd.api.types.is_numeric_dtype(result["b"])  # b remains object because all non-numeric
    assert result["a"].iloc[0] == 1.0
    assert result["a"].iloc[1] == 2.0
    assert pd.isna(result["a"].iloc[2])  # invalid becomes NaN


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