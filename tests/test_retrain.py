"""
Tests for pipeline/retrain.py functions.
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline.retrain import (
    _build_anomaly_dataset,
    _build_classifier_dataset,
    _build_predictor_dataset,
    _read_csv_safe,
    _rule_based_label,
    should_retrain,
)


def test_read_csv_safe():
    """Test reading a CSV file, including missing file."""
    # Test missing file
    df = _read_csv_safe("non_existent_file.csv")
    assert df.empty

    # Test existing file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n4,5,6")
        temp_path = f.name
    try:
        df = _read_csv_safe(temp_path)
        assert not df.empty
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2
    finally:
        os.unlink(temp_path)


def test_rule_based_label():
    """Test rule-based label from CPU and RAM."""
    # Using default thresholds from config: Critical: cpu>75, ram>85; High: cpu>50, ram>70; Medium: cpu>20, ram>50
    assert _rule_based_label(80.0, 90.0) == "Critical"  # both over critical
    assert _rule_based_label(60.0, 80.0) == "High"      # cpu over high, ram over critical -> high? Actually, the function checks critical first, so if either cpu or ram over critical, it returns Critical. Here ram=80>85? No, 80<85, cpu=60<75, so not critical. Then high: cpu>50 (60>50) or ram>70 (80>70) -> yes, so High.
    assert _rule_based_label(30.0, 60.0) == "Medium"    # cpu>20 (30>20) or ram>50 (60>50) -> Medium
    assert _rule_based_label(10.0, 40.0) == "Low"       # neither over medium thresholds


def test_build_predictor_dataset():
    """Test building predictor dataset with mock data."""
    # Create mock live_df and log_df
    live_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:02"],
        "cpu_pct": [50.0, 55.0],
        "ram_pct": [60.0, 65.0],
        "disk_pct": [30.0, 35.0],
        "bandwidth_kbps": [100.0, 110.0],
        "packet_rate_pps": [10.0, 12.0],
    })
    log_df = pd.DataFrame()  # empty

    # We need to set the PREDICTOR_DATASET_CSV to a temporary file or empty.
    # We'll mock the config.PREDICTOR_DATASET_CSV to point to a temporary empty CSV.
    with patch("pipeline.retrain.PREDICTOR_DATASET_CSV", ""), \
         patch("pipeline.retrain.WINDOW_SIZE", 1):
        # With WINDOW_SIZE=1, we need at least WINDOW_SIZE+1 = 2 rows to build windows.
        # Our live_df has 2 rows, so it should work.
        result = _build_predictor_dataset(live_df, log_df)
        # We expect the result to have the runtime rows (since base is empty)
        # The function returns the merged DataFrame of raw columns.
        # We don't need to check the exact values, just that it's not empty and has the expected columns.
        assert not result.empty
        assert set(result.columns) == set(["cpu_pct", "ram_pct", "disk_pct", "bandwidth_kbps", "packet_rate_pps"])


def test_build_classifier_dataset():
    """Test building classifier dataset with mock data."""
    live_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:02"],
        "cpu_pct": [50.0, 55.0],
        "ram_pct": [60.0, 65.0],
        "bandwidth_kbps": [100.0, 110.0],
        "packet_rate_pps": [10.0, 12.0],
        "task_priority": ["Medium", "High"],
    })
    log_df = pd.DataFrame()

    with patch("pipeline.retrain.CLASSIFIER_DATASET_CSV", ""), \
         patch("pipeline.retrain.LIVE_DATA_WEIGHT", 1), \
         patch("pipeline.retrain.LABELED_LOG_WEIGHT", 2), \
         patch("pipeline.retrain.RANDOM_STATE", 42):
        result = _build_classifier_dataset(live_df, log_df)
        assert not result.empty
        assert "task_priority" in result.columns
        assert set(result["task_priority"].unique()).issubset({"Low", "Medium", "High", "Critical"})


def test_build_anomaly_dataset():
    """Test building anomaly dataset with mock data."""
    live_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:02"],
        "cpu_pct": [50.0, 55.0],
        "ram_pct": [60.0, 65.0],
        "bandwidth_kbps": [100.0, 110.0],
        "packet_rate_pps": [10.0, 12.0],
        "protocol": ["TCP", "UDP"],
        "attack_type": ["", ""],
        "severity": ["", ""],
        "threat_type": ["", ""],
        "label": ["Normal", "Alert"],
    })
    log_df = pd.DataFrame()  # empty

    with patch("pipeline.retrain.ANOMALY_DATASET_CSV", ""), \
         patch("pipeline.retrain.LABELED_LOG_WEIGHT", 2):
        result = _build_anomaly_dataset(live_df, log_df)
        assert not result.empty
        assert "label" in result.columns
        assert set(result["label"].unique()).issubset({"Normal", "Alert"})


def test_should_retrain():
    """Test the should_retrain function."""
    # We need to mock the runtime row count to be above or below threshold.
    # We'll mock _runtime_row_count.
    with patch("pipeline.retrain._runtime_row_count", return_value=600), \
         patch("pipeline.retrain.RETRAIN_ROW_THRESHOLD", 500):
        assert should_retrain(pd.DataFrame(), pd.DataFrame()) == True

    with patch("pipeline.retrain._runtime_row_count", return_value=400), \
         patch("pipeline.retrain.RETRAIN_ROW_THRESHOLD", 500):
        assert should_retrain(pd.DataFrame(), pd.DataFrame()) == False