"""
Tests for pipeline/retrain.py functions.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.retrain import (
    _build_anomaly_dataset,
    _build_classifier_dataset,
    _build_predictor_dataset,
    _read_csv_safe,
    _validate_resource_bounds,
    _rule_based_label,
    should_retrain,
    _load_and_prepare_predictor_runtime,
    _load_and_prepare_classifier_runtime,
    _load_and_prepare_anomaly_runtime,
    _combine_predictor_runtime,
    _combine_classifier_runtime,
    _merge_predictor_runtime_and_base,
    _merge_classifier_runtime_and_base,
    _merge_anomaly_runtime_and_base,
)


def test_validate_resource_bounds_clips_outliers():
    frame = pd.DataFrame({"cpu_pct": [-1, 50, 105], "ram_pct": [101, 40, 60]})
    result = _validate_resource_bounds(frame)
    assert result["cpu_pct"].tolist() == [0, 50, 100]
    assert result["ram_pct"].tolist() == [100, 40, 60]


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
    # Need at least WINDOW_SIZE + 2 rows to build windows and pass the merge check
    live_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:02", "2026-05-07T12:00:04", "2026-05-07T12:00:06"],
        "cpu_pct": [50.0, 55.0, 60.0, 65.0],
        "ram_pct": [60.0, 65.0, 70.0, 75.0],
        "disk_pct": [30.0, 35.0, 40.0, 45.0],
        "bandwidth_kbps": [100.0, 110.0, 120.0, 130.0],
        "packet_rate_pps": [10.0, 12.0, 14.0, 16.0],
    })
    log_df = pd.DataFrame()

    with patch("pipeline.retrain.PREDICTOR_DATASET_CSV", ""), \
         patch("pipeline.retrain.WINDOW_SIZE", 1):
        result = _build_predictor_dataset(live_df, log_df)
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
    """Test building anomaly dataset with mock data.

    Note: anomaly runtime data comes from log_df (sit_log.csv), not live_df.
    """
    live_df = pd.DataFrame()  # live_df not used for anomaly
    log_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:02", "2026-05-07T12:00:04", "2026-05-07T12:00:06"],
        "cpu_pct": [50.0, 55.0, 60.0, 65.0],
        "ram_pct": [60.0, 65.0, 70.0, 75.0],
        "bandwidth_kbps": [100.0, 110.0, 120.0, 130.0],
        "packet_rate_pps": [10.0, 12.0, 14.0, 16.0],
        "protocol": ["TCP", "UDP", "TCP", "UDP"],
        "anomaly_label": ["Normal", "Alert", "Normal", "Alert"],  # log_df uses anomaly_label
    })

    with patch("pipeline.retrain.ANOMALY_DATASET_CSV", ""), \
         patch("pipeline.retrain.LABELED_LOG_WEIGHT", 2):
        result = _build_anomaly_dataset(live_df, log_df)
        assert not result.empty
        assert "label" in result.columns
        assert set(result["label"].unique()).issubset({"Normal", "Alert"})


def test_should_retrain():
    """Test the should_retrain function."""
    with patch("pipeline.retrain._runtime_row_count", return_value=600), \
         patch("pipeline.retrain.RETRAIN_ROW_THRESHOLD", 500):
        assert should_retrain(pd.DataFrame(), pd.DataFrame()) == True

    with patch("pipeline.retrain._runtime_row_count", return_value=400), \
         patch("pipeline.retrain.RETRAIN_ROW_THRESHOLD", 500):
        assert should_retrain(pd.DataFrame(), pd.DataFrame()) == False


# ── Edge case tests for dataset builders ──

def test_build_predictor_dataset_empty_runtime():
    """Test predictor dataset with empty runtime data falls back to base."""
    live_df = pd.DataFrame()
    log_df = pd.DataFrame()

    # Create a mock base CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        base_df = pd.DataFrame({
            "cpu_pct": [10, 20, 30, 40, 50, 60],
            "ram_pct": [30, 40, 50, 60, 70, 80],
            "disk_pct": [10, 15, 20, 25, 30, 35],
            "bandwidth_kbps": [100, 110, 120, 130, 140, 150],
            "packet_rate_pps": [10, 11, 12, 13, 14, 15],
        })
        base_df.to_csv(f.name, index=False)
        base_path = f.name

    try:
        with patch("pipeline.retrain.PREDICTOR_DATASET_CSV", base_path), \
             patch("pipeline.retrain.WINDOW_SIZE", 2):
            result = _build_predictor_dataset(live_df, log_df)
            assert not result.empty
            assert len(result) == 6  # base rows preserved
    finally:
        os.unlink(base_path)


def test_build_classifier_dataset_empty_runtime():
    """Test classifier dataset with empty runtime data."""
    live_df = pd.DataFrame()
    log_df = pd.DataFrame()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        base_df = pd.DataFrame({
            "cpu_pct": [10, 50, 80],
            "ram_pct": [30, 60, 90],
            "bandwidth_kbps": [100, 200, 300],
            "packet_rate_pps": [10, 20, 30],
            "task_priority": ["Low", "Medium", "Critical"],
        })
        base_df.to_csv(f.name, index=False)
        base_path = f.name

    try:
        with patch("pipeline.retrain.CLASSIFIER_DATASET_CSV", base_path), \
             patch("pipeline.retrain.LIVE_DATA_WEIGHT", 1), \
             patch("pipeline.retrain.RANDOM_STATE", 42):
            result = _build_classifier_dataset(live_df, log_df)
            assert not result.empty
            assert "task_priority" in result.columns
    finally:
        os.unlink(base_path)


def test_build_anomaly_dataset_single_class_runtime():
    """Test anomaly dataset skips runtime when single-class."""
    live_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:02"],
        "cpu_pct": [50.0, 55.0],
        "ram_pct": [60.0, 65.0],
        "bandwidth_kbps": [100.0, 110.0],
        "packet_rate_pps": [10.0, 12.0],
        "protocol": ["TCP", "TCP"],
        "label": ["Normal", "Normal"],  # single class
    })
    log_df = pd.DataFrame()

    with patch("pipeline.retrain.ANOMALY_DATASET_CSV", ""), \
         patch("pipeline.retrain.LABELED_LOG_WEIGHT", 2):
        result = _build_anomaly_dataset(live_df, log_df)
        # Should return empty because runtime is single-class and no base
        assert result.empty


def test_build_anomaly_dataset_with_base():
    """Test anomaly dataset merges runtime with base."""
    live_df = pd.DataFrame({
        "timestamp": ["2026-05-07T12:00:00"],
        "cpu_pct": [50.0],
        "ram_pct": [60.0],
        "bandwidth_kbps": [100.0],
        "packet_rate_pps": [10.0],
        "protocol": ["TCP"],
        "label": ["Alert"],
    })
    log_df = pd.DataFrame()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        base_df = pd.DataFrame({
            "cpu_pct": [10, 20, 30],
            "ram_pct": [30, 40, 50],
            "bandwidth_kbps": [100, 110, 120],
            "packet_rate_pps": [10, 11, 12],
            "protocol": ["TCP", "UDP", "TCP"],
            "label": ["Normal", "Normal", "Alert"],
        })
        base_df.to_csv(f.name, index=False)
        base_path = f.name

    try:
        with patch("pipeline.retrain.ANOMALY_DATASET_CSV", base_path), \
             patch("pipeline.retrain.LABELED_LOG_WEIGHT", 2):
            result = _build_anomaly_dataset(live_df, log_df)
            assert not result.empty
            assert "label" in result.columns
            assert set(result["label"].unique()) == {"Normal", "Alert"}
    finally:
        os.unlink(base_path)


def test_combine_predictor_runtime_sorts_and_deduplicates():
    """Test runtime combiner sorts by timestamp and deduplicates."""
    frames = [
        pd.DataFrame({
            "timestamp": ["2026-05-07T12:00:02", "2026-05-07T12:00:00"],
            "cpu_pct": [55.0, 50.0],
            "ram_pct": [65.0, 60.0],
            "disk_pct": [35.0, 30.0],
            "bandwidth_kbps": [110.0, 100.0],
            "packet_rate_pps": [12.0, 10.0],
        }),
        pd.DataFrame({
            "timestamp": ["2026-05-07T12:00:00", "2026-05-07T12:00:04"],  # duplicate timestamp
            "cpu_pct": [51.0, 60.0],
            "ram_pct": [61.0, 70.0],
            "disk_pct": [31.0, 40.0],
            "bandwidth_kbps": [101.0, 120.0],
            "packet_rate_pps": [11.0, 14.0],
        }),
    ]
    result = _combine_predictor_runtime(frames)
    assert len(result) == 3  # deduplicated
    # timestamp column is dropped, check cpu_pct values in order
    assert result["cpu_pct"].tolist() == [51.0, 55.0, 60.0]


def test_merge_predictor_runtime_and_base():
    """Test merging runtime and base predictor data."""
    # Need > WINDOW_SIZE + 1 = 2 rows for runtime to be used
    runtime = pd.DataFrame({
        "cpu_pct": [50.0, 55.0, 60.0],
        "ram_pct": [60.0, 65.0, 70.0],
        "disk_pct": [30.0, 35.0, 40.0],
        "bandwidth_kbps": [100.0, 110.0, 120.0],
        "packet_rate_pps": [10.0, 12.0, 14.0],
    })
    base = pd.DataFrame({
        "cpu_pct": [10.0, 20.0],
        "ram_pct": [30.0, 40.0],
        "disk_pct": [10.0, 15.0],
        "bandwidth_kbps": [50.0, 60.0],
        "packet_rate_pps": [5.0, 6.0],
    })

    with patch("pipeline.retrain.WINDOW_SIZE", 1):
        result = _merge_predictor_runtime_and_base(runtime, base)
        assert len(result) == 5  # 3 runtime + 2 base


def test_merge_classifier_runtime_and_base_weights():
    """Test classifier merge respects LIVE_DATA_WEIGHT."""
    runtime = pd.DataFrame({
        "cpu_pct": [50.0],
        "ram_pct": [60.0],
        "bandwidth_kbps": [100.0],
        "packet_rate_pps": [10.0],
        "task_priority": ["Medium"],
    })
    base = pd.DataFrame({
        "cpu_pct": [10.0],
        "ram_pct": [30.0],
        "bandwidth_kbps": [50.0],
        "packet_rate_pps": [5.0],
        "task_priority": ["Low"],
    })

    with patch("pipeline.retrain.LIVE_DATA_WEIGHT", 3):
        result = _merge_classifier_runtime_and_base(base, runtime)
        # base (1) + runtime * 3 = 4 rows
        assert len(result) == 4
        # Runtime should appear 3 times
        medium_count = len(result[result["task_priority"] == "Medium"])
        assert medium_count == 3


def test_merge_anomaly_runtime_and_base_weights():
    """Test anomaly merge respects LABELED_LOG_WEIGHT."""
    runtime = pd.DataFrame({
        "cpu_pct": [50.0],
        "ram_pct": [60.0],
        "bandwidth_kbps": [100.0],
        "packet_rate_pps": [10.0],
        "protocol": ["TCP"],
        "label": ["Alert"],
    })
    base = pd.DataFrame({
        "cpu_pct": [10.0],
        "ram_pct": [30.0],
        "bandwidth_kbps": [50.0],
        "packet_rate_pps": [5.0],
        "protocol": ["TCP"],
        "label": ["Normal"],
    })

    with patch("pipeline.retrain.LABELED_LOG_WEIGHT", 2):
        result = _merge_anomaly_runtime_and_base(base, runtime)
        # base (1) + runtime * 2 = 3 rows
        assert len(result) == 3
        alert_count = len(result[result["label"] == "Alert"])
        assert alert_count == 2
