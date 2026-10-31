"""
Tests for model classes.
"""

import os

import pytest

from Core.anomaly import AnomalyDetector
from Core.classifier import WorkloadClassifier
from Core.predictor import ResourcePredictor
from Core.schema import SystemSnapshot


def test_anomaly_detector_load_and_predict(temp_model_dir):
    """Test AnomalyDetector can load and predict."""
    detector = AnomalyDetector(models_dir=temp_model_dir)
    # Load should succeed
    assert detector.load() == True
    assert detector.loaded == True

    # Create a mock snapshot
    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
        protocol="TCP",
        attack_type="",
        severity="",
        threat_type="",
    )
    # Predict should return a dict with expected keys
    result = detector.predict(snap.to_anomaly_input())
    assert "label" in result
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert "confidence" in result
    assert result["label"] in ["Normal", "Alert"]
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert isinstance(result["is_anomaly"], bool)
    assert 0.0 <= result["confidence"] <= 1.0


def test_workload_classifier_load_and_predict(temp_model_dir):
    """Test WorkloadClassifier can load and predict."""
    classifier = WorkloadClassifier(models_dir=temp_model_dir)
    assert classifier.load() == True
    assert classifier.loaded == True

    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    result = classifier.predict(snap.to_classifier_input())
    assert "workload" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["workload"] in ["Low", "Medium", "High", "Critical"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["probabilities"], dict)
    assert set(result["probabilities"].keys()) == {"Low", "Medium", "High", "Critical"}


def test_resource_predictor_load_and_predict(temp_model_dir):
    """Test ResourcePredictor can load and predict."""
    predictor = ResourcePredictor(models_dir=temp_model_dir)
    assert predictor.load() == True
    assert predictor.loaded == True

    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
    )
    # Predictor expects the feature dict from to_predictor_input
    features = snap.to_predictor_input()
    result = predictor.predict(features)
    assert "cpu_pct" in result
    assert "ram_pct" in result
    assert "disk_pct" in result
    assert isinstance(result["cpu_pct"], float)
    assert isinstance(result["ram_pct"], float)
    assert isinstance(result["disk_pct"], float)
    assert 0.0 <= result["cpu_pct"] <= 100.0
    assert 0.0 <= result["ram_pct"] <= 100.0
    assert 0.0 <= result["disk_pct"] <= 100.0


def test_model_not_loaded_raises(temp_model_dir):
    """Test that predicting before loading raises RuntimeError."""
    detector = AnomalyDetector(models_dir=temp_model_dir)
    # Do not load
    snap = SystemSnapshot(
        cpu_pct=50.0,
        ram_pct=60.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
        protocol="TCP",
        attack_type="",
        severity="",
        threat_type="",
    )
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.predict(snap.to_anomaly_input())

    classifier = WorkloadClassifier(models_dir=temp_model_dir)
    with pytest.raises(RuntimeError, match="not loaded"):
        classifier.predict(snap.to_classifier_input())

    predictor = ResourcePredictor(models_dir=temp_model_dir)
    with pytest.raises(RuntimeError, match="not loaded"):
        predictor.predict(snap.to_predictor_input())