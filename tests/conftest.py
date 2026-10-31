"""Shared fixtures for the SIT-Py test suite."""

import hashlib
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from Core.anomaly import FEATURE_COLS as ANOMALY_FEATURES, _build_pipeline
from Core.classifier import CLASS_NAMES, FEATURE_COLS as CLASSIFIER_FEATURES
from Core.predictor import FEATURE_COLS as PREDICTOR_FEATURES
from Core.schema import SystemSnapshot


@pytest.fixture
def mock_system_snapshot():
    """Return representative system telemetry for model tests."""
    return SystemSnapshot(
        timestamp="2026-05-07T12:00:00",
        cpu_pct=50.0,
        ram_pct=60.0,
        disk_pct=30.0,
        bandwidth_kbps=100.0,
        packet_rate_pps=10.0,
        temperature_c=45.0,
        fan_rpm=1500,
        process_count=100,
        task_priority="Medium",
        protocol="TCP",
        attack_type="",
        severity="",
        threat_type="",
    )


def _save_with_hash(obj, directory, filename):
    path = os.path.join(directory, filename)
    joblib.dump(obj, path)
    with open(path, "rb") as artifact:
        digest = hashlib.sha256(artifact.read()).hexdigest()
    with open(path + ".sha256", "w", encoding="ascii") as sidecar:
        sidecar.write(digest + "\n")


@pytest.fixture
def temp_model_dir():
    """Create complete, checksum-verified model bundles for inference tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(42)

        # Anomaly features: numeric + categorical (protocol as string)
        anomaly_x = pd.DataFrame({
            "cpu_pct": rng.normal(size=24),
            "ram_pct": rng.normal(size=24),
            "bandwidth_kbps": rng.normal(size=24),
            "packet_rate_pps": rng.normal(size=24),
            "protocol": rng.choice(["TCP", "UDP", "ICMP"], size=24),
        })
        anomaly_y = np.array(["Normal", "Alert"] * 12)
        anomaly_model = _build_pipeline()
        anomaly_model.fit(anomaly_x, anomaly_y)
        _save_with_hash(anomaly_model, tmpdir, "anomaly_model.pkl")

        classifier_x = rng.normal(size=(40, len(CLASSIFIER_FEATURES)))
        classifier_y = np.tile(np.arange(len(CLASS_NAMES)), 10)
        scaler = StandardScaler().fit(classifier_x)
        classifier_model = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier_model.fit(scaler.transform(classifier_x), classifier_y)
        encoder = LabelEncoder().fit(CLASS_NAMES)
        _save_with_hash(classifier_model, tmpdir, "classifier_model.pkl")
        _save_with_hash(encoder, tmpdir, "classifier_target_encoder.pkl")
        _save_with_hash(scaler, tmpdir, "classifier_scaler.pkl")

        predictor_x = rng.normal(size=(40, len(PREDICTOR_FEATURES)))
        predictor_y = rng.uniform(0, 100, size=(40, 3))
        predictor_scaler = StandardScaler().fit(predictor_x)
        predictor_model = RandomForestRegressor(n_estimators=10, random_state=42)
        predictor_model.fit(predictor_scaler.transform(predictor_x), predictor_y)
        _save_with_hash(predictor_model, tmpdir, "predictor_model.pkl")
        _save_with_hash(predictor_scaler, tmpdir, "predictor_scaler.pkl")

        yield tmpdir
