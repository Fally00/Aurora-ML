"""
Shared test fixtures for SIT-Py test suite.
"""

import os
import tempfile
from unittest.mock import MagicMock

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from Core.schema import SystemSnapshot


@pytest.fixture
def mock_system_snapshot():
    """Return a mock SystemSnapshot for testing."""
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


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory with mock model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple mock model for anomaly detector
        anomaly_model = RandomForestClassifier(n_estimators=10, random_state=42)
        # We need to fit it to have a valid model, but for simplicity we'll create an unfitted model
        # and mock the predict method. Alternatively, we can fit it on dummy data.
        # Let's create a dummy dataset to fit the model.
        import numpy as np
        X = np.random.rand(10, 5)  # 5 features
        y = np.random.randint(0, 2, 10)
        anomaly_model.fit(X, y)

        anomaly_model_path = os.path.join(tmpdir, "anomaly_model.pkl")
        joblib.dump(anomaly_model, anomaly_model_path)

        # Create hash file for anomaly model
        with open(anomaly_model_path, "rb") as f:
            model_hash = joblib.hash(anomaly_model) if hasattr(joblib, 'hash') else __import__('hashlib').sha256(f.read()).hexdigest()
        # Actually, we need to compute the hash of the file, not the model object.
        # Let's compute the hash of the file we just wrote.
        with open(anomaly_model_path, "rb") as f:
            file_hash = __import__('hashlib').sha256(f.read()).hexdigest()
        with open(anomaly_model_path + ".sha256", "w") as f:
            f.write(file_hash)

        # Create a simple mock model for classifier
        classifier_model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_clf = np.random.rand(10, 4)  # 4 features
        y_clf = np.random.randint(0, 4, 10)  # 4 classes
        classifier_model.fit(X_clf, y_clf)

        classifier_model_path = os.path.join(tmpdir, "classifier_model.pkl")
        joblib.dump(classifier_model, classifier_model_path)
        with open(classifier_model_path, "rb") as f:
            file_hash = __import__('hashlib').sha256(f.read()).hexdigest()
        with open(classifier_model_path + ".sha256", "w") as f:
            f.write(file_hash)

        # Create a simple mock model for predictor
        predictor_model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_pred = np.random.rand(10, 8)  # 8 features
        y_pred = np.random.rand(10, 3)  # 3 targets
        predictor_model.fit(X_pred, y_pred)

        predictor_model_path = os.path.join(tmpdir, "predictor_model.pkl")
        joblib.dump(predictor_model, predictor_model_path)
        with open(predictor_model_path, "rb") as f:
            file_hash = __import__('hashlib').sha256(f.read()).hexdigest()
        with open(predictor_model_path + ".sha256", "w") as f:
            f.write(file_hash)

        # Also create a simple scaler for predictor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_pred)
        scaler_path = os.path.join(tmpdir, "predictor_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        with open(scaler_path, "rb") as f:
            file_hash = __import__('hashlib').sha256(f.read()).hexdigest()
        with open(scaler_path + ".sha256", "w") as f:
            f.write(file_hash)

        yield tmpdir