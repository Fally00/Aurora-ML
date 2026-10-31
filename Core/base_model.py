"""
Base model class for SIT-Py models to eliminate duplicate loading logic.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import joblib


class BaseModel:
    """Base class for model loading and artifact management."""

    model_files: list[str] = []  # override in subclass

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.loaded = False
        self._artifacts = {}

    def load(self) -> bool:
        """Load model artifacts if they exist, else stay in bridge mode."""
        for fname in self.model_files:
            path = os.path.join(self.models_dir, fname)
            hash_path = path + ".sha256"
            if not os.path.exists(path):
                logging.warning(f"Model file not found: {fname}")
                self.loaded = False
                return False
            if not os.path.exists(hash_path):
                logging.warning(f"Hash file not found for model file: {fname}")
                self.loaded = False
                return False
            # Compute hash of model file
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            # Read expected hash from hash file
            with open(hash_path, "r") as f:
                expected_hash = f.read().strip()
            if file_hash != expected_hash:
                logging.warning(f"Hash mismatch for model file: {fname}")
                self.loaded = False
                return False
            self._artifacts[fname] = joblib.load(path)
        self.loaded = True
        return True

    def _require_loaded(self):
        """Raise RuntimeError if model not loaded."""
        if not self.loaded:
            raise RuntimeError(f"{self.__class__.__name__} model not loaded. Call load() first.")

    def get_artifact(self, fname: str):
        """Get a loaded artifact by filename."""
        return self._artifacts.get(fname)