# SIT-Py (Python) — Fix & Improvement Plan

---

## Phase 1 — Critical Patches (Do First)

### 1.1 Replace Pickle with Safe Serialization
**Files:** `Core/anomaly.py:197`, `Core/classifier.py:188-192`, `Core/predictor.py:195-202`  
**Problem:** `joblib.load` deserializes pickle files which can execute arbitrary code. If a model file gets swapped out (accidentally or maliciously), you're cooked.

**Option A — ONNX (Recommended for production):**
```python
# Export during training
from skl2onnx import convert_sklearn
model_onnx = convert_sklearn(model, initial_types=[...])
with open("model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

# Load for inference
import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")
```

**Option B — safetensors or joblib with trust boundary (quick fix):**
If ONNX is too heavy for now, at minimum:
1. Store model files outside the repo (never commit them).
2. Verify a checksum/hash of the model file before loading:
   ```python
   import hashlib

   def safe_load(path: str, expected_sha256: str):
       data = Path(path).read_bytes()
       actual = hashlib.sha256(data).hexdigest()
       if actual != expected_sha256:
           raise ValueError(f"Model file tampered or corrupted: {path}")
       return joblib.load(path)
   ```
3. Store expected hashes in `config.py` or a sidecar `.sha256` file generated at train time.

> Start with Option B now, migrate to ONNX later when ready.

---

### 1.2 Add Unit Tests
**Location:** New `tests/` directory at project root  
**Problem:** Zero tests on an ML system making predictions about live hardware. Any silent regression is invisible.

**Setup:**
```bash
pip install pytest pytest-cov
```

**Files to create:**
```
tests/
  test_schema.py       — normalize_columns, ensure_columns, coerce_numeric_columns
  test_feature_eng.py  — enrich(), statistical_anomaly(), cold start defaults
  test_models.py       — load/predict cycle for all 3 models with mock data
  test_retrain.py      — _read_csv_safe, dataset builder functions
  conftest.py          — shared fixtures (mock SystemSnapshot, mock model files)
```

**Minimum viable test for each model:**
```python
def test_anomaly_detector_predict(mock_snapshot, mock_model_path):
    detector = AnomalyDetector()
    detector.load()
    result = detector.predict(mock_snapshot)
    assert "is_anomaly" in result
    assert isinstance(result["confidence"], float)
```

**Run coverage:**
```bash
pytest tests/ --cov=Core --cov=Data --cov=pipeline --cov-report=term-missing
```

---

## Phase 2 — High Priority Fixes

### 2.1 Move All Hyperparameters to config.py
**Files:** `Core/anomaly.py:106-111`, `Core/classifier.py:142-146`, `Core/predictor.py:150-156`  
**Problem:** Hyperparameters scattered across 3 files. Tuning requires touching model code instead of config.

**In `config.py`, add:**
```python
# --- Model Hyperparameters ---
ANOMALY_MODEL_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

CLASSIFIER_MODEL_PARAMS = {
    "n_estimators": 250,
    "random_state": 42,
    "n_jobs": -1,
}

PREDICTOR_MODEL_PARAMS = {
    "n_estimators": 250,
    "max_depth": 16,
    "random_state": 42,
    "n_jobs": -1,
}
```

**In each model file:**
```python
from config import ANOMALY_MODEL_PARAMS
self.model = RandomForestClassifier(**ANOMALY_MODEL_PARAMS)
```

---

### 2.2 Add a Base Model Class
**Files:** `Core/anomaly.py`, `Core/classifier.py`, `Core/predictor.py`  
**Problem:** All 3 model classes duplicate the same loading logic, file existence checks, and `loaded` flag.

**Create `Core/base_model.py`:**
```python
from pathlib import Path
import joblib
import logging

class BaseModel:
    model_files: list[str] = []  # override in subclass

    def __init__(self):
        self.loaded = False
        self._artifacts = {}

    def load(self) -> bool:
        for fname in self.model_files:
            path = Path(fname)
            if not path.exists():
                logging.warning(f"Model file not found: {fname}")
                self.loaded = False
                return False
            self._artifacts[fname] = joblib.load(path)
        self.loaded = True
        return True

    def _require_loaded(self):
        if not self.loaded:
            raise RuntimeError(f"{self.__class__.__name__} model not loaded. Call load() first.")
```

**Then in each model:**
```python
class AnomalyDetector(BaseModel):
    model_files = ["Models/anomaly_model.pkl", "Models/anomaly_scaler.pkl"]

    def predict(self, snapshot):
        self._require_loaded()
        # ... prediction logic only
```

---

### 2.3 Fix Classifier File Loading
**File:** `Core/classifier.py:188-192`  
**Problem:** `load()` throws `FileNotFoundError` if any artifact is missing, no graceful handling.

**Fix:** Handled by the `BaseModel` class above — file existence check before every `joblib.load`. If you're not adding a base class yet, at minimum:
```python
def load(self) -> bool:
    for path in [self.model_path, self.scaler_path, self.encoder_path]:
        if not Path(path).exists():
            logging.error(f"Missing model artifact: {path}")
            return False
    # load all three
    self.loaded = True
    return True
```

---

## Phase 3 — Medium Priority

### 3.1 Input Validation for External Data
**Files:** `Data/preprocess.py`, `pipeline/retrain.py`  
**Problem:** CSVs are loaded with no validation — malicious or corrupted data with bad types/extreme values goes straight into training.

**Add a validation step after loading:**
```python
NUMERIC_BOUNDS = {
    "cpu_percent": (0, 100),
    "ram_percent": (0, 100),
    "disk_percent": (0, 100),
}

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col in df.columns:
            invalid = df[(df[col] < lo) | (df[col] > hi)]
            if not invalid.empty:
                logging.warning(f"Clipping {len(invalid)} out-of-range rows in '{col}'")
            df[col] = df[col].clip(lo, hi)
    return df
```

Also log skipped lines in `retrain.py`:
```python
# Instead of silent skip:
df = pd.read_csv(path, on_bad_lines="warn")  # 'warn' logs instead of silencing
```

---

### 3.2 Break Up Long retrain.py Functions
**File:** `pipeline/retrain.py:121-245`  
**Problem:** `_build_*_dataset` functions are 30-80 lines each — hard to test and read.

**Refactor pattern:**
```python
def _build_anomaly_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = _filter_anomaly_columns(raw)
    df = _label_anomalies(df)
    df = _balance_classes(df)
    return df
```
Each helper is 5-15 lines, independently testable.

---

### 3.3 Move Schema Check Out of Flush Loop
**File:** `Data/collect.py:242-247`  
**Problem:** `_ensure_csv_schema` is called on every 10-row flush — wasteful repeated I/O.

**Fix:**
```python
class Collector:
    def __init__(self):
        self._schema_initialized = False

    def _flush_to_csv(self, rows):
        if not self._schema_initialized:
            self._ensure_csv_schema()
            self._schema_initialized = True
        # write rows
```

---

### 3.4 Clean Up Dashboard render() Signature
**File:** `Dashboard/app.py:256-264`  
**Problem:** 8-parameter function signature is hard to read and extend.

**Fix — use a dataclass:**
```python
from dataclasses import dataclass

@dataclass
class RenderContext:
    snapshot: SystemSnapshot
    anomaly_result: dict
    workload_result: dict
    prediction_result: dict
    history: list
    alerts: list
    cycle_time: float
    status: str

def render(ctx: RenderContext): ...
```

---

### 3.5 Rename Misleading "confidence" Key
**File:** `Core/feature_engineering.py:114-116`  
**Problem:** The `"confidence"` key in `statistical_anomaly()` is a derived score, not a real confidence value. Misleading for anyone reading the output.

**Fix:**
```python
return {
    "is_anomaly": is_anomaly,
    "anomaly_score": anomaly_score,
    "anomaly_confidence": anomaly_score if is_anomaly else 1.0 - anomaly_score,
    "z_scores": z_scores,
}
```

---

## Phase 4 — Documentation & CI

### 4.1 Add Docstrings
Priority functions missing docs:
- `feature_engineering.py` — `enrich()`, `statistical_anomaly()`, `_apply_cold_start_defaults()`
- `schema.py` — `normalize_columns()`, `ensure_columns()`
- `collect.py` — `take_snapshot()`, `_get_task_priority()`
- `config.py` — expand module docstring to list all config sections

Use Google style:
```python
def enrich(self, snapshot: SystemSnapshot) -> EnrichedSnapshot:
    """Adds rolling features and temporal deltas to a raw snapshot.

    Args:
        snapshot: Raw system snapshot from the collector.

    Returns:
        EnrichedSnapshot with delta and rolling stat fields populated.
    """
```

### 4.2 Add GitHub Actions CI
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ --cov=Core --cov-report=term-missing
```

---

## Summary Table

| Priority | Task | File(s) |
|----------|------|---------|
| 🔴 Critical | Replace pickle with checksum-verified loading (→ ONNX later) | All 3 model files |
| 🔴 Critical | Add unit tests for core functionality | New `tests/` dir |
| 🟠 High | Move hyperparameters to `config.py` | `anomaly.py`, `classifier.py`, `predictor.py` |
| 🟠 High | Add `BaseModel` class to eliminate duplicate load logic | New `Core/base_model.py` |
| 🟠 High | Fix classifier missing-file crash | `classifier.py` |
| 🟡 Medium | Input validation for CSV/external data | `preprocess.py`, `retrain.py` |
| 🟡 Medium | Break up long `_build_*_dataset` functions | `retrain.py` |
| 🟡 Medium | Move schema check out of flush loop | `collect.py` |
| 🟡 Medium | Refactor `render()` with dataclass | `Dashboard/app.py` |
| 🟡 Medium | Rename misleading `"confidence"` key | `feature_engineering.py` |
| 🟢 Low | Add docstrings to all public functions | Multiple files |
| 🟢 Low | GitHub Actions CI | `.github/workflows/ci.yml` |
