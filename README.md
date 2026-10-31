# Aurora ML is sit branch made in python

SIT-Py is the Python prototype of the System Insight Toolkit. It watches live system telemetry, enriches it with rolling features, runs three ML agents on every cycle, and exposes the results in both a terminal loop and a Streamlit dashboard.

The current codebase is built around a shared runtime schema and separate datasets per agent:

- `anomaly` uses anomaly-focused labeled data
- `classifier` uses workload-priority labeled data
- `predictor` uses resource time-series style data

## Current Architecture

The runtime path is:

1. `Data/collect.py` captures a raw `SystemSnapshot`
2. `Core/feature_engineering.py` adds deltas, rolling means/std, and spike signals
3. `Core/anomaly.py` scores anomaly risk
4. `Core/classifier.py` predicts workload level
5. `Core/predictor.py` forecasts next-cycle CPU / RAM / disk usage
6. `main.py` and `Dashboard/app.py` log and display the results

The whole project now shares one normalization layer in `Core/schema.py`, so live telemetry, old logs, retrain inputs, and base datasets all map into the same column names before training or inference.

## Project Layout

```text
SIT-Py/
├── Core/
│   ├── anomaly.py
│   ├── alerts.py
│   ├── base_model.py
│   ├── classifier.py
│   ├── feature_engineering.py
│   ├── predictor.py
│   ├── schema.py
│   └── schema_v2.py
├── Dashboard/
│   └── app.py
├── Data/
│   ├── collect.py
│   ├── collector_v2.py
│   ├── eventlog_monitor.py
│   ├── file_monitor.py
│   ├── preprocess.py
│   ├── storage.py
│   ├── verify_nvml.py
│   ├── watcher_config.json
│   ├── live_metrics.csv
│   ├── sit_log.csv
│   ├── retrain_log.csv
│   └── datasets/
│       ├── anomaly_FINAL.csv
│       ├── classifier_FINAL.csv
│       ├── predictor_FINAL.csv
│       ├── anomaly_cybersecurity.csv
│       ├── anomaly_telesurgery.csv
│       ├── anomaly_threat_logs.csv
│       ├── classifier_distributed.csv
│       ├── predictor_distributed.csv
│       ├── dataset.csv
│       └── Laptop_Motherboard_Health_Monitoring_Dataset.csv
├── pipeline/
│   └── retrain.py
├── config.py
├── requirements.txt
├── main.py
└── README.md
```

## Core Modules

### `config.py`

Central source of truth for:

- dataset paths
- model and data directories
- polling cadence
- retrain thresholds
- anomaly bridge threshold
- workload label thresholds
- model artifact directory (`SITPY_MODELS_DIR`, default `~/.sit-py/models`)

The rest of the project should import from here instead of hardcoding paths or constants.

### `Core/schema.py`

This is the schema hub for the whole project.

It provides:

- the `SystemSnapshot` dataclass used at runtime
- canonical feature lists for each model
- legacy column normalization
- label normalization for anomaly and workload data
- dataframe helpers used during training and retraining

If a column name changes, this is the first place that should be updated.

### `Core/feature_engineering.py`

Adds temporal context to live telemetry:

- `cpu_delta`
- `ram_delta`
- `cpu_rolling_mean`
- `ram_rolling_mean`
- `cpu_rolling_std`
- `ram_rolling_std`
- `bw_rolling_mean`
- `bw_spike`

It also provides the statistical bridge used when no trained anomaly model is available yet.

### `Core/predictor.py`

Predicts next-cycle:

- `cpu_pct`
- `ram_pct`
- `disk_pct`

Current model:

- `RandomForestRegressor`

Training priority:

1. explicit dataset path
2. `Data/datasets/predictor_FINAL.csv`
3. `Data/live_metrics.csv`

The predictor expects live rolling features from `FeatureEngineer`, but its base training dataset is stored in raw resource columns and converted into sliding windows during training.

### `Core/classifier.py`

Predicts workload class:

- `Low`
- `Medium`
- `High`
- `Critical`

Current model:

- `RandomForestClassifier`

Training priority:

1. explicit dataset path
2. `Data/datasets/classifier_FINAL.csv`
3. `Data/live_metrics.csv`

At inference time it uses the simplified live feature set:

- `cpu_pct`
- `ram_pct`
- `bandwidth_kbps`
- `packet_rate_pps`

### `Core/anomaly.py`

Detects whether the current snapshot is `Normal` or `Alert`.

Current model:

- supervised `RandomForestClassifier`

Training priority:

1. explicit dataset path
2. `Data/datasets/anomaly_FINAL.csv`

Inference uses:

- numeric telemetry: CPU, RAM, bandwidth, packet rate
- categorical context: protocol, attack type, severity, threat type

If no verified anomaly model is available, runtime falls back to the statistical bridge in `FeatureEngineer`.

Trained models and their SHA-256 sidecars are stored outside the repository by default at `~/.sit-py/models`. Set `SITPY_MODELS_DIR` before training or starting the app to use another private model directory. The repository's legacy `Models/` files are no longer loaded; retrain the models to create verified local artifacts:

```bash
python Core/anomaly.py
python Core/classifier.py
python Core/predictor.py
```

Install the application dependencies with `pip install -r requirements.txt`. Run the test suite with `pip install pytest pytest-cov` followed by `pytest tests/ --cov=Core --cov=Data --cov=pipeline`.

## Runtime Entry Points

### `main.py`

Runs the terminal monitoring loop.

It:

- loads trained models if available
- falls back safely when a model is missing
- enriches each live snapshot
- prints a cycle summary
- appends normalized rows to `Data/sit_log.csv`

Run it with:

```bash
python main.py
```

### `Dashboard/app.py`

Streamlit dashboard for live monitoring.

Features include:

- live CPU / RAM / disk metrics
- anomaly status
- workload status
- next-cycle prediction
- telemetry charts
- recent event table
- one-click retraining trigger

Run it with:

```bash
python -m streamlit run Dashboard/app.py
```

### Module B passive file and event-log monitor

Install the pinned dependencies, then start the notification-based monitor in a separate terminal:

```bash
python -m Data.file_monitor
```

Edit `Data/watcher_config.json` to adjust watched roots, exclusions, critical files, burst grouping, and cooldowns. Windows Event Log subscriptions need an elevated terminal; without it the monitor logs one warning and continues watching files. Module B only records observations and alerts; it never changes or removes files.

### Collector v2 (GPU and SQLite telemetry)

Install dependencies with `pip install -r requirements.txt`. On Windows, CPU temperature and per-core clock readings also require LibreHardwareMonitor to be running with its WMI interface enabled. Start a tagged 10-minute collection with:

```bash
python -m Data.collector_v2 --tag gaming --duration 600
```

Change the active workload tag during collection by writing one supported tag (`idle`, `gaming`, `compile`, `browsing`, or `stress`) to `Data/workload_tag.txt`. SQLite data is stored in `Data/db/aurora_metrics.db` using daily tables and a seven-day retention policy. Run `python -m Data.verify_nvml` on an NVIDIA-equipped host to check driver and sensor access.

### Aurora tray app

Install `requirements.txt`, then launch the four-window tray app with `python -m Dashboard.tray_app`. The tray exposes Live stats, System info, System state, and Detections log. Collector and Module B work run outside the UI thread; warning alerts use tray toasts and critical alerts remain open until dismissed. The workload classifier and resource-abuse detector are interfaces only and show “not trained yet.”

Build the Windows one-file app with `./build_aurora.ps1` or `python -m PyInstaller --clean --noconfirm Aurora.spec`. The executable writes its SQLite data under `%LOCALAPPDATA%\Aurora\Data`; it does not write into the temporary PyInstaller extraction directory. Run elevated only when Windows Event Log clear subscriptions are needed; file watching and telemetry work without elevation.

## Data Flow

### Live Files

- `Data/live_metrics.csv`: raw collected telemetry
- `Data/sit_log.csv`: enriched runtime rows plus model outputs
- `Data/retrain_log.csv`: retraining history and metrics

When the runtime schema changes, the collector and main pipeline rotate old CSVs into `*_legacy_*.csv` backups so new rows do not corrupt old files.

### Base Datasets

The project now uses separate base datasets for each model:

- `Data/datasets/anomaly_FINAL.csv`
- `Data/datasets/classifier_FINAL.csv`
- `Data/datasets/predictor_FINAL.csv`

Supporting source/merged datasets currently in `Data/datasets/` include:

- `anomaly_cybersecurity.csv`
- `anomaly_telesurgery.csv`
- `anomaly_threat_logs.csv`
- `classifier_distributed.csv`
- `predictor_distributed.csv`

`Data/preprocess.py` exists to normalize and merge raw source CSVs into those final per-agent datasets.

## Retraining

Use:

```bash
python pipeline/retrain.py
```

Or force a run regardless of runtime row count:

```bash
python pipeline/retrain.py --force
```

The retrainer:

1. reads `live_metrics.csv` and `sit_log.csv`
2. normalizes them through `Core/schema.py`
3. builds a separate training dataset for each agent
4. blends runtime data with the base dataset when appropriate
5. retrains all models
6. writes new artifacts to `Models/`
7. logs the run to `Data/retrain_log.csv`

Important behavior:

- predictor retraining uses runtime telemetry only when there is enough sequential history
- classifier retraining can derive labels from `WORKLOAD_THRESHOLDS` when labels are missing
- anomaly retraining skips runtime blending if the runtime anomaly labels are single-class

## Trained Model Artifacts

Current saved artifacts:

- `Models/predictor_model.pkl`
- `Models/predictor_scaler.pkl`
- `Models/classifier_model.pkl`
- `Models/classifier_scaler.pkl`
- `Models/classifier_target_encoder.pkl`
- `Models/anomaly_model.pkl`

The loaders validate artifact compatibility against the current feature schema. If an old model was trained on a previous feature set, SIT-Py will ask for retraining instead of crashing on inference.

## Setup

There is no pinned `requirements.txt` in the repo at the moment, so install the project dependencies in your environment manually.

Typical packages used by this codebase:

```bash
pip install pandas numpy scikit-learn joblib psutil streamlit plotly
```

## Training Commands

Train each model directly:

```bash
python Core/predictor.py
python Core/classifier.py
python Core/anomaly.py
```

Because the code now defaults to the per-agent final datasets, these commands will train against:

- `predictor_FINAL.csv`
- `classifier_FINAL.csv`
- `anomaly_FINAL.csv`

unless you pass a custom CSV path.

## Notes

- The anomaly detector is no longer the old Isolation Forest path documented in earlier versions of the repo.
- The predictor is no longer the old linear-regression stack; it is a multi-output random forest with rolling features.
- The whole codebase now assumes the unified schema names like `cpu_pct`, `ram_pct`, `bandwidth_kbps`, and `packet_rate_pps`.
- Dataset files inside `Data/datasets/` are now ignored by Git except `Laptop_Motherboard_Health_Monitoring_Dataset.csv`.

## Next Improvements

Useful follow-up work from here:

- add a real `requirements.txt` or `pyproject.toml`
- improve predictor data quality with more sequential live telemetry
- tune anomaly thresholds on real machine data
- add tests for schema normalization and retrain dataset builders
