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
│   ├── classifier.py
│   ├── feature_engineering.py
│   ├── predictor.py
│   └── schema.py
├── Dashboard/
│   └── app.py
├── Data/
│   ├── collect.py
│   ├── preprocess.py
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
├── Models/
│   ├── anomaly_model.pkl
│   ├── classifier_model.pkl
│   ├── classifier_scaler.pkl
│   ├── classifier_target_encoder.pkl
│   ├── predictor_model.pkl
│   └── predictor_scaler.pkl
├── pipeline/
│   └── retrain.py
├── config.py
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

If `Models/anomaly_model.pkl` does not exist, runtime falls back to the statistical bridge in `FeatureEngineer`.

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
