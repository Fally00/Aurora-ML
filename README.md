# Aurora

Aurora is a Windows passive watcher and system telemetry tray app in the SIT-Py repository. It records system measurements and file/event observations, then notifies the user. It does not execute, modify, quarantine, or delete watched files, and it does not automatically remediate anything.

The repository also contains the earlier SIT-Py machine-learning prototype. Its terminal and Streamlit entry points are separate from Aurora's tray app.

## Project status

| Phase | Status |
|---|---|
| 1. Repository and ML pipeline cleanup | Implemented; the legacy ML tests are in `tests/`. |
| 2. Collector v2 and SQLite telemetry | Implemented with mocked tests. Real 10-minute sensor acceptance on the target hardware is still pending. |
| 3. Module B file and Event Log monitor | Implemented with rule, storage, and mocked Event Log tests. The scenario runner has not been exercised here because the OS watcher dependency could not be installed. |
| 4. Tray UI and packaging | UI and one-file PyInstaller recipe are implemented. The `.exe` has not been built or launched in this environment. |
| 5. Tests and evaluation tools | Implemented. The suite passed **117 tests** on 2026-10-31. Scenario results and live overhead measurements still need to be collected on a Windows machine with the dependencies installed. |

The Aurora workload classifier and resource-abuse detector are interfaces only. The tray reports **not trained yet** for both; no ML detector is enabled in Aurora.

## Requirements

- Windows 10 or 11, 64-bit, for the tray app, filesystem notifications, and packaged executable.
- Python 3.12 64-bit is recommended for this repository's pinned NumPy and scikit-learn versions. Their Windows wheels include CPython 3.12 builds: [NumPy 1.26.4](https://pypi.org/project/numpy/1.26.4/) and [scikit-learn 1.5.1](https://pypi.org/project/scikit-learn/1.5.1/).
- Install the pinned packages from `requirements.txt`. This includes PySide6, pyqtgraph, watchdog, psutil, PyInstaller, and the legacy ML packages.
- NVIDIA GPU measurements require an installed NVIDIA driver. Run `python -m Data.verify_nvml` to inspect NVML access. CPU package temperature and clocks are optional and require LibreHardwareMonitor running with its WMI interface enabled.
- Administrator rights are only needed for Windows Event Log clear subscriptions (Security event 1102 and System event 104). File watching and telemetry continue without elevation; Aurora logs one warning if Event Log access is unavailable.

## Install

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the Python launcher command differs on your machine, create the virtual environment with your Python 3.12 executable and use `.venv\Scripts\python.exe` for the remaining commands.

## Run Aurora

### Tray app (recommended)

```powershell
.\.venv\Scripts\python.exe -m Dashboard.tray_app
```

The app opens Live stats and stays in the system tray. The hub and tray menu open four windows:

1. **Live stats**: CPU, GPU, RAM, available temperatures, Aurora UI CPU/RAM, and collector CPU/RAM.
2. **System info**: machine, OS, processor, memory, GPU/driver when `nvidia-smi` is available, and Aurora version.
3. **System state**: the collector's current workload tag and the workload-classifier status.
4. **Detections log**: all alert severities and their recorded signals. “This is normal” stores feedback in SQLite; dismissing a critical alert records it as dismissed.

Warnings use tray notifications. Critical alerts stay in a dialog until dismissed. Closing the hub hides Aurora to the tray; use the tray's **Quit** action to stop collection and monitoring.

The collector and Module B watcher run in background threads. SQLite reads keep the UI responsive. GPU or temperature fields show N/A when a sensor is unavailable.

### Standalone Module B watcher

Use this when you want file and Event Log monitoring without the tray UI:

```powershell
.\.venv\Scripts\python.exe -m Data.file_monitor
```

Do not run this alongside the tray app unless duplicate watcher instances are intended. The tray app already starts Module B.

### Standalone Collector v2

Use this when you want telemetry collection without the tray UI:

```powershell
.\.venv\Scripts\python.exe -m Data.collector_v2 --tag gaming --duration 600
```

Supported tags are `idle`, `gaming`, `compile`, `browsing`, and `stress`. `--duration` is optional; without it, the collector runs until stopped. `--quiet` suppresses per-sample console output. Do not run a second collector alongside the tray app unless duplicate samples are intended.

To change the tag during a source checkout run, write one supported value to `Data/workload_tag.txt`. In the packaged app, the runtime data directory is `%LOCALAPPDATA%\Aurora\Data`.

## File monitoring and alerts

The watcher uses OS filesystem change notifications through watchdog; it does not poll directory contents. Default roots cover Desktop, Documents, Downloads, Music, Pictures, Videos, OneDrive, startup folders, and the temporary folder. The JSON config is [Data/watcher_config.json](Data/watcher_config.json).

By default, cache folders, temporary non-executables, build outputs, `.git` internals, and common game-library folders are excluded. Executables in temporary folders are still observed. Executables and configured critical files are SHA-256 hashed. File edits, executable changes, and startup-path changes are warnings; high-volume ordinary file creation is grouped into an informational event. Event Log clear detections are critical. Event Log subscriptions watch future clear events only; they do not scan historical events.

The shared SQLite `alerts` table stores alert ID, timestamp, module, severity, title, signals JSON, grouped count, dismissed state, and “marked normal” feedback. Warnings and critical events can notify; info alerts are log-only.

The packaged app embeds the default watcher config. To customize defaults for a packaged build, edit `Data/watcher_config.json` before rebuilding.

## Build the Windows executable

Build on Windows from the repository root with the virtual environment active or its executable on PATH:

```powershell
.\build_aurora.ps1
```

The one-file app is written to `dist\Aurora.exe`. The PyInstaller recipe is [Aurora.spec](Aurora.spec). Build and runtime data are ignored by Git. The packaged database and workload tag are stored under `%LOCALAPPDATA%\Aurora\Data`, outside PyInstaller's temporary extraction directory.

Run Aurora normally for telemetry and file monitoring. Run it elevated only if you need Security/System Event Log clear subscriptions. The app does not request elevation or change system settings.

## Tests and evaluation

Run the unit suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests cover exclusions, severity mapping, burst grouping, cooldowns, alert storage/schema, feedback, file hashing, Event Log subscription behavior with mocked Windows APIs, and the evaluation scripts.

The scenario runner writes to an isolated temporary watched tree and a fresh SQLite database under `Data/scenario_runs/<run-id>`. It edits a text file, creates a harmless `.exe` data stub that is never executed, copies 100 text files into a test Downloads folder, and simulates Security event 1102 through Module B's queue. It **never clears a real Windows Event Log**. The default run includes this simulated log-clear case, so explicitly enable it only on a test VM:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_scenarios --i-am-on-a-test-vm
```

To run only non-log scenarios, select them explicitly and omit the test-VM flag:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_scenarios --scenario watched-edit --scenario startup-executable --scenario bulk-copy
```

Score a completed run with its run directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.alert_metrics Data/scenario_runs/<run-id>
```

The metrics output includes precision, recall, F1, and false alarms per hour. By default, false alarms per hour uses the scenario run's elapsed time; `--observation-hours` can provide a longer observation period.

Measure a running Aurora process against the budget of average CPU below 1% and peak RSS below 150 MB:

```powershell
.\.venv\Scripts\python.exe -m scripts.overhead --pid <Aurora-PID> --minutes 5 --output Data/scenario_runs/overhead.json
```

Frame-time impact is measured manually: compare median and 95th-percentile frame times for the same fixed game/workload interval with Aurora stopped and running.

## Data and configuration

- Source checkout: SQLite telemetry and alerts are in `Data/db/aurora_metrics.db`. Daily telemetry tables are retained for seven days; alert records are kept separately in the same database.
- Packaged app: runtime files are in `%LOCALAPPDATA%\Aurora\Data`.
- Model artifacts for the legacy ML prototype are stored outside the checkout by default at `~/.sit-py/models`. Set `SITPY_MODELS_DIR` to use another directory. Model files require matching SHA-256 sidecars before they are loaded.
- `AURORA_MACHINE_ID` sets the Collector v2 machine identifier; the default is `laptop`.
- Runtime output under `Data/db`, `Data/scenario_runs`, CSV logs, and trained models is local and ignored by Git.

## Legacy SIT-Py prototype

These entry points are independent of the Aurora tray app:

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m streamlit run Dashboard/app.py
.\.venv\Scripts\python.exe pipeline/retrain.py --force
```

The prototype has anomaly, workload, and resource-prediction models. Its trained artifact loader verifies sidecar hashes. If artifacts are absent or invalid, the prototype may use its available fallback behavior; train/retrain artifacts before expecting trained predictions. This legacy pipeline uses its own CSV telemetry path and is not the classifier used by Aurora's System state window.

## Repository map

```text
Core/        Schemas, alerts, legacy models, and Aurora model interfaces
Data/        Collector v2, SQLite storage, Module B, watcher configuration
Dashboard/   Legacy Streamlit app and Aurora tray app
scripts/     Scenario runner, alert metrics, and overhead sampler
tests/       Unit tests (including mocked Windows API tests)
Aurora.spec  PyInstaller one-file build definition
```
