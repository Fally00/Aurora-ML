# Aurora

Aurora is a CLI-only passive watcher and system telemetry tool in the SIT-Py repository. It records system measurements and file/Event Log observations in SQLite and prints new alerts to the console. It does not execute, modify, quarantine, or delete watched files, and it does not automatically remediate anything.

The repository also contains the earlier SIT-Py machine-learning prototype. Its `main.py` and retraining entry points are separate from Aurora's CLI.

## Project status

| Phase | Status |
|---|---|
| 1. Repository and ML pipeline cleanup | Implemented; the legacy ML tests are in `tests/`. |
| 2. Collector v2 and SQLite telemetry | Implemented with mocked tests. Real 10-minute sensor acceptance on target hardware is pending. |
| 3. Module B file and Event Log monitor | Implemented with rule, storage, and mocked Event Log tests. |
| 4. CLI and packaging | CLI commands and one-file PyInstaller build are implemented. The Windows executable built and launched from a console on 2026-11-16. |
| 5. Tests and evaluation tools | Latest run: 119 passed out of 119. The NVML-unavailable test is now hermetic (mocks the pynvml import) so it passes on hosts with a real GPU. Scenario results and live overhead measurements are still pending. |

The Aurora workload classifier and resource-abuse detector are interfaces only; no ML detector is enabled in Aurora.

## Requirements

- Windows 10 or 11, 64-bit, for filesystem notifications and the packaged executable.
- Python 3.12 64-bit is recommended for this repository's pinned NumPy and scikit-learn versions. Their Windows wheels include CPython 3.12 builds: [NumPy 1.26.4](https://pypi.org/project/numpy/1.26.4/) and [scikit-learn 1.5.1](https://pypi.org/project/scikit-learn/1.5.1/).
- Install the pinned packages from `requirements.txt`. This includes watchdog, psutil, PyInstaller, and the legacy ML packages.
- NVIDIA GPU measurements require an installed NVIDIA driver. Run `python -m Data.verify_nvml` to inspect NVML access. CPU package temperature and clocks are optional and require LibreHardwareMonitor running with its WMI interface enabled.
- Administrator rights are only needed for Windows Event Log clear subscriptions (Security event 1102 and System event 104). File watching and telemetry continue without elevation; one warning is logged when Event Log access is unavailable.

## Install

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the Python launcher command differs on your machine, create the virtual environment with your Python 3.12 executable and use `.venv\Scripts\python.exe` for the remaining commands.

## Run Aurora

Start telemetry collection, Module B monitoring, and the btop-style
terminal dashboard with:

```powershell
.\.venv\Scripts\python.exe aurora_cli.py
```

The default `run` command shows a live dashboard (CPU, GPU, memory, disk,
network, top processes, and an alerts feed) and refreshes once per second.
The collector and Module B watcher run in the background; Module B startup
failure does not stop telemetry.

Available commands:

```powershell
.\.venv\Scripts\python.exe aurora_cli.py            # dashboard (default)
.\.venv\Scripts\python.exe aurora_cli.py run --tag gaming
.\.venv\Scripts\python.exe aurora_cli.py run --plain   # headless, one alert per line
.\.venv\Scripts\python.exe aurora_cli.py run --refresh 2 # dashboard refresh interval (s)
.\.venv\Scripts\python.exe aurora_cli.py alerts --limit 20
.\.venv\Scripts\python.exe aurora_cli.py status
```

Dashboard keys:

| Key | Action |
|-----|--------|
| q | Quit and shut down cleanly |
| Tab | Cycle focus between panels |
| ↑ / ↓ | Scroll the focused list (alerts) |
| n | Mark the selected alert normal |
| d | Dismiss the selected alert |

Supported tags are `idle`, `gaming`, `compile`, `browsing`, and `stress`.
`--tag` selects and saves a workload tag in the runtime data directory.
`--plain` keeps the legacy headless alert stream (tests and background use
rely on it); `--quiet` suppresses collector sample output. `alerts` prints
recent records from SQLite as severity, timestamp, and title; `status`
prints a labeled system report and database counts, with `N/A` for
unavailable sensors. The CLI path does not load the legacy machine-learning
models.

For standalone collection or monitoring, run `python -m Data.collector_v2`
or `python -m Data.file_monitor`. Do not run these alongside the CLI `run`
command unless duplicate collection or watcher instances are intended.

To change the tag during a source checkout run, write one supported value to
`Data/workload_tag.txt`. In the packaged app, the runtime data directory is
`%LOCALAPPDATA%\Aurora\Data`.

## File monitoring and alerts

The watcher uses OS filesystem change notifications through watchdog; it does not poll directory contents. Default roots cover Desktop, Documents, Downloads, Music, Pictures, Videos, OneDrive, startup folders, and the temporary folder. The JSON config is [Data/watcher_config.json](Data/watcher_config.json).

By default, cache folders, temporary non-executables, build outputs, `.git` internals, and common game-library folders are excluded. Executables in temporary folders are still observed. Executables and configured critical files are SHA-256 hashed. File edits, executable changes, and startup-path changes are warnings; high-volume ordinary file creation is grouped into an informational event. Event Log clear detections are critical. Event Log subscriptions watch future clear events only; they do not scan historical events.

The shared SQLite `alerts` table stores alert ID, timestamp, module, severity, title, signals JSON, grouped count, dismissed state, and marked-normal feedback. Alerts are retained in SQLite and new records are printed by `run`. The packaged app embeds the default watcher config; edit `Data/watcher_config.json` before rebuilding to customize packaged defaults.

## Build the Windows executable

Build on Windows from the repository root with the virtual environment active or its executable on PATH:

```powershell
.\build_aurora.ps1
```

The one-file console executable is written to `dist\Aurora.exe`. The PyInstaller recipe is [Aurora.spec](Aurora.spec). Build and runtime data are ignored by Git. The packaged database and workload tag are stored under `%LOCALAPPDATA%\Aurora\Data`, outside PyInstaller's temporary extraction directory. The executable was built and its `status` command was verified from a console. Run elevated only if you need Security/System Event Log clear subscriptions. It does not request elevation or change system settings.

## Tests and evaluation

Run the unit suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests cover the CLI lifecycle, exclusions, severity mapping, burst grouping, cooldowns, alert storage/schema, feedback, file hashing, Event Log subscription behavior with mocked Windows APIs, and the evaluation scripts. On 2026-11-16, all 119 tests passed; the NVML-unavailable test is now hermetic so it passes on hosts with a real GPU.

The scenario runner writes to an isolated temporary watched tree and a fresh SQLite database under `Data/scenario_runs/<run-id>`. It edits a text file, creates a harmless `.exe` data stub that is never executed, copies 100 text files into a test Downloads folder, and simulates Security event 1102 through Module B's queue. It never clears a real Windows Event Log. The default run includes this simulated log-clear case, so explicitly enable it only on a test VM:

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

## Data and configuration

- Source checkout: SQLite telemetry and alerts are in `Data/db/aurora_metrics.db`. Daily telemetry tables are retained for seven days; alert records are kept separately in the same database.
- Packaged app: runtime files are in `%LOCALAPPDATA%\Aurora\Data`.
- Model artifacts for the legacy ML prototype are stored outside the checkout by default at `~/.sit-py/models`. Set `SITPY_MODELS_DIR` to use another directory. Model files require matching SHA-256 sidecars before they are loaded.
- `AURORA_MACHINE_ID` sets the Collector v2 machine identifier; the default is `laptop`.
- Runtime output under `Data/db`, `Data/scenario_runs`, CSV logs, and trained models is local and ignored by Git.

## Legacy SIT-Py prototype

These entry points are independent of Aurora's CLI:

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe pipeline/retrain.py --force
```

The prototype has anomaly, workload, and resource-prediction models. Its trained artifact loader verifies sidecar hashes. If artifacts are absent or invalid, the prototype may use its available fallback behavior; train/retrain artifacts before expecting trained predictions. This legacy pipeline uses its own CSV telemetry path and is not used by Aurora.

## Repository map

```text
Core/         Schemas, alerts, legacy models, and Aurora model interfaces
Data/         Collector v2, SQLite storage, Module B, watcher configuration
scripts/      Scenario runner, alert metrics, and overhead sampler
tests/        Unit tests (including mocked Windows API tests)
aurora_cli.py CLI entry point for run, alerts, status, and the dashboard
aurora_ui.py  Pure Rich render functions for the btop-style dashboard
Aurora.spec   PyInstaller one-file console build definition
build_aurora.ps1 Windows build script
```
