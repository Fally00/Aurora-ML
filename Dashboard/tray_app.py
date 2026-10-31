"""Aurora Windows tray app. Monitoring and collection run on background threads."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

import psutil
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QStyle,
)

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - displayed as an install hint at runtime
    pg = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_PATH, MACHINE_ID  # noqa: E402
from Core.workload import (  # noqa: E402
    UntrainedResourceAbuseDetector,
    UntrainedWorkloadClassifier,
)
from Data.collector_v2 import run_collector  # noqa: E402
from Data.file_monitor import FileMonitor  # noqa: E402
from Data.storage import MetricsDB, get_db  # noqa: E402

APP_VERSION = "0.4.0"
LOGGER = logging.getLogger("aurora.tray")


class BackgroundServices:
    """Own the collector thread and Module B watcher lifecycle."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.collector_thread: threading.Thread | None = None
        self.monitor_thread: threading.Thread | None = None
        self.file_monitor = FileMonitor()

    def start(self) -> None:
        self.stop_event.clear()
        self.collector_thread = threading.Thread(
            target=self._run_collector,
            name="aurora-collector",
            daemon=True,
        )
        self.collector_thread.start()
        self.monitor_thread = threading.Thread(
            target=self._start_file_monitor,
            name="aurora-monitor-bootstrap",
            daemon=True,
        )
        self.monitor_thread.start()

    def _start_file_monitor(self) -> None:
        try:
            if not self.stop_event.is_set():
                self.file_monitor.start()
        except Exception:
            LOGGER.exception("Module B could not start; telemetry collection will continue")

    def _run_collector(self) -> None:
        try:
            run_collector(verbose=False, stop_event=self.stop_event)
        except Exception:
            LOGGER.exception("Collector stopped after an error")

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.file_monitor.stop()
        except Exception:
            LOGGER.exception("Error while stopping Module B")
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
        if self.collector_thread:
            self.collector_thread.join(timeout=3.0)


class HardwareInfoWorker(QThread):
    result_ready = Signal(dict)

    def run(self) -> None:
        info: dict[str, str] = {
            "machine": platform.node() or "Unknown",
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "cpu": platform.processor() or "Unknown",
            "logical_cores": str(psutil.cpu_count(logical=True) or "Unknown"),
            "physical_cores": str(psutil.cpu_count(logical=False) or "Unknown"),
            "memory": f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB",
            "gpu": "Unavailable",
            "gpu_driver": "Unavailable",
        }
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode == 0 and completed.stdout.strip():
                gpu, _, driver = completed.stdout.strip().splitlines()[0].partition(",")
                info["gpu"] = gpu.strip()
                info["gpu_driver"] = driver.strip() or "Unknown"
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        self.result_ready.emit(info)


class CriticalToast(QDialog):
    """Persistent critical notification. It can only close through a button."""

    def __init__(self, alert: Any, storage: MetricsDB, on_closed) -> None:
        super().__init__()
        self.alert = alert
        self.storage = storage
        self.on_closed = on_closed
        self.setWindowTitle("Aurora critical alert")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        layout = QVBoxLayout(self)
        title = QLabel(f"<b>{alert.title}</b>")
        title.setWordWrap(True)
        layout.addWidget(title)
        detail = QLabel(json.dumps(alert.signals, indent=2, ensure_ascii=False))
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.clicked.connect(self._dismiss)
        layout.addWidget(self.dismiss_button)
        self.setMinimumWidth(400)

    def _dismiss(self) -> None:
        self.storage.dismiss_alert(self.alert.id)
        self.accept()
        self.on_closed(self.alert.id)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class PanelWindow(QMainWindow):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Aurora — {title}")
        self.resize(880, 600)
        self.setCentralWidget(QWidget())
        self.content = QVBoxLayout(self.centralWidget())

    def add_heading(self, title: str, subtitle: str | None = None) -> None:
        heading = QLabel(f"<h2>{title}</h2>")
        self.content.addWidget(heading)
        if subtitle:
            label = QLabel(subtitle)
            label.setWordWrap(True)
            self.content.addWidget(label)


class LiveStatsWindow(PanelWindow):
    def __init__(self, storage: MetricsDB, process: psutil.Process, parent=None) -> None:
        super().__init__("Live stats", parent)
        self.storage = storage
        self.process = process
        self.times: deque[str] = deque(maxlen=60)
        self.add_heading("Live stats", "One-second telemetry from Collector v2. Missing sensors are shown as N/A.")
        self.summary = QLabel("Waiting for the first SQLite sample…")
        self.content.addWidget(self.summary)
        self.overhead = QLabel("Aurora UI overhead: collecting…")
        self.content.addWidget(self.overhead)
        self.chart_notice = QLabel("")
        self.content.addWidget(self.chart_notice)
        self.usage_plot = None
        self.temp_plot = None
        self.curves: dict[str, Any] = {}
        if pg is not None:
            self.usage_plot = pg.PlotWidget(title="CPU / GPU / RAM (%)")
            self.usage_plot.setYRange(0, 100)
            self.usage_plot.addLegend()
            self.temp_plot = pg.PlotWidget(title="Package temperature (°C)")
            self.curves = {
                "CPU": self.usage_plot.plot(pen="#5bbcff", name="CPU"),
                "GPU": self.usage_plot.plot(pen="#c58cff", name="GPU"),
                "RAM": self.usage_plot.plot(pen="#57d6a1", name="RAM"),
                "CPU temp": self.temp_plot.plot(pen="#ff986b", name="CPU temp"),
                "GPU temp": self.temp_plot.plot(pen="#ffd166", name="GPU temp"),
            }
            self.content.addWidget(self.usage_plot, 1)
            self.content.addWidget(self.temp_plot, 1)
        else:
            self.chart_notice.setText("Charts unavailable. Install pyqtgraph from requirements.txt.")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def refresh(self) -> None:
        try:
            rows = list(reversed(self.storage.query(machine_id=MACHINE_ID, limit=60)))
            if not rows:
                return
            latest = rows[-1]
            self.summary.setText(
                f"CPU {latest.cpu_pct:.1f}%  |  GPU {_fmt(latest.gpu_util_pct, '%')}  |  "
                f"RAM {latest.ram_pct:.1f}%  |  CPU temp {_fmt(latest.cpu_package_temp_c, '°C')}  |  "
                f"GPU temp {_fmt(latest.gpu_temp_c, '°C')}  |  Workload tag: {latest.workload_tag}"
            )
            self.times = deque((str(i) for i in range(len(rows))), maxlen=60)
            if self.curves:
                x = list(range(len(rows)))
                values = {
                    "CPU": [r.cpu_pct for r in rows],
                    "GPU": [_nan(r.gpu_util_pct) for r in rows],
                    "RAM": [r.ram_pct for r in rows],
                    "CPU temp": [_nan(r.cpu_package_temp_c) for r in rows],
                    "GPU temp": [_nan(r.gpu_temp_c) for r in rows],
                }
                for name, curve in self.curves.items():
                    curve.setData(x, values[name])
            rss_mb = self.process.memory_info().rss / (1024 * 1024)
            ui_cpu = self.process.cpu_percent(None) / max(psutil.cpu_count(logical=True) or 1, 1)
            self.overhead.setText(
                f"Aurora UI CPU {ui_cpu:.2f}%  |  Aurora UI RAM {rss_mb:.1f} MB  |  "
                f"Collector CPU {latest.collector_cpu_pct:.2f}%  |  Collector RAM {latest.collector_rss_mb:.1f} MB"
            )
        except Exception:
            LOGGER.exception("Unable to refresh live stats")


class SystemInfoWindow(PanelWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("System info", parent)
        self.add_heading("System info", "Hardware details are queried in a worker thread.")
        self.details = QLabel("Loading system information…")
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content.addWidget(self.details)
        self.content.addWidget(QLabel(f"Aurora version: {APP_VERSION}"))
        self.content.addWidget(QLabel("Resource abuse model: not trained yet"))
        self.content.addWidget(QLabel("Workload classifier: not trained yet"))
        self.content.addStretch(1)
        self.worker = HardwareInfoWorker(self)
        self.worker.result_ready.connect(self._show_info)
        self.worker.start()

    def _show_info(self, info: dict) -> None:
        self.details.setText("\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in info.items()))


class SystemStateWindow(PanelWindow):
    def __init__(self, storage: MetricsDB, classifier=None, detector=None, parent=None) -> None:
        super().__init__("System state", parent)
        self.storage = storage
        self.classifier = classifier or UntrainedWorkloadClassifier()
        self.detector = detector or UntrainedResourceAbuseDetector()
        self.add_heading("System state", "Rule based status now; model interfaces are placeholders until trained.")
        self.workload = QLabel("Workload: waiting for telemetry")
        self.content.addWidget(self.workload)
        self.model = QLabel(f"Classifier: {self.classifier.predict(None).status}")
        self.content.addWidget(self.model)
        self.resource_model = QLabel(f"Resource abuse detector: {self.detector.status}")
        self.content.addWidget(self.resource_model)
        self.content.addStretch(1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1500)
        self.refresh()

    def refresh(self) -> None:
        latest = self.storage.get_latest(MACHINE_ID)
        if latest:
            state = self.classifier.predict(latest)
            self.workload.setText(f"Current workload tag: {latest.workload_tag}  |  Classifier: {state.status}")


class DetectionsWindow(PanelWindow):
    def __init__(self, storage: MetricsDB, parent=None) -> None:
        super().__init__("Detections log", parent)
        self.storage = storage
        self.add_heading("Detections log", "Alerts and their recorded signals. Mark expected events as normal to save feedback.")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Severity", "Time", "Title", "Count", "Feedback"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self.show_details)
        self.content.addWidget(self.table, 2)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select an alert to inspect its signals.")
        self.content.addWidget(self.details, 1)
        buttons = QHBoxLayout()
        self.normal_button = QPushButton("This is normal")
        self.normal_button.clicked.connect(self.mark_normal)
        self.dismiss_button = QPushButton("Dismiss alert")
        self.dismiss_button.clicked.connect(self.dismiss_selected)
        buttons.addWidget(self.normal_button)
        buttons.addWidget(self.dismiss_button)
        self.content.addLayout(buttons)
        self.alerts: list[Any] = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()

    def refresh(self) -> None:
        selected = self.selected_alert_id()
        self.alerts = self.storage.get_alerts(limit=500)
        self.table.setRowCount(len(self.alerts))
        for row, alert in enumerate(self.alerts):
            values = [alert.severity.upper(), alert.timestamp, alert.title,
                      str(alert.grouped_count), "Normal" if alert.marked_normal else ""]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, alert.id)
                self.table.setItem(row, col, cell)
        if selected:
            for row, alert in enumerate(self.alerts):
                if alert.id == selected:
                    self.table.selectRow(row)
                    break

    def selected_alert_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selected(self):
        alert_id = self.selected_alert_id()
        return next((alert for alert in self.alerts if alert.id == alert_id), None)

    def show_details(self) -> None:
        alert = self._selected()
        self.details.setPlainText(
            json.dumps(alert.to_dict(), indent=2, ensure_ascii=False) if alert else ""
        )

    def mark_normal(self) -> None:
        alert = self._selected()
        if alert and self.storage.mark_alert_normal(alert.id):
            self.refresh()

    def dismiss_selected(self) -> None:
        alert = self._selected()
        if alert and self.storage.dismiss_alert(alert.id):
            self.refresh()


class HubWindow(QMainWindow):
    def __init__(self, storage: MetricsDB, tray: QSystemTrayIcon, process: psutil.Process) -> None:
        super().__init__()
        self.tray = tray
        self.setWindowTitle("Aurora")
        self.resize(420, 270)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel("<h2>Aurora</h2><p>Passive system watcher</p>"))
        layout.addWidget(QLabel("Aurora observes and notifies. It never changes system files."))
        self.windows = {
            "Live stats": LiveStatsWindow(storage, process),
            "System info": SystemInfoWindow(),
            "System state": SystemStateWindow(storage),
            "Detections log": DetectionsWindow(storage),
        }
        for title, window in self.windows.items():
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, w=window: self.open_panel(w))
            layout.addWidget(button)
        admin_note = QLabel("Windows event log monitoring requires administrator rights; file watching remains available without elevation.")
        admin_note.setWordWrap(True)
        layout.addWidget(admin_note)
        self.setCentralWidget(central)
        self._quitting = False

    def open_panel(self, window: QWidget) -> None:
        window.show()
        window.raise_()
        window.activateWindow()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._quitting:
            event.accept()
        else:
            event.ignore()
            self.hide()
            self.tray.showMessage("Aurora", "Aurora is still running in the system tray.", QSystemTrayIcon.MessageIcon.Information, 2500)


class AuroraTrayApp:
    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.storage = get_db()
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        self.services = BackgroundServices()
        self.services.start()
        self.tray = QSystemTrayIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), app)
        self.tray.setToolTip("Aurora passive watcher")
        self.hub = HubWindow(self.storage, self.tray, self.process)
        menu = QMenu()
        open_hub = QAction("Open Aurora", menu)
        open_hub.triggered.connect(self.hub.show)
        menu.addAction(open_hub)
        menu.addSeparator()
        for title, window in self.hub.windows.items():
            action = QAction(title, menu)
            action.triggered.connect(lambda _checked=False, w=window: self.hub.open_panel(w))
            menu.addAction(action)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self.quit)
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._activated)
        self.tray.show()
        self.seen_alert_ids = {alert.id for alert in self.storage.get_alerts(limit=500)}
        self.critical_dialog: CriticalToast | None = None
        self.critical_ids_seen: set[str] = set()
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_alerts)
        self.timer.start(1000)
        self.refresh_alerts(startup=True)
        self.hub.show()
        self.hub.open_panel(self.hub.windows["Live stats"])

    def _activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.hub.showNormal()
            self.hub.raise_()
            self.hub.activateWindow()

    def refresh_alerts(self, startup: bool = False) -> None:
        alerts = self.storage.get_alerts(limit=100, include_dismissed=False)
        for alert in alerts:
            if alert.severity == "critical" and not alert.dismissed:
                self._show_critical(alert)
            if not startup and alert.id not in self.seen_alert_ids and alert.severity == "warning":
                self.tray.showMessage("Aurora warning", alert.title, QSystemTrayIcon.MessageIcon.Warning, 8000)
            self.seen_alert_ids.add(alert.id)

    def _show_critical(self, alert) -> None:
        if self.critical_dialog is not None or alert.id in self.critical_ids_seen:
            return
        self.critical_ids_seen.add(alert.id)
        self.critical_dialog = CriticalToast(alert, self.storage, self._critical_closed)
        self.critical_dialog.show()

    def _critical_closed(self, alert_id: str) -> None:
        self.critical_dialog = None
        QTimer.singleShot(250, self.refresh_alerts)

    def quit(self) -> None:
        self.timer.stop()
        self.services.stop()
        for window in self.hub.windows.values():
            window.close()
        self.hub._quitting = True
        self.hub.close()
        self.tray.hide()
        self.app.quit()


def _fmt(value: float | None, suffix: str) -> str:
    return f"{value:.1f}{suffix}" if value is not None else "N/A"


def _nan(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("Aurora")
    app.setQuitOnLastWindowClosed(False)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.warning(None, "Aurora", "No system tray is available. Aurora will run with its app window only.")
    controller = AuroraTrayApp(app)
    app.aboutToQuit.connect(controller.services.stop)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
