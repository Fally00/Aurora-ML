"""Live Textual dashboard and pure presentation/command helpers."""

from __future__ import annotations

import shlex
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Sequence

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import DataTable, Input, RichLog, Static

from Core.alerts import Alert
from Core.schema_v2 import ProcessInfo, SystemSnapshotV2
from Data.collector_v2 import set_workload_tag
from Data.storage import MetricsDB


_BAR_FULL = "█"
_BAR_EMPTY = "░"
_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_VALID_TAGS = {"idle", "gaming", "compile", "browsing", "stress"}
_VALID_FILTERS = {"info", "warn", "crit", "all"}
_SEVERITY_NAMES = {"warn": "warning", "crit": "critical"}
_SEVERITY_COLORS = {"critical": "bold red", "warning": "yellow", "info": "cyan"}


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    argument: Optional[str] = None


def parse_command(line: str) -> Optional[ParsedCommand]:
    """Parse a command line without performing I/O or changing application state."""
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        raise ValueError("Unclosed quote.") from exc
    if not parts:
        return None

    name = parts[0].lower()
    args = parts[1:]
    no_arg = {"status", "pause", "resume", "clear", "help", "quit"}
    if name in no_arg and not args:
        return ParsedCommand(name)
    if name == "tag" and len(args) == 1 and args[0].lower() in _VALID_TAGS:
        return ParsedCommand(name, args[0].lower())
    if name == "filter" and len(args) == 1 and args[0].lower() in _VALID_FILTERS:
        return ParsedCommand(name, args[0].lower())
    if name == "alerts" and len(args) <= 1:
        if not args:
            return ParsedCommand(name)
        if args[0].isdigit():
            return ParsedCommand(name, args[0])
    if name in {"normal", "dismiss"} and len(args) <= 1:
        return ParsedCommand(name, args[0] if args else None)
    if name in {"tag", "filter", "alerts", "normal", "dismiss"}:
        raise ValueError(f"Invalid arguments for {name}. Type help.")
    raise ValueError(f"Unknown command '{parts[0]}'. Type help.")


def ellipsize(value: object, width: int) -> str:
    """Return a single-line value clipped to a terminal-cell approximation."""
    text = " ".join(str(value).split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "…"
    return text[: width - 1] + "…"


def usage_bar(percent: Optional[float], width: int = 20) -> Text:
    """Render a correctly scaled bar; 0% is empty and 9% fills about 9%."""
    if width <= 0:
        return Text()
    if percent is None:
        return Text(_BAR_EMPTY * width, style="dim")
    bounded = max(0.0, min(100.0, float(percent)))
    filled = min(width, int(round(width * bounded / 100.0)))
    return Text(_BAR_FULL * filled + _BAR_EMPTY * (width - filled), style="green")


def sparkline(values: Sequence[float], width: int = 28) -> Text:
    """Render a compact history line, downsampled to the requested width."""
    if width <= 0:
        return Text()
    samples = [float(value) for value in values if value is not None]
    if not samples:
        return Text("·" * width, style="dim")
    if len(samples) == 1:
        samples *= width
    elif len(samples) > width:
        samples = [
            samples[round(i * (len(samples) - 1) / (width - 1))]
            for i in range(width)
        ] if width > 1 else [samples[-1]]
    low, high = min(samples), max(samples)
    span = high - low
    chars = []
    for value in samples:
        index = 0 if span == 0 else round((value - low) * 7 / span)
        chars.append(_SPARK_CHARS[max(0, min(7, index))])
    return Text("".join(chars), style="cyan")


def header_text(
    machine_id: str,
    tag: str,
    timestamp: str,
    uptime_seconds: int,
    module_b: str,
    collector_cpu: Optional[float],
    collector_rss_mb: Optional[float],
) -> Text:
    hours, remainder = divmod(max(0, uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    cpu = "N/A" if collector_cpu is None else f"{collector_cpu:.1f}%"
    rss = "N/A" if collector_rss_mb is None else f"{collector_rss_mb:.1f} MB"
    return Text(
        f"AURORA  {machine_id}  tag={tag}  {timestamp}  "
        f"up {hours}:{minutes:02d}:{seconds:02d}  {module_b}  "
        f"collector {cpu} / {rss}  Module A: not trained yet",
        style="bold cyan",
        no_wrap=True,
        overflow="ellipsis",
    )


def cpu_text(
    snapshot: Optional[SystemSnapshotV2],
    history: Sequence[SystemSnapshotV2],
    width: int = 36,
) -> Text:
    if snapshot is None:
        return Text("CPU  N/A\nNo telemetry samples yet.", style="dim")
    lines = [
        Text.assemble(
            ("CPU ", "bold"),
            (f"{snapshot.cpu_pct:.1f}%  ", "bold"),
            sparkline([sample.cpu_pct for sample in history], max(8, width - 12)),
        )
    ]
    for index, percent in enumerate(snapshot.cpu_per_core_pct):
        lines.append(
            Text.assemble(
                (f"{index:02d} ", "dim"),
                usage_bar(percent, 12),
                (f" {percent:.0f}%", "dim"),
            )
        )
    temperature = (
        f"CPU temp: {snapshot.cpu_package_temp_c:.1f}°C"
        if snapshot.cpu_package_temp_c is not None
        else "CPU temp: N/A (LibreHardwareMonitor not running)"
    )
    lines.append(Text(temperature, style="dim"))
    return _join_lines(lines)


def gpu_text(snapshot: Optional[SystemSnapshotV2]) -> Text:
    if snapshot is None or snapshot.gpu_util_pct is None:
        return Text("GPU  N/A\nGPU metrics unavailable.", style="dim")
    memory = (
        f"{snapshot.gpu_mem_used_mb:.0f}/{snapshot.gpu_mem_total_mb:.0f} MB"
        if snapshot.gpu_mem_used_mb is not None and snapshot.gpu_mem_total_mb
        else "N/A"
    )
    return Text(
        f"GPU {snapshot.gpu_util_pct:.1f}%  VRAM {memory}\n"
        f"Temp {format_optional(snapshot.gpu_temp_c, '°C')}  "
        f"Power {format_optional(snapshot.gpu_power_w, ' W')}",
        no_wrap=True,
        overflow="ellipsis",
    )


def memory_text(snapshot: Optional[SystemSnapshotV2]) -> Text:
    if snapshot is None:
        return Text("MEM  N/A", style="dim")
    return Text.assemble(
        (f"MEM {snapshot.ram_pct:.1f}%  ", "bold"),
        usage_bar(snapshot.ram_pct, 20),
        (f"\n{snapshot.ram_used_mb:.0f}/{snapshot.ram_total_mb:.0f} MB", "dim"),
    )


def disk_text(snapshot: Optional[SystemSnapshotV2]) -> Text:
    if snapshot is None:
        return Text("DISK  N/A", style="dim")
    return Text.assemble(
        (f"DISK {snapshot.disk_pct:.1f}%  ", "bold"),
        usage_bar(snapshot.disk_pct, 20),
        (f"\n{snapshot.disk_free_gb:.1f}/{snapshot.disk_total_gb:.1f} GB free", "dim"),
    )


def network_text(snapshot: Optional[SystemSnapshotV2], history: Sequence[SystemSnapshotV2]) -> Text:
    if snapshot is None:
        return Text("NET  N/A", style="dim")
    return Text.assemble(
        (f"NET {snapshot.bandwidth_kbps:.1f} kbps  ", "bold"),
        sparkline([sample.bandwidth_kbps for sample in history], 24),
        (f"\n{snapshot.packet_rate_pps:.1f} packets/s", "dim"),
    )


def process_rows(
    snapshot: Optional[SystemSnapshotV2],
) -> list[tuple[str, str, str, str, str]]:
    """Build bounded, non-wrapping rows combining top CPU and GPU processes."""
    if snapshot is None:
        return []
    cpu_by_pid = {proc.pid: proc for proc in snapshot.top_cpu_processes if proc.pid != 0}
    gpu_by_pid = {proc.pid: proc for proc in snapshot.top_gpu_processes if proc.pid != 0}
    ordered = list(dict.fromkeys([*cpu_by_pid, *gpu_by_pid]))
    rows = []
    for pid in ordered[:10]:
        cpu = cpu_by_pid.get(pid)
        gpu = gpu_by_pid.get(pid)
        process = cpu or gpu
        if process is None:
            continue
        cpu_percent = min(max(cpu.cpu_percent, 0.0), 100.0) if cpu else None
        gpu_mb = f"{gpu.gpu_memory_mb:.0f}" if gpu and gpu.gpu_memory_mb is not None else "—"
        rows.append(
            (
                str(pid),
                ellipsize(process.name, 24),
                "—" if cpu_percent is None else f"{cpu_percent:.1f}",
                gpu_mb,
                "CPU" if cpu else "GPU",
            )
        )
    return rows


def filtered_alerts(alerts: Sequence[Alert], severity_filter: str) -> list[Alert]:
    if severity_filter == "all":
        return list(alerts)
    severity = _SEVERITY_NAMES.get(severity_filter, severity_filter)
    return [alert for alert in alerts if alert.severity.lower() == severity]


def alert_counts(alerts: Sequence[Alert]) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        counts[alert.severity.lower()] = counts.get(alert.severity.lower(), 0) + 1
    return counts


def format_optional(value: Optional[float], suffix: str) -> str:
    return "N/A" if value is None else f"{value:.1f}{suffix}"


def _join_lines(lines: Sequence[Text]) -> Text:
    output = Text()
    for index, line in enumerate(lines):
        if index:
            output.append("\n")
        output.append(line)
    return output


class AuroraApp(App[None]):
    """Interactive, live dashboard. Storage reads run on the Textual thread."""

    TITLE = "Aurora"
    CSS = """
    Screen { layout: vertical; }
    #header { height: 3; border: round $accent; padding: 0 1; }
    #warning-strip { height: 2; border: round $warning; padding: 0 1; color: $warning; }
    #metrics-grid { height: 1fr; min-height: 10; grid-size: 3 2; grid-columns: 1fr 1fr 1fr; grid-rows: 1fr 1fr; }
    #tables-grid { height: 1fr; min-height: 10; grid-size: 2 1; grid-columns: 1fr 1fr; }
    .metric { border: round $primary; padding: 0 1; height: 1fr; overflow: hidden; }
    .table-panel { border: round $primary; height: 1fr; }
    .table-title { height: 1; padding: 0 1; color: $accent; text-style: bold; }
    DataTable { height: 1fr; }
    #console-title { height: 1; padding: 0 1; color: $accent; }
    #console { height: 7; border: round $secondary; }
    #command { height: 3; }
    """
    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=False),
        Binding("n", "mark_normal", "Normal", show=False),
        Binding("d", "dismiss_alert", "Dismiss", show=False),
        Binding("p", "toggle_pause", "Pause", show=False),
        Binding("escape", "cancel_command", "Cancel", show=False),
    ]

    def __init__(
        self,
        storage: MetricsDB,
        *,
        machine_id: str = "laptop",
        monitor: object = None,
        warning_provider: Optional[Callable[[], Sequence[str]]] = None,
        tag_setter: Callable[[str], bool] = set_workload_tag,
        collector_failed: Optional[threading.Event] = None,
    ) -> None:
        super().__init__()
        self.storage = storage
        self.machine_id = machine_id
        self.monitor = monitor
        self.warning_provider = warning_provider or (lambda: ())
        self.tag_setter = tag_setter
        self.collector_failed = collector_failed
        self.started_at = time.monotonic()
        self.paused = False
        self.severity_filter = "all"
        self.alert_limit = 500
        self.latest: Optional[SystemSnapshotV2] = None
        self.history: list[SystemSnapshotV2] = []
        self.alerts: list[Alert] = []
        self.visible_alerts: list[Alert] = []

    def compose(self) -> ComposeResult:
        yield Static("Aurora - waiting for telemetry", id="header")
        yield Static("No warnings", id="warning-strip")
        with Grid(id="metrics-grid"):
            yield Static("CPU  N/A", id="cpu", classes="metric")
            yield Static("GPU  N/A", id="gpu", classes="metric")
            yield Static("MEM  N/A", id="mem", classes="metric")
            yield Static("DISK  N/A", id="disk", classes="metric")
            yield Static("NET  N/A", id="net", classes="metric")
            yield Static("Processes", id="process-summary", classes="metric")
        with Grid(id="tables-grid"):
            with Vertical(classes="table-panel"):
                yield Static("PROCESSES — top CPU and GPU", classes="table-title")
                table = DataTable(id="processes", cursor_type="row", zebra_stripes=True)
                table.add_columns("PID", "Process", "CPU %", "GPU MB", "Source")
                yield table
            with Vertical(classes="table-panel"):
                yield Static("ALERTS", id="alerts-title", classes="table-title")
                table = DataTable(id="alerts", cursor_type="row", zebra_stripes=True)
                table.add_columns("Time", "Severity", "Alert")
                yield table
        yield Static("CONSOLE", id="console-title")
        yield RichLog(id="console", wrap=False, max_lines=500)
        yield Input(placeholder="Type : or / for commands; try help", id="command")

    def on_mount(self) -> None:
        self.query_one("#command", Input).blur()
        self.refresh_data()
        self.set_interval(1.0, self.refresh_data)

    def on_key(self, event: events.Key) -> None:
        if event.character in {":", "/"} and self.focused is not self.query_one("#command", Input):
            event.stop()
            self.query_one("#command", Input).focus()

    def refresh_data(self) -> None:
        if self.collector_failed is not None and self.collector_failed.is_set():
            self.exit()
            return
        if self.paused:
            return
        self.latest = self.storage.get_latest(self.machine_id)
        self.history = list(reversed(self.storage.query(machine_id=self.machine_id, limit=60)))
        self.alerts = self.storage.get_alerts(limit=self.alert_limit)
        self._render_widgets()

    def _render_widgets(self) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        module_b = "Module B: starting"
        if self.monitor is not None:
            worker = getattr(self.monitor, "_worker", None)
            if worker is not None and worker.is_alive():
                module_b = f"Module B: active ({getattr(self.monitor, 'watched_root_count', 0)} roots)"
            elif worker is not None:
                module_b = "Module B: stopped"
        self.query_one("#header", Static).update(
            header_text(
                self.machine_id,
                self.latest.workload_tag if self.latest else "idle",
                timestamp,
                int(time.monotonic() - self.started_at),
                module_b,
                self.latest.collector_cpu_pct if self.latest else None,
                self.latest.collector_rss_mb if self.latest else None,
            )
        )
        warnings = list(self.warning_provider())[-2:]
        self.query_one("#warning-strip", Static).update(
            "\n".join(warnings) if warnings else "No WARNING+ messages"
        )
        self.query_one("#cpu", Static).update(cpu_text(self.latest, self.history))
        self.query_one("#gpu", Static).update(gpu_text(self.latest))
        self.query_one("#mem", Static).update(memory_text(self.latest))
        self.query_one("#disk", Static).update(disk_text(self.latest))
        self.query_one("#net", Static).update(network_text(self.latest, self.history))

        process_table = self.query_one("#processes", DataTable)
        process_table.clear()
        process_rows_data = process_rows(self.latest)
        self.query_one("#process-summary", Static).update(
            f"Processes: {len(process_rows_data)}"
        )
        for row in process_rows_data:
            process_table.add_row(*(self._cell(value) for value in row))

        counts = alert_counts(self.alerts)
        self.query_one("#alerts-title", Static).update(
            f"ALERTS  {counts['critical']} crit  {counts['warning']} warn  {counts['info']} info"
        )
        self.visible_alerts = filtered_alerts(self.alerts, self.severity_filter)
        alert_table = self.query_one("#alerts", DataTable)
        alert_table.clear()
        for alert in self.visible_alerts:
            severity = alert.severity.lower()
            style = _SEVERITY_COLORS.get(severity, "white")
            alert_table.add_row(
                self._cell(alert.timestamp[-8:], "dim"),
                self._cell(severity, style),
                self._cell(ellipsize(alert.title, 48)),
                key=alert.id,
            )

    @staticmethod
    def _cell(value: str, style: str = "") -> Text:
        return Text(value, style=style, no_wrap=True, overflow="ellipsis")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""
        event.input.blur()
        try:
            command = parse_command(line)
        except ValueError as exc:
            self.write_console(str(exc))
            return
        if command is not None:
            self.execute_command(command)

    def execute_command(self, command: ParsedCommand) -> None:
        name, argument = command.name, command.argument
        if name == "help":
            self.write_console(
                "status | alerts [n] | tag <tag> | filter <info|warn|crit|all> | "
                "normal [id] | dismiss [id] | pause | resume | clear | help | quit"
            )
        elif name == "status":
            if self.latest is None:
                self.write_console("No telemetry samples yet.")
            else:
                self.write_console(
                    f"{self.latest.timestamp} CPU {self.latest.cpu_pct:.1f}% "
                    f"MEM {self.latest.ram_pct:.1f}% DISK {self.latest.disk_pct:.1f}%"
                )
        elif name == "alerts":
            self.alert_limit = max(1, int(argument)) if argument else 500
            self.refresh_data()
            self.write_console(f"Showing up to {self.alert_limit} alerts.")
        elif name == "tag" and argument:
            if self.tag_setter(argument):
                self.write_console(f"Workload tag set to {argument}.")
                self.refresh_data()
            else:
                self.write_console(f"Could not set workload tag: {argument}.")
        elif name == "filter" and argument:
            self.severity_filter = argument
            self._render_widgets()
            self.write_console(f"Alert filter: {argument}.")
        elif name == "pause":
            self.paused = True
            self.write_console("Dashboard refresh paused.")
        elif name == "resume":
            self.paused = False
            self.refresh_data()
            self.write_console("Dashboard refresh resumed.")
        elif name == "clear":
            self.query_one("#console", RichLog).clear()
        elif name in {"normal", "dismiss"}:
            self._update_alert(name, argument)
        elif name == "quit":
            self.exit()

    def _selected_alert_id(self, requested: Optional[str]) -> Optional[str]:
        if requested:
            return next(
                (alert.id for alert in self.alerts if alert.id == requested),
                None,
            )
        table = self.query_one("#alerts", DataTable)
        if 0 <= table.cursor_row < len(self.visible_alerts):
            return self.visible_alerts[table.cursor_row].id
        return None

    def _update_alert(self, action: str, requested: Optional[str]) -> None:
        alert_id = self._selected_alert_id(requested)
        if alert_id is None:
            self.write_console("Select an alert or provide its exact id.")
            return
        if action == "normal":
            changed = self.storage.mark_alert_normal(alert_id)
            result = "marked normal"
        else:
            changed = self.storage.dismiss_alert(alert_id)
            result = "dismissed"
        self.write_console(f"Alert {alert_id} {result}." if changed else "Alert not found.")
        self.refresh_data()

    def write_console(self, message: str) -> None:
        self.query_one("#console", RichLog).write(
            Text(ellipsize(message, 180), no_wrap=True, overflow="ellipsis")
        )

    def action_focus_command(self) -> None:
        self.query_one("#command", Input).focus()

    def action_cancel_command(self) -> None:
        command = self.query_one("#command", Input)
        if self.focused is command:
            command.value = ""
            command.blur()

    def action_quit_app(self) -> None:
        if self.focused is not self.query_one("#command", Input):
            self.exit()

    def action_mark_normal(self) -> None:
        if self.focused is not self.query_one("#command", Input):
            self._update_alert("normal", None)

    def action_dismiss_alert(self) -> None:
        if self.focused is not self.query_one("#command", Input):
            self._update_alert("dismiss", None)

    def action_toggle_pause(self) -> None:
        if self.focused is self.query_one("#command", Input):
            return
        self.paused = not self.paused
        self.write_console("Dashboard refresh paused." if self.paused else "Dashboard refresh resumed.")
        if not self.paused:
            self.refresh_data()


__all__ = [
    "AuroraApp",
    "ParsedCommand",
    "alert_counts",
    "cpu_text",
    "ellipsize",
    "filtered_alerts",
    "parse_command",
    "process_rows",
    "usage_bar",
]
