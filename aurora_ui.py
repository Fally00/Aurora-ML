"""Aurora btop-style dashboard — pure render functions.

Every function here is terminal-free: it takes snapshots / alerts /
layout hints and returns Rich renderables. The Live loop, keyboard
handling and SQLite reads live in aurora_cli.py, which keeps this module
unit-testable without a real terminal.
"""

from __future__ import annotations

import sys as _sys

from typing import Optional, Sequence

from rich.align import Align
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from Core.alerts import Alert
from Core.schema_v2 import ProcessInfo, SystemSnapshotV2

import locale as _locale

# ── unicode support ─────────────────────────────────────────────────────
# Legacy conhost / piped output (cp1252) cannot render block graphics.
# Detect once and fall back to ASCII so CI smoke tests and plain consoles
# still render something useful; modern Windows Terminal gets the real blocks.
# We inspect the encoding *name* only (never encode the block characters at
# import time) so the module imports cleanly under any code page.


def _unicode_supported() -> bool:
    try:
        enc = (
            _sys.stdout.encoding
            or _locale.getpreferredencoding(False)
            or ""
        ).lower()
        return "utf" in enc
    except Exception:
        return False


USE_UNICODE = _unicode_supported()
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█" if USE_UNICODE else " .:-=+*#"
FULL_BLOCK = "▮" if USE_UNICODE else "#"
EMPTY_BLOCK = "▯" if USE_UNICODE else "."
DEG = "°C" if USE_UNICODE else "C"
SEP = " · " if USE_UNICODE else " | "
ARROWS = "↑↓" if USE_UNICODE else "^/v"

# ── thresholds (green → yellow → red) ──────────────────────────────────
CPU_WARN, CPU_CRIT = 50.0, 75.0
RAM_WARN, RAM_CRIT = 50.0, 75.0
DISK_WARN, DISK_CRIT = 70.0, 85.0
GPU_WARN, GPU_CRIT = 70.0, 85.0

# Severity → (style, label prefix)
SEVERITY_STYLE = {
    "critical": ("red bold", "! "),
    "warning": ("yellow", ">> "),
    "info": ("dim", "- "),
}


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _color_for(pct: Optional[float], warn: float, crit: float) -> str:
    """Return a Rich color name for a percentage reading."""
    if pct is None:
        return "gray50"
    if pct >= crit:
        return "red"
    if pct >= warn:
        return "yellow"
    return "green"


def sparkline(
    values: Sequence[float],
    width: int = 40,
    warn: float = 50.0,
    crit: float = 75.0,
    *,
    current_color_override: Optional[str] = None,
) -> Text:
    """Render a unicode-block sparkline, coloring blocks by severity level."""
    text = Text()
    pts = [float(v) for v in values if v is not None]
    if not pts or width <= 0:
        text.append(EMPTY_BLOCK * max(width, 0), style="gray50")
        return text

    # Downsample to exactly `width` points by even spacing across the series.
    n = len(pts)
    if n == 1:
        samples = list(pts) * width
    elif width == 1:
        samples = [pts[-1]]
    else:
        samples = [
            pts[min(n - 1, int(round(i * (n - 1) / (width - 1))))]
            for i in range(width)
        ]

    vmin = min(samples)
    vmax = max(samples)
    span = vmax - vmin
    last = samples[-1]
    for i, v in enumerate(samples):
        if span == 0:
            ch = SPARKLINE_CHARS[-1]
        else:
            idx = int(round(((v - vmin) / span) * (len(SPARKLINE_CHARS) - 1)))
            idx = max(0, min(len(SPARKLINE_CHARS) - 1, idx))
            ch = SPARKLINE_CHARS[idx]
        style = current_color_override or _color_for(v, warn, crit)
        # Emphasize the most recent sample.
        if i == len(samples) - 1:
            style = f"bold {style}"
        text.append(ch, style=style)
    return text


def bar(
    pct: Optional[float],
    width: int = 20,
    warn: float = 50.0,
    crit: float = 75.0,
    *,
    show_label: bool = True,
) -> Text:
    """A ▮/▯ usage bar colored by threshold, optionally with a % label."""
    text = Text()
    if width <= 0:
        return text
    if pct is None:
        text.append(EMPTY_BLOCK * width, style="gray50")
        if show_label:
            text.append(" N/A")
        return text
    filled = int(round((_clip(pct) / 100.0) * width))
    filled = max(0, min(width, filled))
    color = _color_for(pct, warn, crit)
    text.append(FULL_BLOCK * filled, style=color)
    text.append(EMPTY_BLOCK * (width - filled), style="gray50")
    if show_label:
        text.append(f" {pct:5.1f}%", style=color)
    return text


def _fmt(value: Optional[float], suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _join(values: Sequence[str], sep: str = "  ") -> str:
    return sep.join(v for v in values if v)


# ── Header ──────────────────────────────────────────────────────────────

def header_panel(
    machine_id: str,
    workload_tag: str,
    timestamp: str,
    uptime: str,
    module_b_roots: int,
    event_log_on: bool,
    collector_cpu: Optional[float],
    collector_rss: Optional[float],
    module_a_status: str = "not trained yet",
    width: int = 80,
) -> Panel:
    """Top status bar: machine, tag, clock, uptime, Module B, overhead."""
    left = Text()
    left.append("Aurora  ", style="bold cyan")
    left.append(f"{machine_id}  ", style="bold")
    left.append(f"tag={workload_tag}  ", style="magenta")
    left.append(f"{timestamp}  ", style="green")
    left.append(f"up={uptime}  ", style="green")

    right = Text()
    el = "on" if event_log_on else "off"
    right.append(f"Module B: {module_b_roots} roots{SEP}eventlog={el}  ", style="cyan")
    if collector_cpu is not None:
        right.append(f"collector {collector_cpu:.1f}% / {collector_rss:.1f}MB  ", style="cyan")
    right.append(f"Module A: {module_a_status}", style="gray50")

    # Combine left + right on one line, right-aligned.
    line = Text()
    line.append(left)
    pad = max(0, width - left.cell_len - right.cell_len - 2)
    line.append(" " * pad)
    line.append(right)

    return Panel(line, title="Aurora Dashboard", border_style="cyan", width=width, box=box.ASCII)


# ── CPU panel ───────────────────────────────────────────────────────────

def cpu_panel(snap: SystemSnapshotV2, history: Sequence[SystemSnapshotV2], width: int) -> Panel:
    """CPU utilization: big %, sparkline, per-core bars, pkg temp, clocks."""
    hist = list(history)[-60:]
    cpu_vals = [s.cpu_pct for s in hist]

    lines: list[Text] = []

    # Big number + sparkline
    head = Text()
    head.append(f"CPU  {snap.cpu_pct:5.1f}%  ", style="bold " + _color_for(snap.cpu_pct, CPU_WARN, CPU_CRIT))
    head.append(sparkline(cpu_vals, width=max(20, width - 20), warn=CPU_WARN, crit=CPU_CRIT))
    lines.append(head)

    # Per-core bars (compact, two rows of 8)
    cores = snap.cpu_per_core_pct or []
    if cores:
        for row_start in (0, 8):
            row = cores[row_start : row_start + 8]
            if not row:
                continue
            core_line = Text()
            for i, c in enumerate(row):
                core_line.append(f"{i + row_start}:")
                core_line.append(bar(c, width=6, warn=CPU_WARN, crit=CPU_CRIT, show_label=False))
                core_line.append(" ")
            lines.append(core_line)

    # Package temp + clocks
    foot = Text()
    foot.append("pkg: ")
    foot.append(_fmt(snap.cpu_package_temp_c, DEG), style=_color_for(snap.cpu_package_temp_c, 70.0, 85.0))
    foot.append("   ")
    if snap.cpu_core_clocks_mhz:
        foot.append(f"clocks: {', '.join(f'{c:.0f}' for c in snap.cpu_core_clocks_mhz[:4])} MHz")
    else:
        foot.append("clocks: N/A")
    lines.append(foot)

    return Panel(
        Align.left(_stack(lines)),
        title="CPU",
        border_style=_color_for(snap.cpu_pct, CPU_WARN, CPU_CRIT),
        width=width,
        box=box.ASCII,
    )


def _stack(texts: Sequence[Text]) -> Text:
    """Stack multiple Text objects with newlines."""
    out = Text()
    for i, t in enumerate(texts):
        if i:
            out.append("\n")
        out.append(t)
    return out


# ── GPU panel ───────────────────────────────────────────────────────────

def gpu_panel(snap: SystemSnapshotV2, width: int) -> Panel:
    """GPU: util sparkline, VRAM, temp, power/limit, fan, throttle; N/A if unavailable."""
    lines: list[Text] = []

    available = snap.gpu_util_pct is not None

    head = Text()
    if available:
        head.append(f"GPU  {snap.gpu_util_pct:5.1f}%  ", style="bold " + _color_for(snap.gpu_util_pct, GPU_WARN, GPU_CRIT))
        hist_vals = []  # history not passed here; keep a flat sparkline
        head.append(sparkline([snap.gpu_util_pct], width=max(10, width - 20), warn=GPU_WARN, crit=GPU_CRIT))
    else:
        head.append("GPU  N/A", style="gray50")
    lines.append(head)

    # VRAM bar
    vram_line = Text()
    vram_line.append("VRAM: ")
    if snap.gpu_mem_used_mb is not None and snap.gpu_mem_total_mb:
        vram_pct = (snap.gpu_mem_used_mb / snap.gpu_mem_total_mb) * 100.0
        vram_line.append(bar(vram_pct, width=max(10, width - 18), warn=GPU_WARN, crit=GPU_CRIT))
        vram_line.append(f"  {_fmt(snap.gpu_mem_used_mb, ' MB', 1)}/{_fmt(snap.gpu_mem_total_mb, ' MB', 1)}")
    else:
        vram_line.append(bar(None, width=max(10, width - 18)))
        vram_line.append("  N/A")
    lines.append(vram_line)

    # Temp / power / fan / throttle
    stat = Text()
    stat.append("temp: ")
    stat.append(_fmt(snap.gpu_temp_c, DEG), style=_color_for(snap.gpu_temp_c, 75.0, 85.0))
    stat.append("   ")
    if snap.gpu_power_w is not None and snap.gpu_power_limit_w is not None:
        stat.append(f"power: {snap.gpu_power_w:.0f}/{snap.gpu_power_limit_w:.0f} W   ")
    else:
        stat.append("power: N/A   ")
    stat.append(f"fan: {_fmt(snap.gpu_fan_pct, '%')}")
    lines.append(stat)

    throttle = ", ".join(snap.gpu_throttle_reasons) if snap.gpu_throttle_reasons else "none"
    lines.append(Text(f"throttle: {throttle}", style="gray50"))

    border = "green" if available else "gray50"
    return Panel(_stack(lines), title="GPU", border_style=border, width=width, box=box.ASCII)


# ── Memory panel ────────────────────────────────────────────────────────

def memory_panel(snap: SystemSnapshotV2, width: int) -> Panel:
    """RAM usage bar, used/avail/total."""
    lines: list[Text] = []

    head = Text()
    head.append(f"MEM  {snap.ram_pct:5.1f}%  ", style="bold " + _color_for(snap.ram_pct, RAM_WARN, RAM_CRIT))
    head.append(bar(snap.ram_pct, width=max(15, width - 20), warn=RAM_WARN, crit=RAM_CRIT))
    lines.append(head)

    lines.append(
        Text(
            f"used {_fmt(snap.ram_used_mb, ' MB', 1)}{SEP}"
            f"avail {_fmt(snap.ram_available_mb, ' MB', 1)}{SEP}"
            f"total {_fmt(snap.ram_total_mb, ' MB', 1)}"
        )
    )
    return Panel(_stack(lines), title="MEM", border_style=_color_for(snap.ram_pct, RAM_WARN, RAM_CRIT), width=width, box=box.ASCII)


# ── Disk panel ──────────────────────────────────────────────────────────

def disk_panel(snap: SystemSnapshotV2, width: int) -> Panel:
    """Disk usage bar, free/total GB."""
    lines: list[Text] = []

    head = Text()
    head.append(f"DISK  {snap.disk_pct:5.1f}%  ", style="bold " + _color_for(snap.disk_pct, DISK_WARN, DISK_CRIT))
    head.append(bar(snap.disk_pct, width=max(15, width - 20), warn=DISK_WARN, crit=DISK_CRIT))
    lines.append(head)

    lines.append(
        Text(
            f"free {_fmt(snap.disk_free_gb, ' GB', 1)}{SEP}"
            f"total {_fmt(snap.disk_total_gb, ' GB', 1)}"
        )
    )
    return Panel(_stack(lines), title="DISK", border_style=_color_for(snap.disk_pct, DISK_WARN, DISK_CRIT), width=width, box=box.ASCII)


# ── Network panel ───────────────────────────────────────────────────────

def network_panel(snap: SystemSnapshotV2, history: Sequence[SystemSnapshotV2], width: int) -> Panel:
    """Bandwidth + packet rate with bandwidth sparkline and spike ratio."""
    hist = list(history)[-60:]
    bw_vals = [s.bandwidth_kbps for s in hist]

    lines: list[Text] = []

    head = Text()
    head.append("NET  ", style="bold cyan")
    head.append(f"{snap.bandwidth_kbps:8.1f} kbps  ", style="bold")
    head.append(sparkline(bw_vals, width=max(15, width - 30), warn=500.0, crit=1000.0))
    lines.append(head)

    lines.append(Text(f"pkts: {snap.packet_rate_pps:7.1f} pps   spike: {snap.bw_spike:4.2f}x"))
    return Panel(_stack(lines), title="NET", border_style="cyan", width=width, box=box.ASCII)


# ── Processes panel ─────────────────────────────────────────────────────

def processes_panel(snap: SystemSnapshotV2, width: int, max_rows: int = 6) -> Panel:
    """Top CPU and top GPU processes side by side."""
    table = Table(show_header=True, header_style="bold cyan", width=width, box=box.ASCII)
    table.add_column("PID", style="cyan", width=6)
    table.add_column("Process", width=max(14, width // 3))
    table.add_column("CPU%", justify="right", width=7)
    table.add_column("GPU MB", justify="right", width=8)

    top_cpu = snap.top_cpu_processes or []
    top_gpu = {p.pid: p for p in (snap.top_gpu_processes or [])}

    rows = min(max_rows, len(top_cpu))
    for i in range(rows):
        p = top_cpu[i]
        gpu = top_gpu.get(p.pid)
        gpu_mb = f"{gpu.gpu_memory_mb:.0f}" if gpu and gpu.gpu_memory_mb is not None else "—"
        cpu_color = _color_for(p.cpu_percent, CPU_WARN, CPU_CRIT)
        table.add_row(
            str(p.pid),
            (p.name or "unknown")[: max(10, width // 3 - 2)],
            f"{p.cpu_percent:.1f}",
            gpu_mb,
            end_section=False,
        )
        # color the CPU% cell
        table.columns[2].style = cpu_color

    if rows == 0:
        table.add_row("—", "no processes", "—", "—")

    return Panel(table, title="Processes (top CPU)", border_style="gray50", width=width, box=box.ASCII)


# ── Alerts panel ────────────────────────────────────────────────────────

def alerts_panel(
    alerts: Sequence[Alert],
    width: int,
    height: int,
    scroll: int = 0,
    selected: Optional[int] = None,
) -> Panel:
    """Live alert feed, newest first, colored by severity.

    ``scroll`` is the number of top items skipped; ``selected`` is the
    index (into the displayed, scroll-adjusted list) that 'n'/'d' act on.
    """
    # Counts by severity
    counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for a in alerts:
        sev = a.severity.lower() if a.severity else "info"
        counts[sev] = counts.get(sev, 0) + 1

    title = (
        f"ALERTS  "
        f"[red bold]{counts['critical']} crit[/]  "
        f"[yellow]{counts['warning']} warn[/]  "
        f"[dim]{counts['info']} info[/]"
    )

    # Newest first; apply scroll.
    ordered = list(reversed(list(alerts)))
    visible = ordered[scroll : scroll + max(0, height - 3)]

    table = Table(show_header=False, box=box.ASCII, width=width, padding=(0, 1))
    table.add_column("time", width=17, style="gray50", no_wrap=True)
    table.add_column("sev", width=9, no_wrap=True)
    table.add_column("title", style="white")

    for a in visible:
        sev = (a.severity or "info").lower()
        style, prefix = SEVERITY_STYLE.get(sev, ("dim", "- "))
        if selected is not None and visible.index(a) == selected:
            style = f"reverse {style}"
        table.add_row(a.timestamp[:16], f"{prefix}{sev}", a.title, style=style)

    return Panel(table, title=title, border_style="gray50", width=width, height=height, box=box.ASCII)


# ── Footer ──────────────────────────────────────────────────────────────

def footer_panel(hints: str | None = None) -> Panel:
    if hints is None:
        hints = f"q=quit  tab=focus  {ARROWS}=scroll  n=normal  d=dismiss"
    return Panel(Text(hints, justify="center", style="dim"), border_style="gray50", box=box.ASCII)
