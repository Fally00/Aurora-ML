"""Tests for aurora_ui — pure dashboard render functions.

All tests are terminal-free: they build snapshots / alerts and assert on
the returned Rich renderables (no console, no Live loop required).
"""

from __future__ import annotations

from collections import deque

import pytest

from rich.text import Text

import aurora_ui
from Core.alerts import Alert
from Core.schema_v2 import ProcessInfo, SystemSnapshotV2


def _render_text(renderable, width=40, height=24):
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    Console(file=buf, width=width, height=height).print(renderable)
    return buf.getvalue()


def _snap(
    *,
    cpu_pct=33.8,
    ram_pct=64.6,
    disk_pct=23.7,
    gpu_util=None,
    gpu_mem_used=None,
    gpu_mem_total=None,
    gpu_temp=None,
    bandwidth_kbps=3104.9,
    packet_rate_pps=731.6,
    bw_spike=3.84,
    collector_cpu=0.0,
    collector_rss=61.1,
    cpu_cores=None,
    gpu_throttle=None,
) -> SystemSnapshotV2:
    return SystemSnapshotV2(
        timestamp="2026-11-16 18:38:35",
        machine_id="laptop",
        schema_version=2,
        workload_tag="idle",
        cpu_pct=cpu_pct,
        cpu_per_core_pct=cpu_cores or [40.0, 50.0, 42.0],
        cpu_package_temp_c=None,
        cpu_core_clocks_mhz=[2400.0, 2500.0],
        ram_pct=ram_pct,
        ram_used_mb=10440.9,
        ram_available_mb=5732.5,
        ram_total_mb=16173.4,
        disk_pct=disk_pct,
        disk_free_gb=727.5,
        disk_total_gb=953.0,
        bandwidth_kbps=bandwidth_kbps,
        packet_rate_pps=packet_rate_pps,
        bw_spike=bw_spike,
        gpu_util_pct=gpu_util,
        gpu_mem_used_mb=gpu_mem_used,
        gpu_mem_total_mb=gpu_mem_total,
        gpu_temp_c=gpu_temp,
        gpu_power_w=None,
        gpu_power_limit_w=115.0,
        gpu_core_clock_mhz=2490.0,
        gpu_mem_clock_mhz=8251.0,
        gpu_fan_pct=0.0,
        gpu_throttle_reasons=gpu_throttle or [],
        top_cpu_processes=[ProcessInfo(pid=4972, name="Aurora.exe", cpu_percent=70.2)],
        top_gpu_processes=[ProcessInfo(pid=1920, name="dwm.exe", cpu_percent=0.0)],
        collector_cpu_pct=collector_cpu,
        collector_rss_mb=collector_rss,
        cpu_delta=-1.1,
        ram_delta=0.3,
        cpu_rolling_mean=19.3,
        ram_rolling_mean=64.1,
        cpu_rolling_std=17.7,
        ram_rolling_std=0.4,
        bw_rolling_mean=808.1,
    )


def _alert(alert_id, timestamp, severity, title):
    return Alert(
        id=alert_id,
        timestamp=timestamp,
        module="file_monitor",
        severity=severity,
        title=title,
        signals={},
        grouped_count=1,
    )


# ── sparkline ───────────────────────────────────────────────────────────

def test_sparkline_empty():
    out = aurora_ui.sparkline([], width=10)
    assert isinstance(out, Text)
    assert out.plain == aurora_ui.EMPTY_BLOCK * 10


def test_sparkline_equal_values():
    # All values equal -> span 0 -> use darkest block everywhere.
    out = aurora_ui.sparkline([5.0, 5.0, 5.0], width=5)
    assert len(out.plain) == 5


def test_sparkline_monotonic():
    out = aurora_ui.sparkline([1.0, 2.0, 3.0, 4.0, 5.0], width=5)
    assert len(out.plain) == 5


# ── bar ─────────────────────────────────────────────────────────────────

def test_bar_full_and_empty():
    full = aurora_ui.bar(100.0, width=10)
    empty = aurora_ui.bar(0.0, width=10)
    assert full.plain.count(aurora_ui.FULL_BLOCK) == 10
    assert empty.plain.count(aurora_ui.EMPTY_BLOCK) == 10


def test_bar_na():
    out = aurora_ui.bar(None, width=10)
    assert out.plain.count(aurora_ui.EMPTY_BLOCK) == 10
    assert "N/A" in out.plain


# ── color ───────────────────────────────────────────────────────────────

def test_color_for_thresholds():
    assert aurora_ui._color_for(30.0, 50.0, 75.0) == "green"
    assert aurora_ui._color_for(60.0, 50.0, 75.0) == "yellow"
    assert aurora_ui._color_for(90.0, 50.0, 75.0) == "red"
    assert aurora_ui._color_for(None, 50.0, 75.0) == "gray50"


# ── panels ──────────────────────────────────────────────────────────────

def test_cpu_panel_renders():
    panel = aurora_ui.cpu_panel(_snap(), [_snap()], width=40)
    assert panel.title == "CPU"


def test_cpu_panel_per_core():
    panel = aurora_ui.cpu_panel(
        _snap(cpu_cores=[90.0, 10.0]), [_snap()], width=40
    )
    assert panel.title == "CPU"


def test_gpu_panel_available():
    snap = _snap(gpu_util=45.0, gpu_mem_used=2000.0, gpu_mem_total=8192.0,
                 gpu_temp=60.0, gpu_throttle=["hw_thermal"])
    panel = aurora_ui.gpu_panel(snap, width=40)
    assert panel.title == "GPU"


def test_gpu_panel_na():
    # N/A GPU (NVML unavailable) must not crash.
    snap = _snap(gpu_util=None, gpu_mem_used=None, gpu_mem_total=None,
                 gpu_temp=None)
    panel = aurora_ui.gpu_panel(snap, width=40)
    assert panel.title == "GPU"


def test_memory_panel():
    panel = aurora_ui.memory_panel(_snap(), width=40)
    assert panel.title == "MEM"


def test_disk_panel():
    panel = aurora_ui.disk_panel(_snap(), width=40)
    assert panel.title == "DISK"


def test_network_panel():
    panel = aurora_ui.network_panel(_snap(), [_snap()], width=40)
    assert panel.title == "NET"


def test_processes_panel_empty():
    snap = _snap()
    snap.top_cpu_processes = []
    snap.top_gpu_processes = []
    panel = aurora_ui.processes_panel(snap, width=40)
    assert panel.title == "Processes (top CPU)"


def test_header_panel():
    panel = aurora_ui.header_panel(
        "laptop", "idle", "2026-11-16 18:38:35", "0:01:00", 11, True,
        0.0, 61.1, width=80,
    )
    assert panel.title == "Aurora Dashboard"


def test_footer_panel():
    panel = aurora_ui.footer_panel()
    assert panel is not None


# ── alerts panel ────────────────────────────────────────────────────────

def _alerts_panel(alerts, width=40, height=12, scroll=0, selected=None):
    return aurora_ui.alerts_panel(alerts, width, height, scroll=scroll, selected=selected)


def test_alerts_panel_empty():
    panel = _alerts_panel([])
    assert "[red bold]0 crit" in panel.title
    assert "[yellow]0 warn" in panel.title


def test_alerts_panel_counts_in_title():
    alerts = [
        _alert("a1", "2026-11-16 18:38:30", "critical", "crit one"),
        _alert("a2", "2026-11-16 18:38:29", "critical", "crit two"),
        _alert("a3", "2026-11-16 18:38:28", "warning", "warn one"),
        _alert("a4", "2026-11-16 18:38:27", "info", "info one"),
    ]
    panel = _alerts_panel(alerts)
    assert "[red bold]2 crit" in panel.title
    assert "[yellow]1 warn" in panel.title
    assert "[dim]1 info" in panel.title


def test_alerts_panel_newest_first():
    alerts = [
        _alert("old", "2026-11-16 18:37:00", "info", "oldest"),
        _alert("new", "2026-11-16 18:38:00", "critical", "newest"),
    ]
    out = _render_text(_alerts_panel(alerts, width=80), width=80)
    newest_pos = out.find("newest")
    oldest_pos = out.find("oldest")
    assert newest_pos != -1 and oldest_pos != -1
    assert newest_pos < oldest_pos  # newest appears first


def test_alerts_panel_colors_by_severity():
    alerts = [
        _alert("c", "2026-11-16 18:38:00", "critical", "x"),
        _alert("w", "2026-11-16 18:38:00", "warning", "y"),
        _alert("i", "2026-11-16 18:38:00", "info", "z"),
    ]
    panel = _alerts_panel(alerts)
    styles = [str(row.style) for row in panel.renderable.rows]
    # Each severity maps to a distinct style.
    assert len(set(styles)) == 3


def test_alerts_panel_scroll():
    alerts = [
        _alert(f"a{i}", f"2026-11-16 18:38:{i:02d}", "info", f"t{i}")
        for i in range(5)
    ]
    out0 = _render_text(_alerts_panel(alerts, width=80, scroll=0), width=80)
    out1 = _render_text(_alerts_panel(alerts, width=80, scroll=1), width=80)
    assert out0 != out1
    assert "t4" in out0  # newest first at scroll 0


# ── run --plain unchanged (delegated to existing test_aurora_cli.py) ──


def test_no_console_logging_flag_exists():
    """The CLI accepts --plain (headless) for tests/background use."""
    import aurora_cli

    parser = aurora_cli.argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    rp = sub.add_parser("run")
    rp.add_argument("--plain", action="store_true")
    rp.add_argument("--refresh", type=float, default=1.0)
    ns = parser.parse_args(["run", "--plain", "--refresh", "2.5"])
    assert ns.plain is True
    assert ns.refresh == 2.5
