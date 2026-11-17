from __future__ import annotations

import asyncio

import pytest
from textual.widgets import DataTable, Input, Static

from aurora_ui import (
    AuroraApp,
    alert_counts,
    filtered_alerts,
    parse_command,
    process_rows,
    usage_bar,
)
from Core.alerts import Alert
from Core.schema_v2 import ProcessInfo, SystemSnapshotV2
from Data.storage import MetricsDB


def _snapshot(**kwargs) -> SystemSnapshotV2:
    fields = {
        "machine_id": "laptop",
        "workload_tag": "gaming",
        "cpu_pct": 42.0,
        "cpu_per_core_pct": [0.0, 9.0, 42.0],
        "cpu_package_temp_c": None,
        "ram_pct": 50.0,
        "ram_used_mb": 4000.0,
        "ram_total_mb": 8000.0,
        "disk_pct": 20.0,
        "disk_free_gb": 400.0,
        "disk_total_gb": 500.0,
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "top_cpu_processes": [
            ProcessInfo(pid=100, name="worker-with-a-very-long-process-name.exe", cpu_percent=120.0),
            ProcessInfo(pid=0, name="System Idle Process", cpu_percent=98.0),
        ],
        "top_gpu_processes": [
            ProcessInfo(pid=200, name="gpu-worker.exe", cpu_percent=6.0, gpu_memory_mb=2048.0),
        ],
    }
    fields.update(kwargs)
    return SystemSnapshotV2(**fields)


def _alert(severity: str, title: str, timestamp: str = "2026-11-17 01:00:00") -> Alert:
    return Alert(
        module="module_b",
        severity=severity,
        title=title,
        timestamp=timestamp,
        signals={},
    )


def _pilot(app: AuroraApp, scenario, size=(120, 45)) -> None:
    async def run_scenario():
        async with app.run_test(size=size) as pilot:
            await scenario(pilot)

    asyncio.run(run_scenario())


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("status", ("status", None)),
        ("alerts 20", ("alerts", "20")),
        ("tag stress", ("tag", "stress")),
        ("filter crit", ("filter", "crit")),
        ("normal", ("normal", None)),
        ("dismiss alert-1", ("dismiss", "alert-1")),
        ("quit", ("quit", None)),
    ],
)
def test_command_parser(source, expected):
    command = parse_command(source)
    assert (command.name, command.argument) == expected


def test_command_parser_rejects_unknown_and_invalid_arguments():
    with pytest.raises(ValueError, match="Unknown command"):
        parse_command("reboot")
    with pytest.raises(ValueError, match="Invalid arguments"):
        parse_command("tag unknown")


def test_usage_bar_tracks_real_values():
    assert usage_bar(0, 20).plain == "░" * 20
    assert usage_bar(9, 20).plain.count("█") == 2
    assert usage_bar(100, 20).plain == "█" * 20


def test_empty_database_and_unavailable_gpu_render(tmp_path):
    storage = MetricsDB(str(tmp_path / "empty.db"))
    app = AuroraApp(storage)

    async def scenario(_pilot):
        assert "GPU  N/A" in app.query_one("#gpu", Static).content.plain
        assert app.query_one("#alerts", DataTable).row_count == 0

    try:
        _pilot(app, scenario)
    finally:
        storage.close()


def test_compact_layout_keeps_command_bar_visible(tmp_path):
    storage = MetricsDB(str(tmp_path / "compact.db"))
    app = AuroraApp(storage)

    async def scenario(pilot):
        assert app.query_one("#command").region.bottom <= 24
        assert app.query_one("#metrics-grid").region.y >= 0
        assert app.query_one("#tables-grid").region.y >= 0
        await pilot.resize_terminal(120, 45)
        await pilot.pause()
        assert app.query_one("#metrics-grid").region.height > 8

    try:
        _pilot(app, scenario, size=(80, 24))
    finally:
        storage.close()


def test_interval_refresh_updates_widgets(tmp_path):
    storage = MetricsDB(str(tmp_path / "refresh.db"))
    app = AuroraApp(storage)

    async def scenario(pilot):
        storage.insert(_snapshot())
        await pilot.pause(1.2)
        cpu = app.query_one("#cpu", Static).content.plain
        assert "CPU 42.0%" in cpu
        assert "Per-core:" in cpu
        assert "CPU temp: N/A (LibreHardwareMonitor not running)" in (
            app.query_one("#warning-strip", Static).content.plain
        )
        assert "tag=gaming" in app.query_one("#header", Static).content.plain

    try:
        _pilot(app, scenario)
    finally:
        storage.close()


def test_process_rows_exclude_idle_and_bound_cpu_and_text():
    rows = process_rows(_snapshot())
    assert [row[0] for row in rows] == ["100", "200"]
    assert rows[0][2] == "100.0"
    assert rows[0][1].endswith("…")


def test_alert_filters_counts_and_newest_first():
    alerts = [
        _alert("info", "older", "2026-11-17 00:59:00"),
        _alert("warning", "warning"),
        _alert("critical", "critical", "2026-11-17 01:01:00"),
    ]
    assert alert_counts(alerts) == {"critical": 1, "warning": 1, "info": 1}
    assert [item.title for item in filtered_alerts(alerts, "warn")] == ["warning"]
    assert [item.title for item in alerts] == ["older", "warning", "critical"]


def test_alert_table_order_colors_and_count_title(tmp_path):
    storage = MetricsDB(str(tmp_path / "alert-view.db"))
    storage.record_alert(_alert("info", "older", "2026-11-17 00:59:00"))
    storage.record_alert(_alert("critical", "newest", "2026-11-17 01:01:00"))
    app = AuroraApp(storage)

    async def scenario(_pilot):
        table = app.query_one("#alerts", DataTable)
        assert table.get_row_at(0)[2].plain == "newest"
        assert str(table.get_row_at(0)[1].style) == "bold red"
        assert "1 crit" in str(app.query_one("#alerts-title", Static).content)
        assert "1 info" in str(app.query_one("#alerts-title", Static).content)

    try:
        _pilot(app, scenario)
    finally:
        storage.close()


def test_filter_pause_and_resume_keys(tmp_path):
    storage = MetricsDB(str(tmp_path / "commands.db"))
    storage.record_alert(_alert("info", "info alert"))
    storage.record_alert(_alert("warning", "warning alert"))
    app = AuroraApp(storage)

    async def scenario(pilot):
        alerts = app.query_one("#alerts", DataTable)
        assert alerts.row_count == 2
        await pilot.press(":")
        command = app.query_one("#command", Input)
        assert app.focused is command
        await pilot.press(*tuple("filter"), "space", *tuple("warn"), "enter")
        await pilot.pause()
        assert alerts.row_count == 1
        await pilot.press("escape")
        assert app.focused is not command
        await pilot.press("p")
        assert app.paused
        storage.record_alert(_alert("warning", "new warning"))
        await pilot.pause(1.1)
        assert alerts.row_count == 1
        await pilot.press("p")
        assert not app.paused
        await pilot.pause()
        assert alerts.row_count == 2

    try:
        _pilot(app, scenario)
    finally:
        storage.close()


def test_alert_keys_and_quit(tmp_path):
    storage = MetricsDB(str(tmp_path / "keys.db"))
    alert = _alert("warning", "selected warning")
    storage.record_alert(alert)
    app = AuroraApp(storage)

    async def scenario(pilot):
        table = app.query_one("#alerts", DataTable)
        table.move_cursor(row=0)
        await pilot.press("n")
        assert storage.get_alerts()[0].marked_normal
        await pilot.press("d")
        assert storage.get_alerts()[0].dismissed
        await pilot.press("q")

    try:
        _pilot(app, scenario)
    finally:
        storage.close()


def test_dashboard_does_not_write_to_stdout_or_stderr(tmp_path, capsys):
    storage = MetricsDB(str(tmp_path / "silent.db"))
    app = AuroraApp(storage)

    async def scenario(_pilot):
        return

    try:
        _pilot(app, scenario)
    finally:
        storage.close()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
