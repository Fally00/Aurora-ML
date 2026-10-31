"""
Tests for Core.schema_v2 — round-trip serialization, compatibility methods.
"""

from __future__ import annotations

import json
import pytest

from Core.schema_v2 import SystemSnapshotV2, ProcessInfo, daily_table_name, INSERT_COLUMNS


class TestProcessInfo:
    def test_process_info_to_dict(self):
        p = ProcessInfo(pid=1234, name="test.exe", cpu_percent=12.5, gpu_memory_mb=512.0, gpu_util_percent=45.0)
        d = p.to_dict()
        assert d["pid"] == 1234
        assert d["name"] == "test.exe"
        assert d["cpu_percent"] == 12.5
        assert d["gpu_memory_mb"] == 512.0
        assert d["gpu_util_percent"] == 45.0

    def test_process_info_from_dict(self):
        d = {"pid": 5678, "name": "gpu_app.exe", "cpu_percent": 8.0, "gpu_memory_mb": 1024.0, "gpu_util_percent": 60.0}
        p = ProcessInfo.from_dict(d)
        assert p.pid == 5678
        assert p.name == "gpu_app.exe"
        assert p.cpu_percent == 8.0
        assert p.gpu_memory_mb == 1024.0
        assert p.gpu_util_percent == 60.0

    def test_process_info_optional_fields_none(self):
        p = ProcessInfo(pid=999, name="no_gpu.exe", cpu_percent=1.0)
        d = p.to_dict()
        assert d["gpu_memory_mb"] is None
        assert d["gpu_util_percent"] is None

        p2 = ProcessInfo.from_dict({"pid": 999, "name": "no_gpu.exe", "cpu_percent": 1.0})
        assert p2.gpu_memory_mb is None
        assert p2.gpu_util_percent is None


class TestSystemSnapshotV2:
    def test_snapshot_creation_defaults(self):
        snap = SystemSnapshotV2()
        assert snap.timestamp != ""
        assert snap.timestamp_ns > 0
        assert snap.machine_id == "laptop"
        assert snap.schema_version == 2
        assert snap.workload_tag == "idle"

    def test_snapshot_to_sqlite_row(self):
        snap = SystemSnapshotV2(
            timestamp="2024-01-01 12:00:00",
            timestamp_ns=1704110400000000000,
            machine_id="test_machine",
            schema_version=2,
            workload_tag="gaming",
            cpu_pct=45.5,
            cpu_per_core_pct=[40.0, 50.0, 42.0, 48.0],
            cpu_package_temp_c=65.0,
            cpu_core_clocks_mhz=[3200.0, 3300.0, 3100.0, 3400.0],
            ram_pct=60.0,
            ram_used_mb=8192.0,
            ram_available_mb=5376.0,
            ram_total_mb=13568.0,
            disk_pct=45.0,
            disk_free_gb=500.0,
            disk_total_gb=1000.0,
            bandwidth_kbps=1024.0,
            packet_rate_pps=500.0,
            gpu_util_pct=75.0,
            gpu_mem_used_mb=4096.0,
            gpu_mem_total_mb=8192.0,
            gpu_mem_pct=50.0,
            gpu_temp_c=68.0,
            gpu_power_w=150.0,
            gpu_power_limit_w=170.0,
            gpu_core_clock_mhz=1800.0,
            gpu_mem_clock_mhz=7000.0,
            gpu_fan_pct=55.0,
            gpu_throttle_reasons=["hw_thermal"],
            top_cpu_processes=[ProcessInfo(pid=100, name="game.exe", cpu_percent=30.0)],
            top_gpu_processes=[ProcessInfo(pid=100, name="game.exe", cpu_percent=30.0, gpu_memory_mb=2048.0)],
            collector_cpu_pct=0.5,
            collector_rss_mb=45.0,
        )

        row = snap.to_sqlite_row()

        # Check all columns present
        for col in INSERT_COLUMNS:
            assert col in row, f"Missing column: {col}"

        # Check JSON serialization
        assert isinstance(row["cpu_per_core_pct"], str)
        assert json.loads(row["cpu_per_core_pct"]) == [40.0, 50.0, 42.0, 48.0]

        assert isinstance(row["cpu_core_clocks_mhz"], str)
        assert json.loads(row["cpu_core_clocks_mhz"]) == [3200.0, 3300.0, 3100.0, 3400.0]

        assert isinstance(row["gpu_throttle_reasons"], str)
        assert json.loads(row["gpu_throttle_reasons"]) == ["hw_thermal"]

        assert isinstance(row["top_cpu_processes"], str)
        cpu_procs = json.loads(row["top_cpu_processes"])
        assert len(cpu_procs) == 1
        assert cpu_procs[0]["pid"] == 100

        assert isinstance(row["top_gpu_processes"], str)
        gpu_procs = json.loads(row["top_gpu_processes"])
        assert len(gpu_procs) == 1
        assert gpu_procs[0]["gpu_memory_mb"] == 2048.0

        # Check None values are preserved as None (for SQLite NULL)
        assert row["cpu_package_temp_c"] == 65.0
        snap2 = SystemSnapshotV2(cpu_package_temp_c=None)
        row2 = snap2.to_sqlite_row()
        assert row2["cpu_package_temp_c"] is None

    def test_snapshot_from_sqlite_row(self):
        # Create a row as it would come from SQLite
        row = {
            "timestamp": "2024-01-01 12:00:00",
            "timestamp_ns": 1704110400000000000,
            "machine_id": "test_machine",
            "schema_version": 2,
            "workload_tag": "gaming",
            "cpu_pct": 45.5,
            "cpu_per_core_pct": json.dumps([40.0, 50.0, 42.0, 48.0]),
            "cpu_package_temp_c": 65.0,
            "cpu_core_clocks_mhz": json.dumps([3200.0, 3300.0, 3100.0, 3400.0]),
            "ram_pct": 60.0,
            "ram_used_mb": 8192.0,
            "ram_available_mb": 5376.0,
            "ram_total_mb": 13568.0,
            "disk_pct": 45.0,
            "disk_free_gb": 500.0,
            "disk_total_gb": 1000.0,
            "bandwidth_kbps": 1024.0,
            "packet_rate_pps": 500.0,
            "gpu_util_pct": 75.0,
            "gpu_mem_used_mb": 4096.0,
            "gpu_mem_total_mb": 8192.0,
            "gpu_mem_pct": 50.0,
            "gpu_temp_c": 68.0,
            "gpu_power_w": 150.0,
            "gpu_power_limit_w": 170.0,
            "gpu_core_clock_mhz": 1800.0,
            "gpu_mem_clock_mhz": 7000.0,
            "gpu_fan_pct": 55.0,
            "gpu_throttle_reasons": json.dumps(["hw_thermal"]),
            "top_cpu_processes": json.dumps([{"pid": 100, "name": "game.exe", "cpu_percent": 30.0, "gpu_memory_mb": None, "gpu_util_percent": None}]),
            "top_gpu_processes": json.dumps([{"pid": 100, "name": "game.exe", "cpu_percent": 30.0, "gpu_memory_mb": 2048.0, "gpu_util_percent": 80.0}]),
            "collector_cpu_pct": 0.5,
            "collector_rss_mb": 45.0,
            "cpu_delta": 0.0,
            "ram_delta": 0.0,
            "cpu_rolling_mean": 0.0,
            "ram_rolling_mean": 0.0,
            "cpu_rolling_std": 0.0,
            "ram_rolling_std": 0.0,
            "bw_rolling_mean": 0.0,
            "bw_spike": 1.0,
        }

        snap = SystemSnapshotV2.from_sqlite_row(row)

        assert snap.timestamp == "2024-01-01 12:00:00"
        assert snap.timestamp_ns == 1704110400000000000
        assert snap.machine_id == "test_machine"
        assert snap.schema_version == 2
        assert snap.workload_tag == "gaming"
        assert snap.cpu_pct == 45.5
        assert snap.cpu_per_core_pct == [40.0, 50.0, 42.0, 48.0]
        assert snap.cpu_package_temp_c == 65.0
        assert snap.cpu_core_clocks_mhz == [3200.0, 3300.0, 3100.0, 3400.0]
        assert snap.gpu_util_pct == 75.0
        assert snap.gpu_throttle_reasons == ["hw_thermal"]
        assert len(snap.top_cpu_processes) == 1
        assert snap.top_cpu_processes[0].pid == 100
        assert len(snap.top_gpu_processes) == 1
        assert snap.top_gpu_processes[0].gpu_memory_mb == 2048.0

    def test_snapshot_round_trip(self):
        """Full round-trip: snapshot -> sqlite_row -> snapshot"""
        original = SystemSnapshotV2(
            timestamp="2024-01-01 12:00:00",
            timestamp_ns=1704110400000000000,
            machine_id="roundtrip_test",
            schema_version=2,
            workload_tag="compile",
            cpu_pct=80.0,
            cpu_per_core_pct=[75.0, 85.0, 82.0, 78.0],
            cpu_package_temp_c=72.0,
            cpu_core_clocks_mhz=[3500.0, 3600.0, 3400.0, 3700.0],
            ram_pct=70.0,
            ram_used_mb=12288.0,
            ram_available_mb=5120.0,
            ram_total_mb=17408.0,
            disk_pct=60.0,
            disk_free_gb=400.0,
            disk_total_gb=1000.0,
            bandwidth_kbps=512.0,
            packet_rate_pps=250.0,
            gpu_util_pct=90.0,
            gpu_mem_used_mb=6144.0,
            gpu_mem_total_mb=8192.0,
            gpu_mem_pct=75.0,
            gpu_temp_c=75.0,
            gpu_power_w=200.0,
            gpu_power_limit_w=220.0,
            gpu_core_clock_mhz=1900.0,
            gpu_mem_clock_mhz=7500.0,
            gpu_fan_pct=70.0,
            gpu_throttle_reasons=["sw_power_cap", "hw_thermal"],
            top_cpu_processes=[ProcessInfo(pid=1, name="gcc", cpu_percent=60.0)],
            top_gpu_processes=[ProcessInfo(pid=2, name="nvcc", cpu_percent=20.0, gpu_memory_mb=4096.0)],
            collector_cpu_pct=1.2,
            collector_rss_mb=60.0,
        )

        row = original.to_sqlite_row()
        restored = SystemSnapshotV2.from_sqlite_row(row)

        # Compare all fields
        assert restored.timestamp == original.timestamp
        assert restored.timestamp_ns == original.timestamp_ns
        assert restored.machine_id == original.machine_id
        assert restored.schema_version == original.schema_version
        assert restored.workload_tag == original.workload_tag
        assert restored.cpu_pct == original.cpu_pct
        assert restored.cpu_per_core_pct == original.cpu_per_core_pct
        assert restored.cpu_package_temp_c == original.cpu_package_temp_c
        assert restored.cpu_core_clocks_mhz == original.cpu_core_clocks_mhz
        assert restored.ram_pct == original.ram_pct
        assert restored.ram_used_mb == original.ram_used_mb
        assert restored.ram_available_mb == original.ram_available_mb
        assert restored.ram_total_mb == original.ram_total_mb
        assert restored.disk_pct == original.disk_pct
        assert restored.disk_free_gb == original.disk_free_gb
        assert restored.disk_total_gb == original.disk_total_gb
        assert restored.bandwidth_kbps == original.bandwidth_kbps
        assert restored.packet_rate_pps == original.packet_rate_pps
        assert restored.gpu_util_pct == original.gpu_util_pct
        assert restored.gpu_mem_used_mb == original.gpu_mem_used_mb
        assert restored.gpu_mem_total_mb == original.gpu_mem_total_mb
        assert restored.gpu_mem_pct == original.gpu_mem_pct
        assert restored.gpu_temp_c == original.gpu_temp_c
        assert restored.gpu_power_w == original.gpu_power_w
        assert restored.gpu_power_limit_w == original.gpu_power_limit_w
        assert restored.gpu_core_clock_mhz == original.gpu_core_clock_mhz
        assert restored.gpu_mem_clock_mhz == original.gpu_mem_clock_mhz
        assert restored.gpu_fan_pct == original.gpu_fan_pct
        assert restored.gpu_throttle_reasons == original.gpu_throttle_reasons
        assert len(restored.top_cpu_processes) == len(original.top_cpu_processes)
        assert restored.top_cpu_processes[0].pid == original.top_cpu_processes[0].pid
        assert len(restored.top_gpu_processes) == len(original.top_gpu_processes)
        assert restored.top_gpu_processes[0].gpu_memory_mb == original.top_gpu_processes[0].gpu_memory_mb
        assert restored.collector_cpu_pct == original.collector_cpu_pct
        assert restored.collector_rss_mb == original.collector_rss_mb

    def test_compatibility_methods(self):
        snap = SystemSnapshotV2(
            cpu_pct=50.0,
            ram_pct=60.0,
            bandwidth_kbps=1000.0,
            packet_rate_pps=500.0,
            disk_pct=40.0,
            cpu_delta=5.0,
            ram_delta=-2.0,
            cpu_rolling_mean=48.0,
            ram_rolling_mean=58.0,
            cpu_rolling_std=3.0,
            ram_rolling_std=4.0,
            bw_rolling_mean=950.0,
            bw_spike=1.1,
        )

        # to_anomaly_input
        anomaly_input = snap.to_anomaly_input()
        assert anomaly_input == {
            "cpu_pct": 50.0,
            "ram_pct": 60.0,
            "bandwidth_kbps": 1000.0,
            "packet_rate_pps": 500.0,
            "protocol": "",
        }

        # to_classifier_input
        classifier_input = snap.to_classifier_input()
        assert classifier_input == {
            "cpu_pct": 50.0,
            "ram_pct": 60.0,
            "bandwidth_kbps": 1000.0,
            "packet_rate_pps": 500.0,
        }

        # to_predictor_input
        predictor_input = snap.to_predictor_input()
        assert predictor_input == {
            "cpu_pct": 50.0,
            "ram_pct": 60.0,
            "disk_pct": 40.0,
            "bandwidth_kbps": 1000.0,
            "packet_rate_pps": 500.0,
            "cpu_delta": 5.0,
            "ram_delta": -2.0,
            "cpu_rolling_mean": 48.0,
            "ram_rolling_mean": 58.0,
            "cpu_rolling_std": 3.0,
            "ram_rolling_std": 4.0,
            "bw_rolling_mean": 950.0,
            "bw_spike": 1.1,
        }

    def test_optional_gpu_fields_none(self):
        snap = SystemSnapshotV2()
        row = snap.to_sqlite_row()

        # All GPU fields should be None
        gpu_fields = [
            "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_pct",
            "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_core_clock_mhz",
            "gpu_mem_clock_mhz", "gpu_fan_pct"
        ]
        for field in gpu_fields:
            assert row[field] is None, f"Expected None for {field}, got {row[field]}"

        # throttle_reasons should be empty list
        assert row["gpu_throttle_reasons"] == "[]"

    def test_optional_cpu_fields_none(self):
        snap = SystemSnapshotV2(cpu_package_temp_c=None, cpu_per_core_pct=[], cpu_core_clocks_mhz=[])
        row = snap.to_sqlite_row()
        assert row["cpu_package_temp_c"] is None
        assert row["cpu_per_core_pct"] == "[]"
        assert row["cpu_core_clocks_mhz"] == "[]"

    def test_empty_process_lists(self):
        snap = SystemSnapshotV2(top_cpu_processes=[], top_gpu_processes=[])
        row = snap.to_sqlite_row()
        assert row["top_cpu_processes"] == "[]"
        assert row["top_gpu_processes"] == "[]"

        restored = SystemSnapshotV2.from_sqlite_row(row)
        assert restored.top_cpu_processes == []
        assert restored.top_gpu_processes == []


class TestDailyTableName:
    def test_daily_table_name_format(self):
        assert daily_table_name("2024-01-01") == "metrics_20240101"
        assert daily_table_name("2024-12-31") == "metrics_20241231"
        assert daily_table_name("2025-06-15") == "metrics_20250615"

    def test_daily_table_name_invalid(self):
        # Should still work with any string, just replaces hyphens
        assert daily_table_name("invalid") == "metrics_invalid"


class TestInsertColumns:
    def test_insert_columns_complete(self):
        # Ensure all expected columns are present
        expected = {
            "timestamp", "timestamp_ns", "machine_id", "schema_version", "workload_tag",
            "cpu_pct", "cpu_per_core_pct", "cpu_package_temp_c", "cpu_core_clocks_mhz",
            "ram_pct", "ram_used_mb", "ram_available_mb", "ram_total_mb",
            "disk_pct", "disk_free_gb", "disk_total_gb",
            "bandwidth_kbps", "packet_rate_pps",
            "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_pct",
            "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_core_clock_mhz",
            "gpu_mem_clock_mhz", "gpu_fan_pct", "gpu_throttle_reasons",
            "top_cpu_processes", "top_gpu_processes",
            "collector_cpu_pct", "collector_rss_mb",
            "cpu_delta", "ram_delta", "cpu_rolling_mean", "ram_rolling_mean",
            "cpu_rolling_std", "ram_rolling_std", "bw_rolling_mean", "bw_spike",
        }
        assert set(INSERT_COLUMNS) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])