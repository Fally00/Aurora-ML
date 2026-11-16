"""
Tests for Data.storage — SQLite WAL mode, daily rotation, schema versioning.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Data.storage import MetricsDB, get_db, close_db
from Core.alerts import Alert
from Core.schema_v2 import SystemSnapshotV2, ProcessInfo, daily_table_name, INSERT_COLUMNS


class TestMetricsDB:
    """Tests for MetricsDB class."""

    def setup_method(self):
        """Create a temporary database for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = str(self.temp_dir / "test_aurora.db")
        self.db = MetricsDB(self.db_path)

    def teardown_method(self):
        """Clean up after each test."""
        self.db.close()
        # Force close any remaining connections
        import gc
        gc.collect()
        time.sleep(0.1)
        # Remove temp directory
        try:
            for f in self.temp_dir.glob("*"):
                f.unlink(missing_ok=True)
            self.temp_dir.rmdir()
        except Exception:
            pass

    def test_flush_commits_current_thread_connection(self):
        connection = self.db._get_conn()
        connection.execute("CREATE TABLE flush_probe (value INTEGER)")
        connection.execute("INSERT INTO flush_probe VALUES (1)")
        self.db.flush()

        with sqlite3.connect(self.db_path) as reader:
            assert reader.execute("SELECT value FROM flush_probe").fetchone() == (1,)

    def test_instances_use_their_configured_database(self):
        """Independent instances must honor their own database paths."""
        other_path = str(self.temp_dir / "other.db")
        other_db = MetricsDB(other_path)
        try:
            assert other_db is not self.db
            assert other_db.db_path == other_path
        finally:
            other_db.close()

    def test_wal_mode_enabled(self):
        """Database should be in WAL mode."""
        conn = self.db._get_conn()
        result = conn.execute("PRAGMA journal_mode;").fetchone()
        assert result[0].upper() == "WAL"

    def test_schema_created(self):
        """Migration metadata exists and daily metric tables are created lazily."""
        conn = self.db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "schema_migrations" in table_names
        assert not any(name.startswith("metrics_") for name in table_names)
        today_table = self.db._get_today_table()
        assert today_table in [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]

    def test_migration_recorded(self):
        """Initial migration should be recorded."""
        conn = self.db._get_conn()
        row = conn.execute(
            "SELECT version, description FROM schema_migrations WHERE version = ?",
            (2,)
        ).fetchone()
        assert row is not None
        assert row[0] == 2
        assert "v2 schema" in row[1].lower()

    def test_insert_snapshot(self):
        """Insert a single snapshot."""
        snap = SystemSnapshotV2(
            timestamp="2024-01-01 12:00:00",
            timestamp_ns=1704110400000000000,
            machine_id="test_machine",
            schema_version=2,
            workload_tag="idle",
            cpu_pct=50.0,
            ram_pct=60.0,
            ram_used_mb=8192.0,
            ram_available_mb=5376.0,
            ram_total_mb=13568.0,
            disk_pct=45.0,
            disk_free_gb=500.0,
            disk_total_gb=1000.0,
            bandwidth_kbps=1024.0,
            packet_rate_pps=500.0,
            collector_cpu_pct=0.5,
            collector_rss_mb=45.0,
        )

        self.db.insert(snap)

        # Verify in main table (today's table)
        conn = self.db._get_conn()
        today_table = daily_table_name(date.today().isoformat())
        row = conn.execute(f"SELECT * FROM {today_table} WHERE machine_id = ?", ("test_machine",)).fetchone()
        assert row is not None
        assert row["cpu_pct"] == 50.0
        assert row["workload_tag"] == "idle"

    def test_insert_batch(self):
        """Insert multiple snapshots in batch."""
        snaps = []
        base_ns = 1704110400000000000
        for i in range(5):
            snaps.append(SystemSnapshotV2(
                timestamp=f"2024-01-01 12:00:{i:02d}",
                timestamp_ns=base_ns + i * 1_000_000_000,
                machine_id="batch_test",
                schema_version=2,
                workload_tag="gaming",
                cpu_pct=50.0 + i,
                ram_pct=60.0,
                ram_used_mb=8192.0,
                ram_available_mb=5376.0,
                ram_total_mb=13568.0,
                disk_pct=45.0,
                disk_free_gb=500.0,
                disk_total_gb=1000.0,
                bandwidth_kbps=1024.0,
                packet_rate_pps=500.0,
                collector_cpu_pct=0.5,
                collector_rss_mb=45.0,
            ))

        self.db.insert_batch(snaps)

        conn = self.db._get_conn()
        today_table = daily_table_name(date.today().isoformat())
        rows = conn.execute(f"SELECT * FROM {today_table} WHERE machine_id = ?", ("batch_test",)).fetchall()
        assert len(rows) == 5
        # Check order preserved
        cpu_values = [r["cpu_pct"] for r in rows]
        assert cpu_values == [50.0, 51.0, 52.0, 53.0, 54.0]

    def test_query_with_filters(self):
        """Query snapshots with time, machine, workload filters."""
        base_ns = 1704110400000000000
        for i in range(10):
            snap = SystemSnapshotV2(
                timestamp=f"2024-01-01 12:00:{i:02d}",
                timestamp_ns=base_ns + i * 1_000_000_000,
                machine_id="query_test" if i < 5 else "other_machine",
                schema_version=2,
                workload_tag="idle" if i % 2 == 0 else "gaming",
                cpu_pct=50.0,
                ram_pct=60.0,
                ram_used_mb=8192.0,
                ram_available_mb=5376.0,
                ram_total_mb=13568.0,
                disk_pct=45.0,
                disk_free_gb=500.0,
                disk_total_gb=1000.0,
                bandwidth_kbps=1024.0,
                packet_rate_pps=500.0,
                collector_cpu_pct=0.5,
                collector_rss_mb=45.0,
            )
            self.db.insert(snap)

        # Query by machine
        results = self.db.query(machine_id="query_test", limit=100)
        assert len(results) == 5

        # Query by workload
        results = self.db.query(workload_tag="gaming", limit=100)
        assert len(results) == 5  # 5 gaming across both machines

        # Query with time range
        start = datetime(2024, 1, 1, 12, 0, 2)
        end = datetime(2024, 1, 1, 12, 0, 5)
        results = self.db.query(start=start, end=end, machine_id="query_test", limit=100)
        assert len(results) == 3  # query_test rows at timestamps 2,3,4

        # Results should be ordered by timestamp DESC
        assert results[0].timestamp_ns > results[-1].timestamp_ns

    def test_query_spans_daily_tables(self):
        """Queries combine rows from current and prior daily tables."""
        snapshot = SystemSnapshotV2(machine_id="multi_day", cpu_pct=30.0)
        self.db.insert(snapshot)

        conn = self.db._get_conn()
        today_table = daily_table_name(date.today().isoformat())
        prior_table = daily_table_name((date.today() - timedelta(days=1)).isoformat())
        conn.execute(f'CREATE TABLE "{prior_table}" AS SELECT * FROM "{today_table}" WHERE 0')
        row = dict(conn.execute(
            f'SELECT * FROM "{today_table}" WHERE machine_id = ?', ("multi_day",)
        ).fetchone())
        row["timestamp_ns"] -= 86_400_000_000_000
        row["timestamp"] = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        values = [row[column] for column in INSERT_COLUMNS]
        conn.execute(
            f'INSERT INTO "{prior_table}" ({", ".join(INSERT_COLUMNS)}) '
            f'VALUES ({", ".join("?" for _ in INSERT_COLUMNS)})',
            values,
        )
        conn.commit()

        results = self.db.query(machine_id="multi_day", limit=10)
        assert len(results) == 2
        assert results[0].timestamp_ns > results[1].timestamp_ns

    def test_get_latest(self):
        """Get most recent snapshot for a machine."""
        base_ns = 1704110400000000000
        for i in range(3):
            snap = SystemSnapshotV2(
                timestamp=f"2024-01-01 12:00:{i:02d}",
                timestamp_ns=base_ns + i * 1_000_000_000,
                machine_id="latest_test",
                schema_version=2,
                workload_tag="idle",
                cpu_pct=50.0,
                ram_pct=60.0,
                ram_used_mb=8192.0,
                ram_available_mb=5376.0,
                ram_total_mb=13568.0,
                disk_pct=45.0,
                disk_free_gb=500.0,
                disk_total_gb=1000.0,
                bandwidth_kbps=1024.0,
                packet_rate_pps=500.0,
                collector_cpu_pct=0.5,
                collector_rss_mb=45.0,
            )
            self.db.insert(snap)

        latest = self.db.get_latest("latest_test")
        assert latest is not None
        assert latest.timestamp_ns == base_ns + 2_000_000_000
        assert latest.timestamp == "2024-01-01 12:00:02"

        # Non-existent machine
        assert self.db.get_latest("nonexistent") is None

    def test_count_today(self):
        """Count rows in today's table."""
        base_ns = 1704110400000000000
        for i in range(7):
            snap = SystemSnapshotV2(
                timestamp=f"2024-01-01 12:00:{i:02d}",
                timestamp_ns=base_ns + i * 1_000_000_000,
                machine_id="count_test",
                schema_version=2,
                workload_tag="idle",
                cpu_pct=50.0,
                ram_pct=60.0,
                ram_used_mb=8192.0,
                ram_available_mb=5376.0,
                ram_total_mb=13568.0,
                disk_pct=45.0,
                disk_free_gb=500.0,
                disk_total_gb=1000.0,
                bandwidth_kbps=1024.0,
                packet_rate_pps=500.0,
                collector_cpu_pct=0.5,
                collector_rss_mb=45.0,
            )
            self.db.insert(snap)

        count = self.db.count_today("count_test")
        assert count == 7

        count_other = self.db.count_today("other")
        assert count_other == 0

    def test_daily_table_rotation(self):
        """Test that daily tables are created for different dates."""
        # This is hard to test without mocking date.today()
        # But we can verify the table naming logic
        assert daily_table_name("2024-01-01") == "metrics_20240101"
        assert daily_table_name("2024-12-31") == "metrics_20241231"

        # Verify _get_today_table creates table with indexes
        table = self.db._get_today_table()
        assert table == daily_table_name(date.today().isoformat())

        conn = self.db._get_conn()
        indexes = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name = ?",
            (table,)
        ).fetchall()
        index_names = [i[0] for i in indexes]
        assert f"idx_{table}_timestamp" in index_names
        assert f"idx_{table}_machine" in index_names
        assert f"idx_{table}_workload" in index_names

    def test_prune_old_data(self):
        """Test pruning drops tables older than RETENTION_DAYS."""
        # We can't easily test this without mocking date.today()
        # But we can verify the logic
        cutoff = date.fromordinal(date.today().toordinal() - 7)
        assert cutoff == date.today() - timedelta(days=7)

        # Test with a mock
        with patch("Data.storage.date") as mock_date:
            mock_date.today.return_value = date(2024, 1, 15)
            mock_date.fromordinal.side_effect = date.fromordinal
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            # Create old table manually
            conn = self.db._get_conn()
            old_table = "metrics_20240101"  # 14 days old
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {old_table} (
                    timestamp TEXT, timestamp_ns INTEGER, machine_id TEXT,
                    PRIMARY KEY (timestamp_ns, machine_id)
                )
            """)
            conn.commit()

            # Prune with 7 day retention
            with patch("Data.storage.RETENTION_DAYS", 7):
                dropped = self.db.prune_old_data()

            # Should have dropped the old table
            assert dropped >= 1

    def test_thread_safety(self):
        """Test concurrent inserts from multiple threads."""
        errors = []

        def worker(thread_id: int):
            try:
                db = MetricsDB(self.db_path)
                for i in range(10):
                    snap = SystemSnapshotV2(
                        timestamp=f"2024-01-01 12:00:{i:02d}",
                        timestamp_ns=1704110400000000000 + thread_id * 100 + i,
                        machine_id=f"thread_{thread_id}",
                        schema_version=2,
                        workload_tag="stress",
                        cpu_pct=float(thread_id * 10 + i),
                        ram_pct=60.0,
                        ram_used_mb=8192.0,
                        ram_available_mb=5376.0,
                        ram_total_mb=13568.0,
                        disk_pct=45.0,
                        disk_free_gb=500.0,
                        disk_total_gb=1000.0,
                        bandwidth_kbps=1024.0,
                        packet_rate_pps=500.0,
                        collector_cpu_pct=0.5,
                        collector_rss_mb=45.0,
                    )
                    db.insert(snap)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify all inserted
        conn = self.db._get_conn()
        today_table = daily_table_name(date.today().isoformat())
        count = conn.execute(f"SELECT COUNT(*) FROM {today_table}").fetchone()[0]
        assert count == 50  # 5 threads * 10 inserts

    def test_close(self):
        """Test closing database connections."""
        self.db.close()
        # Should not raise on double close
        self.db.close()

    def test_context_manager(self):
        """Test MetricsDB works as context manager."""
        with MetricsDB(self.db_path) as db:
            snap = SystemSnapshotV2(
                timestamp="2024-01-01 12:00:00",
                timestamp_ns=1704110400000000000,
                machine_id="ctx_test",
                schema_version=2,
                workload_tag="idle",
                cpu_pct=50.0,
                ram_pct=60.0,
                ram_used_mb=8192.0,
                ram_available_mb=5376.0,
                ram_total_mb=13568.0,
                disk_pct=45.0,
                disk_free_gb=500.0,
                disk_total_gb=1000.0,
                bandwidth_kbps=1024.0,
                packet_rate_pps=500.0,
                collector_cpu_pct=0.5,
                collector_rss_mb=45.0,
            )
            db.insert(snap)
        # Should be closed after exit


class TestGlobalFunctions:
    """Tests for global get_db/close_db functions."""

    def setup_method(self):
        close_db()

    def teardown_method(self):
        close_db()

    def test_get_db_returns_singleton(self):
        db1 = get_db()
        db2 = get_db()
        assert db1 is db2

    def test_close_db(self):
        db = get_db()
        close_db()


class TestAlertStorage:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db = MetricsDB(str(self.temp_dir / "alerts.db"))

    def teardown_method(self):
        self.db.close()
        for path in self.temp_dir.glob("*"):
            path.unlink(missing_ok=True)
        self.temp_dir.rmdir()

    def test_alert_schema_and_cooldown_grouping(self):
        first = Alert(
            module="module_b", severity="warning", title="Watched file modified",
            signals={"path": "C:/Users/test/notes.txt"},
        )
        repeated = Alert(
            module="module_b", severity="warning", title="Watched file modified",
            signals={"path": "C:/Users/test/notes.txt", "repeat": True},
        )

        self.db.record_alert(first, dedupe_key="same-file", cooldown_seconds=60)
        grouped = self.db.record_alert(repeated, dedupe_key="same-file", cooldown_seconds=60)

        assert grouped.id == first.id
        assert grouped.grouped_count == 2
        assert grouped.signals["latest_repeat"]["repeat"] is True
        rows = self.db.get_alerts()
        assert len(rows) == 1
        assert rows[0].module == "module_b"
        assert rows[0].severity == "warning"
        assert self.db.get_alerts(include_dismissed=False)[0].dismissed is False

    def test_dismiss_feedback_and_file_hash_observation(self):
        alert = Alert(
            module="module_b", severity="critical", title="Security log cleared",
            signals={"channel": "Security", "event_id": 1102},
        )
        self.db.record_alert(alert)
        assert self.db.mark_alert_normal(alert.id) is True
        assert self.db.dismiss_alert(alert.id) is True
        stored = self.db.get_alerts()[0]
        assert stored.dismissed is True
        assert stored.marked_normal is True
        assert self.db.get_alerts(include_dismissed=False) == []

        observation_id = self.db.record_file_observation(
            file_path="C:/Users/test/sample.exe", event_kind="created",
            sha256="a" * 64, file_size=123,
        )
        assert observation_id > 0
        observation = self.db.get_file_observations()[0]
        assert observation["sha256"] == "a" * 64
        assert observation["file_size"] == 123
        # Should not raise on double close
        close_db()


class TestSchemaCompatibility:
    """Test that schema matches INSERT_COLUMNS and SystemSnapshotV2 fields."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = str(self.temp_dir / "test_schema.db")
        self.db = MetricsDB(self.db_path)

    def teardown_method(self):
        self.db.close()
        import gc
        gc.collect()
        time.sleep(0.1)
        try:
            for f in self.temp_dir.glob("*"):
                f.unlink(missing_ok=True)
            self.temp_dir.rmdir()
        except Exception:
            pass

    def test_insert_columns_match_snapshot_fields(self):
        """Every INSERT_COLUMN should correspond to a SystemSnapshotV2 field."""
        snap = SystemSnapshotV2()
        row = snap.to_sqlite_row()

        for col in INSERT_COLUMNS:
            assert col in row, f"Column {col} not in snapshot row"

        # And vice versa (except derived fields with defaults)
        for key in row.keys():
            assert key in INSERT_COLUMNS, f"Row key {key} not in INSERT_COLUMNS"

    def test_insert_with_all_gpu_fields(self):
        """Test insert with all GPU fields populated."""
        snap = SystemSnapshotV2(
            timestamp="2024-01-01 12:00:00",
            timestamp_ns=1704110400000000000,
            machine_id="gpu_test",
            schema_version=2,
            workload_tag="gaming",
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
            top_cpu_processes=[ProcessInfo(pid=1, name="game.exe", cpu_percent=60.0)],
            top_gpu_processes=[ProcessInfo(pid=1, name="game.exe", cpu_percent=60.0, gpu_memory_mb=4096.0)],
            collector_cpu_pct=1.2,
            collector_rss_mb=60.0,
        )

        self.db.insert(snap)

        conn = self.db._get_conn()
        today_table = daily_table_name(date.today().isoformat())
        row = conn.execute(f"SELECT * FROM {today_table} WHERE machine_id = ?", ("gpu_test",)).fetchone()

        assert row["gpu_util_pct"] == 90.0
        assert row["gpu_temp_c"] == 75.0
        assert row["gpu_throttle_reasons"] == '["sw_power_cap", "hw_thermal"]'

        # Verify round-trip
        restored = SystemSnapshotV2.from_sqlite_row(dict(row))
        assert restored.gpu_util_pct == 90.0
        assert restored.gpu_throttle_reasons == ["sw_power_cap", "hw_thermal"]
        assert restored.top_gpu_processes[0].gpu_memory_mb == 4096.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
