"""
Aurora-ML SQLite Storage Layer (WAL mode, daily rotation, schema versioning).
"""

from __future__ import annotations

import re
import json
import sqlite3
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Iterator

from config import (
    DB_PATH,
    RETENTION_DAYS,
    SCHEMA_VERSION,
    MACHINE_ID,
)
from Core.schema_v2 import (
    SystemSnapshotV2,
    SCHEMA_V2_SQL,
    INSERT_SQL,
    INSERT_COLUMNS,
    daily_table_name,
)
from Core.alerts import Alert


ALERT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    module TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    title TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    grouped_count INTEGER NOT NULL DEFAULT 1 CHECK (grouped_count >= 1),
    dismissed INTEGER NOT NULL DEFAULT 0,
    marked_normal INTEGER NOT NULL DEFAULT 0,
    dedupe_key TEXT,
    last_seen_epoch REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, dismissed);
CREATE INDEX IF NOT EXISTS idx_alerts_dedupe ON alerts(dedupe_key, last_seen_epoch);

CREATE TABLE IF NOT EXISTS file_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    file_path TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    sha256 TEXT,
    file_size INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_file_observations_path_time
    ON file_observations(file_path, observed_at);
"""


class MetricsDB:
    """Thread-safe SQLite database with WAL mode and daily table rotation."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._current_day: Optional[str] = None
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self) -> None:
        Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize migration metadata; daily metric tables are created on demand."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA_V2_SQL)
        # Record initial migration if not exists
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, datetime.now().isoformat(), "Initial v2 schema"),
        )
        conn.executescript(ALERT_SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            (3, datetime.now().isoformat(), "Module B alert and file observation schema"),
        )
        conn.commit()
        conn.close()

    def _get_today_table(self) -> str:
        """Get today's daily table name, creating it if needed."""
        today = date.today().isoformat()
        if self._current_day != today:
            self._current_day = today
            table = daily_table_name(today)
            conn = self._get_conn()
            # A new table is created at day rollover, keeping each table bounded.
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    timestamp TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    machine_id TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    workload_tag TEXT NOT NULL,

                    cpu_pct REAL NOT NULL,
                    cpu_per_core_pct TEXT,
                    cpu_package_temp_c REAL,
                    cpu_core_clocks_mhz TEXT,

                    ram_pct REAL NOT NULL,
                    ram_used_mb REAL NOT NULL,
                    ram_available_mb REAL NOT NULL,
                    ram_total_mb REAL NOT NULL,

                    disk_pct REAL NOT NULL,
                    disk_free_gb REAL NOT NULL,
                    disk_total_gb REAL NOT NULL,

                    bandwidth_kbps REAL NOT NULL,
                    packet_rate_pps REAL NOT NULL,

                    gpu_util_pct REAL,
                    gpu_mem_used_mb REAL,
                    gpu_mem_total_mb REAL,
                    gpu_mem_pct REAL,
                    gpu_temp_c REAL,
                    gpu_power_w REAL,
                    gpu_power_limit_w REAL,
                    gpu_core_clock_mhz REAL,
                    gpu_mem_clock_mhz REAL,
                    gpu_fan_pct REAL,
                    gpu_throttle_reasons TEXT,

                    top_cpu_processes TEXT,
                    top_gpu_processes TEXT,

                    collector_cpu_pct REAL NOT NULL,
                    collector_rss_mb REAL NOT NULL,

                    cpu_delta REAL NOT NULL,
                    ram_delta REAL NOT NULL,
                    cpu_rolling_mean REAL NOT NULL,
                    ram_rolling_mean REAL NOT NULL,
                    cpu_rolling_std REAL NOT NULL,
                    ram_rolling_std REAL NOT NULL,
                    bw_rolling_mean REAL NOT NULL,
                    bw_spike REAL NOT NULL,

                    PRIMARY KEY (timestamp_ns, machine_id)
                );
            """)
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_timestamp ON {table}(timestamp);")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_machine ON {table}(machine_id);")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_workload ON {table}(workload_tag);")
            conn.commit()
        return daily_table_name(today)

    def insert(self, snapshot: SystemSnapshotV2) -> None:
        """Insert a snapshot into today's daily table."""
        table = self._get_today_table()
        row = snapshot.to_sqlite_row()
        values = [row[col] for col in INSERT_COLUMNS]
        conn = self._get_conn()
        conn.execute(INSERT_SQL.format(table=table), values)
        conn.commit()

    def insert_batch(self, snapshots: List[SystemSnapshotV2]) -> None:
        """Insert multiple snapshots in a single transaction."""
        if not snapshots:
            return
        table = self._get_today_table()
        # Use table-specific INSERT
        sql = INSERT_SQL.format(table=table)
        conn = self._get_conn()
        conn.executemany(sql, [[s.to_sqlite_row()[col] for col in INSERT_COLUMNS] for s in snapshots])
        conn.commit()

    def query(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        machine_id: Optional[str] = None,
        workload_tag: Optional[str] = None,
        limit: int = 1000,
    ) -> List[SystemSnapshotV2]:
        """Query snapshots across daily tables."""
        conn = self._get_conn()
        conditions = []
        params = []

        if start:
            conditions.append("timestamp >= ?")
            params.append(start.strftime("%Y-%m-%d %H:%M:%S"))
        if end:
            conditions.append("timestamp <= ?")
            params.append(end.strftime("%Y-%m-%d %H:%M:%S"))
        if machine_id:
            conditions.append("machine_id = ?")
            params.append(machine_id)
        if workload_tag:
            conditions.append("workload_tag = ?")
            params.append(workload_tag)

        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'metrics_[0-9]*'"
            )
            if re.fullmatch(r"metrics_\d{8}", row[0])
        ]
        if not tables or limit <= 0:
            return []

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        union = " UNION ALL ".join(
            f'SELECT * FROM "{table}" {where}' for table in tables
        )
        sql = f"SELECT * FROM ({union}) ORDER BY timestamp_ns DESC LIMIT ?"
        rows = conn.execute(sql, params * len(tables) + [limit]).fetchall()
        return [SystemSnapshotV2.from_sqlite_row(dict(r)) for r in rows]

    def get_latest(self, machine_id: str = MACHINE_ID) -> Optional[SystemSnapshotV2]:
        """Get most recent snapshot for a machine."""
        results = self.query(machine_id=machine_id, limit=1)
        return results[0] if results else None

    def count_today(self, machine_id: str = MACHINE_ID) -> int:
        """Count rows in today's table for a machine."""
        table = self._get_today_table()
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE machine_id = ?",
            (machine_id,),
        ).fetchone()
        return row[0] if row else 0

    def record_alert(
        self,
        alert: Alert,
        *,
        dedupe_key: Optional[str] = None,
        cooldown_seconds: float = 0.0,
    ) -> Alert:
        """Persist an alert, grouping matching repeats during its cooldown."""
        conn = self._get_conn()
        now = time.time()
        if dedupe_key and cooldown_seconds > 0:
            previous = conn.execute(
                """SELECT id, timestamp, module, severity, title, signals_json, grouped_count
                   FROM alerts
                   WHERE dedupe_key = ? AND dismissed = 0 AND last_seen_epoch >= ?
                   ORDER BY last_seen_epoch DESC LIMIT 1""",
                (dedupe_key, now - cooldown_seconds),
            ).fetchone()
            if previous:
                signals = json.loads(previous["signals_json"])
                signals["latest_repeat"] = alert.signals
                grouped_count = previous["grouped_count"] + alert.grouped_count
                conn.execute(
                    """UPDATE alerts SET signals_json = ?, grouped_count = ?, last_seen_epoch = ?
                       WHERE id = ?""",
                    (json.dumps(signals, ensure_ascii=False), grouped_count, now, previous["id"]),
                )
                conn.commit()
                return Alert(
                    id=previous["id"],
                    timestamp=previous["timestamp"],
                    module=previous["module"],
                    severity=previous["severity"],
                    title=previous["title"],
                    signals=signals,
                    grouped_count=grouped_count,
                )

        conn.execute(
            """INSERT INTO alerts
               (id, timestamp, module, severity, title, signals_json, grouped_count,
                dismissed, marked_normal, dedupe_key, last_seen_epoch)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)""",
            (
                alert.id,
                alert.timestamp,
                alert.module,
                alert.severity,
                alert.title,
                json.dumps(alert.signals, ensure_ascii=False),
                alert.grouped_count,
                dedupe_key,
                now,
            ),
        )
        conn.commit()
        return alert

    def get_alerts(self, limit: int = 500, *, include_dismissed: bool = True) -> list[Alert]:
        """Read recent alerts in reverse chronological order."""
        if limit <= 0:
            return []
        where = "" if include_dismissed else "WHERE dismissed = 0"
        rows = self._get_conn().execute(
            f"SELECT * FROM alerts {where} ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Alert(
                id=row["id"], timestamp=row["timestamp"], module=row["module"],
                severity=row["severity"], title=row["title"],
                signals=json.loads(row["signals_json"]), grouped_count=row["grouped_count"],
                dismissed=bool(row["dismissed"]), marked_normal=bool(row["marked_normal"]),
            )
            for row in rows
        ]

    def dismiss_alert(self, alert_id: str) -> bool:
        """Mark an alert dismissed by its tray notification or log view."""
        cursor = self._get_conn().execute(
            "UPDATE alerts SET dismissed = 1 WHERE id = ?", (alert_id,)
        )
        self._get_conn().commit()
        return cursor.rowcount > 0

    def mark_alert_normal(self, alert_id: str) -> bool:
        """Store user feedback that an alert was expected/normal."""
        cursor = self._get_conn().execute(
            "UPDATE alerts SET marked_normal = 1 WHERE id = ?", (alert_id,)
        )
        self._get_conn().commit()
        return cursor.rowcount > 0

    def record_file_observation(
        self,
        *,
        file_path: str,
        event_kind: str,
        sha256: Optional[str],
        file_size: Optional[int],
        error: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> int:
        """Record each executable/critical file observation and its digest."""
        cursor = self._get_conn().execute(
            """INSERT INTO file_observations
               (observed_at, file_path, event_kind, sha256, file_size, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (observed_at or datetime.now().astimezone().isoformat(), file_path,
             event_kind, sha256, file_size, error),
        )
        self._get_conn().commit()
        return int(cursor.lastrowid)

    def get_file_observations(self, limit: int = 500) -> list[dict]:
        """Return recent file hashes for audit and evaluation tools."""
        rows = self._get_conn().execute(
            "SELECT * FROM file_observations ORDER BY id DESC LIMIT ?", (max(limit, 0),)
        ).fetchall()
        return [dict(row) for row in rows]

    def prune_old_data(self) -> int:
        """Drop daily tables older than RETENTION_DAYS. Returns number of tables dropped."""
        cutoff = date.fromordinal(date.today().toordinal() - RETENTION_DAYS)
        conn = self._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'metrics_%'"
        ).fetchall()
        dropped = 0
        for (table,) in tables:
            try:
                if not re.fullmatch(r"metrics_\d{8}", table):
                    continue
                table_date_str = table.removeprefix("metrics_")
                table_date = date(int(table_date_str[:4]), int(table_date_str[4:6]), int(table_date_str[6:8]))
                if table_date < cutoff:
                    conn.execute(f"DROP TABLE {table}")
                    dropped += 1
            except (ValueError, IndexError):
                continue
        conn.commit()
        return dropped

    def close(self) -> None:
        """Close all connections."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def flush(self) -> None:
        """Commit pending work on this thread's connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.commit()

    def __enter__(self) -> "MetricsDB":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# Global instance
_db: Optional[MetricsDB] = None


def get_db() -> MetricsDB:
    """Get global database instance."""
    global _db
    if _db is None:
        _db = MetricsDB()
    return _db


def close_db() -> None:
    """Close global database."""
    global _db
    if _db:
        _db.close()
        _db = None
