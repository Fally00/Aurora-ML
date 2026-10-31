"""Compute alert detection metrics from an isolated scenario-run folder."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def evaluate_scenarios(
    scenarios: list[dict[str, Any]],
    alert_rows: dict[str, dict[str, Any]],
    *,
    duration_seconds: float,
    observation_hours: float | None = None,
) -> dict[str, Any]:
    """Score expected alert-level detections and count unmatched alerts as false alarms."""
    true_positives = false_positives = false_negatives = 0
    referenced_ids: set[str] = set()
    scenario_results = []

    for scenario in scenarios:
        expected_count = max(int(scenario.get("expected_count", 1)), 0)
        expected_severity = scenario.get("expected_severity")
        title_prefix = scenario.get("expected_title_prefix", "")
        alert_ids = list(dict.fromkeys(scenario.get("alert_ids", [])))
        referenced_ids.update(alert_ids)
        matches = []
        for alert_id in alert_ids:
            alert = alert_rows.get(alert_id)
            if not alert:
                continue
            if (
                alert.get("module") == "module_b"
                and alert.get("severity") == expected_severity
                and str(alert.get("title", "")).startswith(title_prefix)
            ):
                matches.append(alert_id)

        tp = min(len(matches), expected_count)
        fp_ids = [alert_id for alert_id in alert_ids if alert_id not in matches]
        fp_ids.extend(matches[expected_count:])
        fn = max(expected_count - tp, 0)
        true_positives += tp
        false_positives += len(fp_ids)
        false_negatives += fn
        scenario_results.append({
            "scenario": scenario.get("scenario", "unknown"),
            "true_positives": tp,
            "false_positives": len(fp_ids),
            "false_negatives": fn,
            "false_positive_ids": fp_ids,
        })

    # An alert not attributed to a scenario is still an observed false alarm.
    orphan_ids = sorted(set(alert_rows) - referenced_ids)
    false_positives += len(orphan_ids)
    hours = observation_hours if observation_hours is not None else max(duration_seconds / 3600.0, 1 / 3600.0)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "observation_hours": hours,
        "false_alarms_per_hour": false_positives / hours if hours > 0 else 0.0,
        "orphan_false_positive_ids": orphan_ids,
        "scenarios": scenario_results,
    }


def evaluate_run(run_dir: Path, *, observation_hours: float | None = None) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    db_uri = (run_dir / "alerts.sqlite").resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT id, module, severity, title, grouped_count FROM alerts"
        ).fetchall()
    finally:
        connection.close()
    alert_rows = {
        row[0]: {
            "module": row[1],
            "severity": row[2],
            "title": row[3],
            "grouped_count": row[4],
        }
        for row in rows
    }
    return evaluate_scenarios(
        manifest.get("scenarios", []),
        alert_rows,
        duration_seconds=float(manifest.get("duration_seconds", 0.0)),
        observation_hours=observation_hours,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A directory emitted by scripts.run_scenarios")
    parser.add_argument("--observation-hours", type=float, default=None, help="Override run wall time for false alarms/hour, e.g. a longer monitored test interval.")
    args = parser.parse_args(argv)
    if args.observation_hours is not None and args.observation_hours <= 0:
        parser.error("--observation-hours must be positive")
    try:
        result = evaluate_run(args.run_dir, observation_hours=args.observation_hours)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"Could not evaluate scenario run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
