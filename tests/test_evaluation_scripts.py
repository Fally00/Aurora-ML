import pytest
import json
import sqlite3

from scripts.alert_metrics import evaluate_run, evaluate_scenarios
from scripts.overhead import summarize_samples
from scripts.run_scenarios import validate_scenario_request


def test_scenario_evaluation_computes_precision_recall_f1_and_false_alarms_rate():
    scenarios = [
        {
            "scenario": "expected-warning",
            "expected_count": 1,
            "expected_severity": "warning",
            "expected_title_prefix": "Watched file modified",
            "alert_ids": ["a1", "a2"],
        },
        {
            "scenario": "expected-critical",
            "expected_count": 1,
            "expected_severity": "critical",
            "expected_title_prefix": "Windows Security event log cleared",
            "alert_ids": ["a3"],
        },
    ]
    alerts = {
        "a1": {"module": "module_b", "severity": "warning", "title": "Watched file modified"},
        "a2": {"module": "module_b", "severity": "info", "title": "Unexpected extra alert"},
        "a3": {"module": "module_b", "severity": "critical", "title": "Windows Security event log cleared"},
        "a4": {"module": "module_b", "severity": "warning", "title": "Unattributed alert"},
    }

    result = evaluate_scenarios(scenarios, alerts, duration_seconds=1800)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 0
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["false_alarms_per_hour"] == pytest.approx(4.0)


def test_scenario_evaluation_counts_missing_expected_alerts_as_false_negatives():
    result = evaluate_scenarios(
        [{"scenario": "missed", "expected_count": 1, "expected_severity": "warning", "expected_title_prefix": "File", "alert_ids": []}],
        {},
        duration_seconds=60,
    )

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 1
    assert result["recall"] == 0.0


def test_evaluate_run_reads_the_run_database_and_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "duration_seconds": 3600,
        "scenarios": [{
            "scenario": "edit",
            "expected_count": 1,
            "expected_severity": "warning",
            "expected_title_prefix": "Watched file modified",
            "alert_ids": ["alert-1"],
        }],
    }), encoding="utf-8")
    connection = sqlite3.connect(run_dir / "alerts.sqlite")
    connection.execute("CREATE TABLE alerts (id TEXT, module TEXT, severity TEXT, title TEXT, grouped_count INTEGER)")
    connection.execute("INSERT INTO alerts VALUES (?, ?, ?, ?, ?)", ("alert-1", "module_b", "warning", "Watched file modified", 1))
    connection.commit()
    connection.close()

    result = evaluate_run(run_dir)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["precision"] == 1.0


def test_scenario_runner_requires_test_vm_flag_for_log_clear():
    with pytest.raises(PermissionError, match="--i-am-on-a-test-vm"):
        validate_scenario_request(["log-clear"], is_test_vm=False)

    validate_scenario_request(["watched-edit", "bulk-copy"], is_test_vm=False)
    validate_scenario_request(["log-clear"], is_test_vm=True)


def test_overhead_summary_compares_average_cpu_and_peak_rss_to_budgets():
    result = summarize_samples([
        {"cpu_percent": 0.4, "rss_mb": 80},
        {"cpu_percent": 1.2, "rss_mb": 160},
    ])

    assert result["cpu_percent_mean"] == pytest.approx(0.8)
    assert result["cpu_budget_pass"] is True
    assert result["rss_budget_pass"] is False
