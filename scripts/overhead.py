"""Sample an Aurora process and compare its CPU/RSS against the resource budget."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import psutil


CPU_BUDGET_PERCENT = 1.0
RSS_BUDGET_MB = 150.0


def summarize_samples(samples: Iterable[dict[str, float]], *, cpu_budget: float = CPU_BUDGET_PERCENT, rss_budget_mb: float = RSS_BUDGET_MB) -> dict:
    rows = list(samples)
    if not rows:
        raise ValueError("at least one process sample is required")
    cpu = [float(row["cpu_percent"]) for row in rows]
    rss = [float(row["rss_mb"]) for row in rows]
    return {
        "sample_count": len(rows),
        "cpu_percent_mean": statistics.fmean(cpu),
        "cpu_percent_p95": _percentile(cpu, 0.95),
        "cpu_percent_max": max(cpu),
        "rss_mb_mean": statistics.fmean(rss),
        "rss_mb_max": max(rss),
        "cpu_budget_percent": cpu_budget,
        "rss_budget_mb": rss_budget_mb,
        "cpu_budget_pass": statistics.fmean(cpu) < cpu_budget,
        "rss_budget_pass": max(rss) < rss_budget_mb,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def sample_process(pid: int, duration_minutes: float, interval_seconds: float = 1.0) -> dict:
    if duration_minutes <= 0 or interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    process = psutil.Process(pid)
    if not process.is_running():
        raise psutil.NoSuchProcess(pid)
    logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
    duration = duration_minutes * 60.0
    process.cpu_percent(interval=None)  # Prime psutil's per-process CPU counter.
    samples: list[dict[str, float]] = []
    started = time.monotonic()
    while time.monotonic() - started < duration:
        remaining = duration - (time.monotonic() - started)
        wait = min(interval_seconds, remaining)
        if wait <= 0:
            break
        process_cpu = process.cpu_percent(interval=wait)
        rss_mb = process.memory_info().rss / (1024 * 1024)
        samples.append({
            "elapsed_seconds": time.monotonic() - started,
            "cpu_percent": process_cpu / logical_cpus,
            "rss_mb": rss_mb,
        })
    result = summarize_samples(samples)
    result["pid"] = pid
    result["duration_minutes"] = duration_minutes
    result["cpu_percent_basis"] = "process CPU normalized by logical CPU count; average is compared to the <1% total-system budget"
    result["passed"] = result["cpu_budget_pass"] and result["rss_budget_pass"]
    result["samples"] = samples
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="PID of Aurora.exe or the Aurora tray Python process.")
    parser.add_argument("--minutes", type=float, default=5.0, help="Sampling duration (default: 5 minutes).")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)
    try:
        result = sample_process(args.pid, args.minutes, args.interval)
    except (ValueError, psutil.Error) as exc:
        print(f"Overhead sampling failed: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)
    print(f"Budget: {'PASS' if result['passed'] else 'FAIL'} (mean CPU {result['cpu_percent_mean']:.3f}% / <1%, peak RSS {result['rss_mb_max']:.1f} MB / <150 MB)")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
