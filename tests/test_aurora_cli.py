from __future__ import annotations

import threading
from unittest.mock import Mock, patch

import aurora_cli


def test_run_collector_passes_stop_event_and_selected_verbosity():
    for quiet in (False, True):
        stop_event = threading.Event()
        failure_event = threading.Event()
        with patch("aurora_cli.run_collector") as collector:
            aurora_cli._run_collector(stop_event, quiet, failure_event)

        collector.assert_called_once_with(verbose=not quiet, stop_event=stop_event)
        assert not failure_event.is_set()


def test_run_stops_and_cleans_up_when_collector_fails():
    storage = Mock()
    storage.get_alerts.return_value = []
    monitor = Mock()
    collector = Mock(side_effect=RuntimeError("collector failed"))

    with (
        patch("aurora_cli.get_db", return_value=storage),
        patch("aurora_cli.FileMonitor", return_value=monitor),
        patch("aurora_cli.run_collector", collector),
    ):
        assert aurora_cli.run(quiet=True) == 1

    storage.close.assert_called_once_with()
    monitor.stop.assert_called_once_with()
    assert collector.call_args.kwargs["stop_event"].is_set()
