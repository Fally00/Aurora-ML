"""Portable tests for Windows event-log subscription behavior."""

from Data.eventlog_monitor import LogClearEvent, WindowsEventLogListener


def test_event_log_listener_degrades_once_without_admin(caplog):
    listener = WindowsEventLogListener(
        lambda event: True,
        platform_name="nt",
        is_admin=lambda: False,
    )

    assert listener.start() is False
    assert listener.start() is False
    assert sum("administrator rights" in record.message for record in caplog.records) == 1


def test_event_log_listener_degrades_off_windows(caplog):
    listener = WindowsEventLogListener(lambda event: True, platform_name="posix")

    assert listener.start() is False
    assert any("available on Windows only" in record.message for record in caplog.records)


def test_pull_subscription_emits_matching_clear_event():
    emitted = []
    listener = WindowsEventLogListener(
        emitted.append, platform_name="nt", is_admin=lambda: True
    )

    class EventApi:
        WAIT_OBJECT_0 = 0

        @staticmethod
        def CreateEvent(*args):
            return object()

        @staticmethod
        def WaitForSingleObject(*args):
            return 0

    class EventLogApi:
        EvtSubscribeToFutureEvents = 1
        EvtRenderEventXml = 2

        @staticmethod
        def EvtSubscribe(*args, **kwargs):
            return object()

        def EvtNext(self, *args, **kwargs):
            listener._stop.set()
            return (object(),)

        @staticmethod
        def EvtRender(*args):
            return "<Event><System><EventID>1102</EventID></System></Event>"

    listener._event = EventApi()
    listener._eventlog = EventLogApi()
    listener._subscription_loop("Security", 1102)

    assert emitted == [LogClearEvent(channel="Security", event_id=1102)]
