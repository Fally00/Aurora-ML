"""Passive Windows Event Log subscriptions for log-clear events."""

from __future__ import annotations

import ctypes
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Optional


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LogClearEvent:
    channel: str
    event_id: int


class WindowsEventLogListener:
    """Subscribe to Security 1102 and System 104 without polling the log."""

    SUBSCRIPTIONS = (("Security", 1102), ("System", 104))

    def __init__(
        self,
        emit: Callable[[LogClearEvent], bool],
        *,
        platform_name: Optional[str] = None,
        is_admin: Optional[Callable[[], bool]] = None,
        eventlog_api=None,
        event_api=None,
    ):
        self._emit = emit
        self._platform_name = platform_name or os.name
        self._is_admin = is_admin or _is_elevated
        self._eventlog = eventlog_api
        self._event = event_api
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._warned = False
        self._warn_lock = threading.Lock()

    def _warn_once(self, message: str, *args) -> None:
        with self._warn_lock:
            if not self._warned:
                LOGGER.warning(message, *args)
                self._warned = True

    def start(self) -> bool:
        """Start both subscriptions, or degrade once when unavailable."""
        if self._platform_name != "nt":
            self._warn_once("Windows Event Log monitoring is available on Windows only")
            return False
        try:
            elevated = bool(self._is_admin())
        except Exception:
            elevated = False
        if not elevated:
            self._warn_once("Windows Event Log monitoring needs administrator rights; continuing without it")
            return False

        try:
            if self._eventlog is None:
                import win32evtlog
                self._eventlog = win32evtlog
            if self._event is None:
                import win32event
                self._event = win32event
        except ImportError as exc:
            self._warn_once("pywin32 Event Log APIs unavailable: %s", exc)
            return False

        self._stop.clear()
        for channel, event_id in self.SUBSCRIPTIONS:
            thread = threading.Thread(
                target=self._subscription_loop,
                args=(channel, event_id),
                name=f"aurora-eventlog-{channel.lower()}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        return True

    def _subscription_loop(self, channel: str, event_id: int) -> None:
        eventlog = self._eventlog
        event_api = self._event
        signal = None
        subscription = None
        query = (
            f"*[System[Provider[@Name='Microsoft-Windows-Eventlog'] "
            f"and EventID={event_id}]]"
        )
        try:
            signal = event_api.CreateEvent(None, 0, 0, None)
            subscription = eventlog.EvtSubscribe(
                channel,
                eventlog.EvtSubscribeToFutureEvents,
                SignalEvent=signal,
                Query=query,
            )
            while not self._stop.is_set():
                status = event_api.WaitForSingleObject(signal, 1000)
                if status != event_api.WAIT_OBJECT_0:
                    continue
                events = eventlog.EvtNext(subscription, 32, Timeout=0)
                event = None
                for event in events:
                    xml = eventlog.EvtRender(event, eventlog.EvtRenderEventXml)
                    if isinstance(xml, bytes):
                        xml = xml.decode("utf-8", errors="replace")
                    match = re.search(r"<EventID(?:\s+[^>]*)?>(\d+)</EventID>", xml)
                    if match and int(match.group(1)) == event_id:
                        self._emit(LogClearEvent(channel=channel, event_id=event_id))
                del event
                del events
        except Exception as exc:
            self._warn_once("Could not subscribe to the %s event log: %s", channel, exc)
        finally:
            # pywin32 EVT handles close when their wrapper objects are released.
            subscription = None
            signal = None

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()


def _is_elevated() -> bool:
    if os.name != "nt":
        return False
    return bool(ctypes.windll.shell32.IsUserAnAdmin())
