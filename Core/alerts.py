"""Shared alert record used by monitoring modules and the tray UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


Severity = Literal["info", "warning", "critical"]


@dataclass(slots=True)
class Alert:
    """A passive notification event persisted in the local SQLite database."""

    module: str
    severity: Severity
    title: str
    signals: dict[str, Any] = field(default_factory=dict)
    grouped_count: int = 1
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    dismissed: bool = False
    marked_normal: bool = False

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "critical"}:
            raise ValueError(f"Unsupported alert severity: {self.severity}")
        if not self.module.strip() or not self.title.strip():
            raise ValueError("Alert module and title are required")
        if self.grouped_count < 1:
            raise ValueError("grouped_count must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Return the public alert schema for APIs and UI consumers."""
        return asdict(self)
