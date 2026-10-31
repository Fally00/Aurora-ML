"""Small interfaces for future workload and resource-abuse models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from Core.schema_v2 import SystemSnapshotV2


@dataclass(frozen=True, slots=True)
class WorkloadState:
    label: str = "unknown"
    source: str = "classifier"
    trained: bool = False
    status: str = "not trained yet"


class WorkloadClassifier(Protocol):
    def predict(self, snapshot: SystemSnapshotV2 | None) -> WorkloadState: ...


class UntrainedWorkloadClassifier:
    """Placeholder contract; deliberately performs no inference."""

    def predict(self, snapshot: SystemSnapshotV2 | None = None) -> WorkloadState:
        return WorkloadState()


class ResourceAbuseDetector(Protocol):
    @property
    def status(self) -> str: ...

    def detect(self, snapshot: SystemSnapshotV2) -> list[object]: ...


class UntrainedResourceAbuseDetector:
    """Module A placeholder; it does not generate alerts or take action."""

    @property
    def status(self) -> str:
        return "not trained yet"

    def detect(self, snapshot: SystemSnapshotV2) -> list[object]:
        return []
