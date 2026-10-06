"""Structured results for desktop capture and input operations."""

from dataclasses import dataclass
from enum import Enum


class IOStatus(str, Enum):
    """Outcome of one desktop operation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CaptureResult:
    """Result of capturing one explicitly selected screen region."""

    status: IOStatus
    output_path: str
    reason: str | None = None
    geometry: dict[str, int] | None = None

    @property
    def succeeded(self):
        return self.status is IOStatus.SUCCEEDED

    def __bool__(self):
        return self.succeeded

    def as_info(self):
        return {
            "status": self.status.value,
            "output_path": self.output_path,
            "reason": self.reason,
            "geometry": self.geometry,
        }


@dataclass(frozen=True)
class InputResult:
    """Result of one mouse or keyboard operation."""

    status: IOStatus
    operation: str
    reason: str | None = None
    target: tuple[int, int] | str | None = None

    @property
    def succeeded(self):
        return self.status is IOStatus.SUCCEEDED

    def __bool__(self):
        return self.succeeded

    def as_info(self):
        return {
            "status": self.status.value,
            "operation": self.operation,
            "reason": self.reason,
            "target": self.target,
        }
