"""Shared values exchanged by independent perception readers."""

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import numpy as np


T = TypeVar("T")


class PredictionStatus(str, Enum):
    """Whether a reader considers its own output safe to consume."""

    ACCEPTED = "accepted"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class ScreenKind(str, Enum):
    """Top-level screen states used to gate specialized readers."""

    GAMEPLAY = "gameplay"
    CARD_SELECTION = "card_selection"
    RETRY = "retry"
    TRANSITION = "transition"


@dataclass(frozen=True)
class Prediction(Generic[T]):
    """One reader result with provenance and an optional model confidence."""

    value: T | None
    status: PredictionStatus
    source: str
    confidence: float | None = None
    reason: str | None = None

    @classmethod
    def accepted(
        cls,
        value: T,
        *,
        source: str,
        confidence: float | None = None,
    ) -> "Prediction[T]":
        return cls(
            value=value,
            status=PredictionStatus.ACCEPTED,
            source=source,
            confidence=confidence,
        )

    @classmethod
    def unknown(cls, *, source: str, reason: str) -> "Prediction[T]":
        return cls(
            value=None,
            status=PredictionStatus.UNKNOWN,
            source=source,
            reason=reason,
        )

    @classmethod
    def rejected(
        cls,
        value: T | None,
        *,
        source: str,
        reason: str,
        confidence: float | None = None,
    ) -> "Prediction[T]":
        return cls(
            value=value,
            status=PredictionStatus.REJECTED,
            source=source,
            confidence=confidence,
            reason=reason,
        )

    @property
    def is_accepted(self) -> bool:
        return self.status is PredictionStatus.ACCEPTED and self.value is not None


@dataclass(frozen=True)
class AmmoReading:
    """Loaded and reserve ammunition read from the HUD."""

    loaded: int
    reserve: int


@dataclass(frozen=True)
class FusedPerception:
    """Validated perception state exposed to an action policy."""

    screen_kind: ScreenKind | None
    board: np.ndarray | None
    ammo: AmmoReading | None
    ready_for_policy: bool
    issues: tuple[str, ...]
    sources: tuple[str, ...]
