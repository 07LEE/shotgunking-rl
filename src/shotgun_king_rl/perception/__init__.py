"""Composable perception contracts for policy-ready game state."""

from .pipeline import PerceptionPipeline
from .readers import OverlayScreenReader, RedPixelAmmoReader, TemplatePieceReader
from .types import (
    AmmoReading,
    FusedPerception,
    Prediction,
    PredictionStatus,
    ScreenKind,
)

__all__ = [
    "AmmoReading",
    "FusedPerception",
    "OverlayScreenReader",
    "PerceptionPipeline",
    "Prediction",
    "PredictionStatus",
    "RedPixelAmmoReader",
    "ScreenKind",
    "TemplatePieceReader",
]
