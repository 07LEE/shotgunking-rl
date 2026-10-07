"""Composable perception contracts for policy-ready game state."""

from .pipeline import PerceptionPipeline
from .readers import OverlayScreenReader, RedPixelAmmoReader, TemplatePieceReader
from .types import (
    AmmoReading,
    CombatStatsReading,
    FusedPerception,
    Prediction,
    PredictionStatus,
    ScreenKind,
)

__all__ = [
    "AmmoReading",
    "CombatStatsReading",
    "FusedPerception",
    "OverlayScreenReader",
    "PerceptionPipeline",
    "Prediction",
    "PredictionStatus",
    "RedPixelAmmoReader",
    "ScreenKind",
    "TemplatePieceReader",
]
