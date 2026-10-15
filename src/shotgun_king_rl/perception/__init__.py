"""Composable perception contracts for policy-ready game state."""

from .attack import AttackReader
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
    "AttackReader",
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
