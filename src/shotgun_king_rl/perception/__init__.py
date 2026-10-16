"""Composable perception contracts for policy-ready game state."""

from .attack import AttackReader
from .knockback import KnockbackReader
from .pipeline import PerceptionPipeline
from .range import RangeReader
from .spread import SpreadReader
from .readers import OverlayScreenReader, RedPixelAmmoReader, TemplatePieceReader
from .types import (
    AmmoReading,
    CombatStatsReading,
    FusedPerception,
    Prediction,
    PredictionStatus,
    RangeReading,
    ScreenKind,
)

__all__ = [
    "AttackReader",
    "AmmoReading",
    "CombatStatsReading",
    "FusedPerception",
    "KnockbackReader",
    "OverlayScreenReader",
    "PerceptionPipeline",
    "Prediction",
    "PredictionStatus",
    "RangeReader",
    "RangeReading",
    "RedPixelAmmoReader",
    "ScreenKind",
    "SpreadReader",
    "TemplatePieceReader",
]
