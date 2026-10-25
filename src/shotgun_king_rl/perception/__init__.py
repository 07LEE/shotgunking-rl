"""Composable perception contracts for policy-ready game state."""

from .combat import CombatStatsReader
from .attack import AttackReader
from .grenade_count import GrenadeCountReader
from .grenade_damage import GrenadeDamageReader
from .knockback import KnockbackReader
from .melee import MeleeDamageReader
from .penetration import PenetrationReader
from .pipeline import PerceptionPipeline
from .range import RangeReader
from .spread import SpreadReader
from .readers import OverlayScreenReader, RedPixelAmmoReader, TemplatePieceReader
from .types import (
    AmmoReading,
    BoardContextReading,
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
    "BoardContextReading",
    "CombatStatsReading",
    "CombatStatsReader",
    "FusedPerception",
    "GrenadeCountReader",
    "GrenadeDamageReader",
    "KnockbackReader",
    "MeleeDamageReader",
    "PenetrationReader",
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
