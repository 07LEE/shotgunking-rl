"""Validation shared by combat aggregation and policy gating."""

from numbers import Real

import numpy as np

from .types import CombatStatsReading


def valid_combat_stats(stats: CombatStatsReading | None) -> bool:
    if stats is None:
        return False
    required = (
        stats.attack,
        stats.range_min,
        stats.range_max,
        stats.spread_degrees,
        stats.penetration_percent,
        stats.knockback_percent,
        stats.grenades_remaining,
        stats.grenade_damage,
        stats.melee_damage,
    )
    if not all(_is_finite_nonnegative(value) for value in required):
        return False
    if not _is_nonnegative_integer(stats.grenades_remaining):
        return False
    return (
        stats.range_min <= stats.range_max
        and stats.spread_degrees <= 360
        and stats.penetration_percent <= 100
        and stats.knockback_percent <= 100
    )


def _is_finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
        and value >= 0
    )


def _is_nonnegative_integer(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= 0
    )
