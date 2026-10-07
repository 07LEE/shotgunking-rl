"""Supported live-assistance scope for the first usable runtime milestone."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssistanceScope:
    """Configured play conditions and whether this runtime supports them."""

    rank: int
    weapon_type: int
    weapon_name: str
    floor: int
    supported: bool
    issues: tuple[str, ...]
    floor_verified: bool = False


def assess_assistance_scope(rank, weapon_type, weapon_name, floor, weapon_registered=True):
    """Accept registered weapons while treating rank and floor as configuration."""
    issues = []
    if type(rank) is not int or rank < 1:
        issues.append(f"invalid_rank:{rank}")
    if not weapon_registered:
        issues.append(f"unknown_weapon_type:{weapon_type}")
    if type(floor) is not int or floor < 1:
        issues.append(f"invalid_floor:{floor}")
    return AssistanceScope(
        rank=rank,
        weapon_type=weapon_type,
        weapon_name=weapon_name,
        floor=floor,
        supported=not issues,
        issues=tuple(issues),
    )


def format_assistance_scope(scope):
    """Return a concise operator-facing description of configured conditions."""
    status = "supported" if scope.supported else "unsupported"
    weapon_label = f"{scope.weapon_name} (weapon {scope.weapon_type})"
    details = (
        f"Rank {scope.rank} | {weapon_label} | Floor {scope.floor}"
    )
    floor_note = "Floor is recorded only and does not restrict support; confirm it on the game screen."
    rank_note = " Rank scaling uses configuration and is not verified from the game screen."
    effect_note = " Card and buff effects must match runtime configuration; live verification is unavailable."
    issue_note = "" if scope.supported else f" Unsupported: {', '.join(scope.issues)}."
    return f"Play scope [{status}]: {details}\n{floor_note}{rank_note}{effect_note}{issue_note}"
