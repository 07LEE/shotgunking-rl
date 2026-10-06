"""Supported live-assistance scope for the first usable runtime milestone."""

from dataclasses import dataclass


SUPPORTED_RANK = 1
SUPPORTED_WEAPON_TYPE = 0
SUPPORTED_FLOOR = 1


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


def assess_assistance_scope(rank, weapon_type, weapon_name, floor):
    """Accept only the rank-one, basic-weapon, first-floor milestone."""
    issues = []
    if rank != SUPPORTED_RANK:
        issues.append(f"rank:{rank}")
    if weapon_type != SUPPORTED_WEAPON_TYPE:
        issues.append(f"weapon_type:{weapon_type}")
    if floor != SUPPORTED_FLOOR:
        issues.append(f"floor:{floor}")
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
    weapon_label = (
        f"{scope.weapon_name} (basic, weapon {scope.weapon_type})"
        if scope.weapon_type == SUPPORTED_WEAPON_TYPE
        else f"{scope.weapon_name} (weapon {scope.weapon_type})"
    )
    details = (
        f"Rank {scope.rank} | {weapon_label} | Floor {scope.floor}"
    )
    floor_note = "Floor is configured only; confirm it on the game screen."
    issue_note = "" if scope.supported else f" Unsupported: {', '.join(scope.issues)}."
    return f"Play scope [{status}]: {details}\n{floor_note}{issue_note}"
