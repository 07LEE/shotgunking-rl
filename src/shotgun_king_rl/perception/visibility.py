"""Keep image visibility labels separate from remembered game-state values."""

COMBAT_FIELDS = ('attack', 'range_min', 'range_max', 'spread_degrees', 'penetration_percent',
                 'knockback_percent', 'grenades_remaining', 'grenade_damage', 'melee_damage')
VISIBILITY_STATES = ('visible', 'occluded', 'not_displayed', 'unreadable', 'unknown')


def combat_visibility(state):
    labels = state.get('combat_visibility', {})
    if (not isinstance(labels, dict) or any(key not in COMBAT_FIELDS for key in labels)
            or any(value not in VISIBILITY_STATES for value in labels.values())):
        raise ValueError('Invalid combat visibility labels')
    if any(status == 'visible' and (type((state.get('combat_stats') or {}).get(field)) is not int
                                  or (state.get('combat_stats') or {})[field] < 0)
           for field, status in labels.items()):
        raise ValueError('Visible combat fields require a numeric value; use unreadable otherwise')
    return dict(labels)


def visual_combat_stats(state, *, legacy=True):
    """Explicit non-visible fields never supply numeric image targets.

    Existing readers retain their legacy behavior for unannotated fields.
    New visual dataset targets require an explicit visible label.
    """
    labels = combat_visibility(state)
    values = state.get('combat_stats') or {}
    return {field: values.get(field) if labels.get(field) == 'visible'
            or (legacy and field not in labels) else None for field in COMBAT_FIELDS}


def combat_visual_targets(state):
    labels = combat_visibility(state)
    values = visual_combat_stats(state, legacy=False)
    reviewed = (state.get('reviews') or {}).get('combat_stats') is True
    return {field: {'visibility': labels.get(field, 'unknown'),
                    'value': values[field] if reviewed else None,
                    'reviewed': reviewed and field in labels}
            for field in COMBAT_FIELDS}
