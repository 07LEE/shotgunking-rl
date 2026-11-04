"""Choose settled captures that fill recognition gaps, never approve them."""

from collections import Counter
import json
from pathlib import Path

from ..perception.frame import PreparedFrame, read_frame
from ..perception.readers import OverlayScreenReader
from ..perception.types import ScreenKind
from ..perception.visibility import visual_combat_stats
from ..perception.visibility_proposals import (
    combat_field_readers, combat_visibility_titles, propose_combat_visibility,
)

PRIMARY_FIELDS = ('attack', 'range_min', 'range_max', 'spread_degrees')


def _value_key(field, value):
    # Readers may return 2.0 where reviewed labels store 2.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f'{field}={value}'


def reviewed_coverage(collection):
    """Count reviewed train values; only train annotations become references."""
    coverage = Counter()
    for path in sorted((Path(collection) / 'sessions').glob('*/annotations/*.json')):
        data = json.loads(path.read_text())
        state = data.get('screen_state') or {}
        if (data.get('split') != 'train' or state.get('screen_kind') != 'gameplay'
                or (state.get('reviews') or {}).get('combat_stats') is not True):
            continue
        for field, value in visual_combat_stats(state).items():
            if value is not None:
                coverage[_value_key(field, value)] += 1
    return coverage


def pending_reasons(collection):
    """Count reasons already waiting in unreviewed automatic drafts."""
    pending = Counter()
    for path in sorted((Path(collection) / 'sessions').glob('*/annotations/*.json')):
        data = json.loads(path.read_text())
        reviews = (data.get('screen_state') or {}).get('reviews') or {}
        if any(reviews.values()):
            continue
        pending.update((data.get('auto_capture') or {}).get('reasons') or ())
    return pending


class GapSelector:
    """Name why a settled screen is worth review; an empty list means skip it.

    Reasons cover unknown screens, recognized titles with unreadable values,
    unrecognized primary rows, unread grenade slots and rarely reviewed values.
    Each reason stops being selected once enough drafts await review.
    """

    method = 'recognition-gaps-v1'

    def __init__(self, screen_reader, readers, grenade_count_reader, titles, coverage, *,
                 pending=(), min_examples=3, max_per_reason=5):
        if type(min_examples) is not int or min_examples < 1:
            raise ValueError('min_examples must be positive')
        if type(max_per_reason) is not int or max_per_reason < 1:
            raise ValueError('max_per_reason must be positive')
        self.screen_reader = screen_reader
        self.readers = dict(readers)
        self.grenade_count_reader = grenade_count_reader
        self.titles = list(titles)
        self.coverage = Counter(coverage)
        self.selected = Counter(pending)
        self.min_examples = min_examples
        self.max_per_reason = max_per_reason

    @classmethod
    def from_collection(cls, collection, **options):
        from .combat_evaluation import build_combat_reader

        combat, _ = build_combat_reader(collection)
        readers = combat_field_readers(combat)
        return cls(OverlayScreenReader(), readers, combat.grenade_count_reader,
                   combat_visibility_titles(readers), reviewed_coverage(collection),
                   pending=pending_reasons(collection), **options)

    def gaps(self, screen):
        """Return every gap in one normalized screen, before applying caps."""
        frame = PreparedFrame(screen, normalized=True)
        kind = read_frame(self.screen_reader, frame)
        if not kind.is_accepted:
            return ['screen:unknown']
        if kind.value is not ScreenKind.GAMEPLAY:
            return []
        values = {}
        for field, reader in self.readers.items():
            if reader is None:
                continue
            result = read_frame(reader, frame)
            if not result.is_accepted:
                continue
            if field == 'range':
                values.update(range_min=result.value.range_min, range_max=result.value.range_max)
            else:
                values[field] = result.value
        count = read_frame(self.grenade_count_reader, frame) if self.grenade_count_reader else None
        if count is not None and count.is_accepted:
            values['grenades_remaining'] = count.value
        labels = propose_combat_visibility(frame.image, values, self.titles)
        if any(label == 'occluded' for label in labels.values()):
            # A tooltip hides the panel; the clean frame after it is more useful.
            return []
        reasons = []
        for field, label in labels.items():
            if label == 'unreadable' or (label == 'unknown' and field in PRIMARY_FIELDS):
                reasons.append(f'{field}:{label}')
        if 'grenades_remaining' not in values:
            reasons.append('grenades_remaining:unknown')
        for field, value in values.items():
            key = _value_key(field, value)
            if self.coverage[key] < self.min_examples:
                reasons.append(f'{key}:rare')
        return reasons

    def __call__(self, screen):
        reasons = [reason for reason in self.gaps(screen) if self.selected[reason] < self.max_per_reason]
        self.selected.update(reasons)
        return reasons
