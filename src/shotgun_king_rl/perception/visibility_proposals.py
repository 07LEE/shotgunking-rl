"""Conservative image-only visibility proposals, never review approvals."""

import cv2
import numpy as np

from .hud import stat_patch
from .visibility import COMBAT_FIELDS
from ..vision.screens import check_gameplay_board


def tooltip_over_stats(image):
    """Recognize a row of large pink tooltip hearts inside the stats panel."""
    panel = image[180:650, 270:354].astype(np.int16)
    blue, green, red = cv2.split(panel)
    mask = ((red > 150) & (green < 100) & (red > blue + 35)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    hearts = [row for row in stats[1:count] if 8 <= row[2] <= 16 and 8 <= row[3] <= 16 and row[4] >= 60]
    return any(sum(abs(int(a[1]) - int(b[1])) <= 2 for b in hearts) >= 3 for a in hearts)


def propose_combat_visibility(image, values, titles):
    labels = dict.fromkeys(COMBAT_FIELDS, 'unknown')
    if not check_gameplay_board(image):
        return labels
    occluded = tooltip_over_stats(image)
    if occluded:
        for field in COMBAT_FIELDS:
            if field != 'grenades_remaining':
                labels[field] = 'occluded'
        return labels

    def row_kind(offset):
        row = stat_patch(image, offset, padding=8)
        if row is None:
            return 'unknown'
        # Use the unpadded row to avoid nearby icons outside the stats column.
        core = stat_patch(image, offset)
        if core.std() < 5:
            return 'blank'
        scores = {}
        for field, title in titles:
            if title.std() < 5:
                continue
            score = float(cv2.matchTemplate(row[:46], title, cv2.TM_CCOEFF_NORMED).max())
            scores[field] = max(scores.get(field, -1), score)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked or ranked[0][1] < .90 or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04):
            return 'unknown'
        return ranked[0][0]

    primary = {'attack': row_kind(0), 'range': row_kind(64), 'spread_degrees': row_kind(128)}
    optional = [row_kind(offset) for offset in (192, 256, 320)]
    clean_panel = all(primary[field] == field for field in primary)
    for field in COMBAT_FIELDS:
        kind = 'range' if field in ('range_min', 'range_max') else field
        if kind in primary:
            if primary[kind] == kind:
                labels[field] = 'visible' if values.get(field) is not None else 'unreadable'
        elif kind != 'grenades_remaining':
            if kind in optional:
                labels[field] = 'visible' if values.get(field) is not None else 'unreadable'
            elif clean_panel and 'unknown' not in optional:
                labels[field] = 'not_displayed'
    return labels


def combat_field_readers(combat):
    """Map each titled HUD row to the composed combat reader that owns it."""
    return {
        'attack': combat.attack_reader, 'range': combat.range_reader,
        'spread_degrees': combat.spread_reader, 'penetration_percent': combat.penetration_reader,
        'knockback_percent': combat.knockback_reader, 'melee_damage': combat.melee_reader,
        'grenade_damage': combat.grenade_damage_reader,
    }


def combat_visibility_titles(readers):
    """Share one reviewed title bank between live memory and offline proposals."""
    return [(field, row[:30]) for field, reader in readers.items()
            for value, row in reader.examples if field == 'range' or value > 0]


class CombatVisibilityClassifier:
    """Reuse one reviewed reference bank when classifying an existing collection."""

    source = 'combat-visibility:titles-and-tooltip-v1'

    @classmethod
    def from_collection(cls, collection):
        from ..collection.combat_evaluation import build_combat_reader

        combat, metadata = build_combat_reader(collection)
        result = cls()
        result.readers = combat_field_readers(combat)
        result.titles = combat_visibility_titles(result.readers)
        result.metadata = metadata
        return result

    def propose(self, image):
        values = {}
        for field, reader in self.readers.items():
            result = reader.read(image)
            if result.is_accepted:
                if field == 'range':
                    values.update(range_min=result.value.range_min, range_max=result.value.range_max)
                else:
                    values[field] = result.value
        return propose_combat_visibility(image, values, self.titles)
