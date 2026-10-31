"""Synthetic HUD cases shared by reader tests."""

from dataclasses import dataclass

import cv2
import numpy as np

from shotgun_king_rl.perception import AttackReader, RangeReader, RangeReading, SpreadReader, GrenadeDamageReader, MeleeDamageReader
from shotgun_king_rl.perception.attack import attack_patch
from shotgun_king_rl.perception.range import range_patch
from shotgun_king_rl.perception.spread import spread_patch
from shotgun_king_rl.perception.grenade_damage import grenade_damage_patch
from shotgun_king_rl.perception.melee import melee_damage_patch


@dataclass(frozen=True)
class HudCase:
    name: str
    reader: type
    patch: object
    offset: int
    value: object
    conflicting: object
    digits: str
    unseen: str | None
    labels: dict
    invalid_labels: tuple = ()


CASES = (
    HudCase('melee_damage', MeleeDamageReader, melee_damage_patch, 192, 2, 3, '2', '8',
            {'melee_damage': 2},
            tuple({'melee_damage': value} for value in (-1, None, True))),
    HudCase('attack', AttackReader, attack_patch, 0, 4, 8, '4', None, {'attack': 4}),
    HudCase('range', RangeReader, range_patch, 64, RangeReading(3, 5), RangeReading(3, 6), '3-5', '2-8',
            {'range_min': 3, 'range_max': 5},
            ({'range_min': 5, 'range_max': 3}, {'range_max': None}, {'range_min': True})),
    HudCase('spread', SpreadReader, spread_patch, 128, 65, 95, '65', '180', {'spread_degrees': 65},
            tuple({'spread_degrees': value} for value in (-1, 361, True))),
    HudCase('grenade_damage', GrenadeDamageReader, grenade_damage_patch, 256, 2, 3, '2', '8',
            {'grenade_damage': 2, 'grenades_remaining': 0},
            tuple({'grenade_damage': value} for value in (-1, None, True))),
)


def screen(case, y=250, seed=7, digits=None):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    digits = case.digits if digits is None else digits
    top = y + case.offset - 35
    image[top:top+25, 270:354] = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    cv2.putText(image, digits if case.offset == 0 else '4', (305, y+20), 0, .8, (220, 220, 220), 2)
    if case.offset:
        cv2.putText(image, digits, (285, y+case.offset+20), 0, .7, (220, 220, 220), 2)
    return image
