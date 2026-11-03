"""Visibility proposals distinguish known titles, blank rows and tooltip hearts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from shotgun_king_rl.perception.hud import stat_patch
from shotgun_king_rl.perception.visibility_proposals import (
    combat_field_readers, combat_visibility_titles, propose_combat_visibility, tooltip_over_stats,
)


def hud():
    image = np.zeros((720, 1280, 3), np.uint8)
    rng = np.random.default_rng(71)
    for offset in (0, 64, 128, 192):
        image[195+offset:220+offset, 270:354] = rng.integers(0, 100, (25, 84, 3), dtype=np.uint8)
        cv2.putText(image, '3', (300, 250+offset), 0, .8, (220, 220, 220), 2)
    return image


class VisibilityProposalTests(unittest.TestCase):
    @patch('shotgun_king_rl.perception.visibility_proposals.check_gameplay_board', return_value=True)
    def test_recognized_title_distinguishes_visible_number_from_unreadable_number(self, board):
        image = hud()
        titles = [(field, stat_patch(image, offset)[:30]) for field, offset in
                  (('attack', 0), ('range', 64), ('spread_degrees', 128), ('grenade_damage', 192))]
        result = propose_combat_visibility(image, {'grenade_damage': 3}, titles)
        self.assertEqual(result['grenade_damage'], 'visible')
        self.assertEqual(propose_combat_visibility(image, {}, titles)['grenade_damage'], 'unreadable')
        self.assertEqual(result['melee_damage'], 'not_displayed')
        self.assertEqual(result['grenades_remaining'], 'unknown')

    @patch('shotgun_king_rl.perception.visibility_proposals.check_gameplay_board', return_value=True)
    def test_unknown_optional_title_does_not_prove_other_fields_absent(self, board):
        image = hud()
        titles = [(field, stat_patch(image, offset)[:30]) for field, offset in
                  (('attack', 0), ('range', 64), ('spread_degrees', 128))]
        result = propose_combat_visibility(image, {}, titles)
        self.assertEqual(result['grenade_damage'], 'unknown')

    @patch('shotgun_king_rl.perception.visibility_proposals.check_gameplay_board', return_value=True)
    def test_aligned_tooltip_hearts_classify_stats_as_occluded(self, board):
        image = hud()
        for x in (280, 300, 320):
            cv2.rectangle(image, (x, 350), (x+11, 361), (80, 0, 255), -1)
        self.assertTrue(tooltip_over_stats(image))
        labels = propose_combat_visibility(image, {'grenade_damage': 3}, [])
        self.assertEqual(labels['grenade_damage'], 'occluded')
        self.assertEqual(labels['attack'], 'occluded')
        self.assertEqual(labels['grenades_remaining'], 'unknown')
        image[350:362, 320:332] = 0
        self.assertFalse(tooltip_over_stats(image))


class VisibilityTitleBankTests(unittest.TestCase):
    def test_title_bank_keeps_range_and_nonzero_rows_as_title_slices(self):
        def row(fill):
            return np.full((40, 84), fill, np.uint8)

        def reader(*examples):
            return SimpleNamespace(examples=examples)

        combat = SimpleNamespace(
            attack_reader=reader((0, row(1)), (3, row(2))),
            range_reader=reader((SimpleNamespace(range_min=0, range_max=0), row(3))),
            spread_reader=reader(), penetration_reader=reader(), knockback_reader=reader(),
            melee_reader=reader(), grenade_damage_reader=reader((0, row(4)), (2, row(5))),
        )
        readers = combat_field_readers(combat)
        self.assertIs(readers['grenade_damage'], combat.grenade_damage_reader)
        titles = combat_visibility_titles(readers)
        self.assertEqual([(field, int(title[0, 0])) for field, title in titles],
                         [('attack', 2), ('range', 3), ('grenade_damage', 5)])
        self.assertTrue(all(title.shape == (30, 84) for _, title in titles))
