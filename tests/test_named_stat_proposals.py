"""Named integer recognition requires independent title and glyph evidence."""

import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.named_stats import match_named_integer
from shotgun_king_rl.collection.numeric import numeric_token
from shotgun_king_rl.perception.hud import stat_patch


class NamedStatProposalTests(unittest.TestCase):
    def image(self, value, offset, seed=18):
        image = np.zeros((720, 1280, 3), np.uint8)
        image[230:245, 300:320] = 220
        image[195 + offset:210 + offset, 270:285] = np.random.default_rng(seed).integers(0, 255, (15, 15, 3), dtype=np.uint8)
        cv2.putText(image, str(value), (300, 250 + offset), 0, .8, (220, 220, 220), 2)
        return image

    def test_integer_from_another_field_requires_matching_damage_title(self):
        reference = self.image(2, 256)
        integer = numeric_token(self.image(3, 192), 192)
        titles = [('grenade_damage', stat_patch(reference, 256)[:30])]
        self.assertEqual(match_named_integer(self.image(3, 192), 'grenade_damage', titles, [(3, integer)])[0], 3)
        self.assertIsNone(match_named_integer(self.image(3, 192, seed=25), 'grenade_damage', titles, [(3, integer)]))

    def test_conflicting_titles_or_integer_labels_remain_unknown(self):
        image = self.image(3, 192)
        title = stat_patch(image, 192)[:30]
        token = numeric_token(image, 192)
        self.assertIsNone(match_named_integer(image, 'grenade_damage', [('grenade_damage', title), ('melee_damage', title)], [(3, token)]))
        self.assertIsNone(match_named_integer(image, 'grenade_damage', [('grenade_damage', title)], [(3, token), (8, token)]))

    def test_missing_evidence_remains_unknown(self):
        image = self.image(3, 192)
        self.assertIsNone(match_named_integer(image, 'grenade_damage', [], []))
        self.assertIsNone(match_named_integer(np.zeros_like(image), 'grenade_damage', [], []))
