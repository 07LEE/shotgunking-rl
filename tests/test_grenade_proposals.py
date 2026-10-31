"""Collection grenade proposals follow shifted slots and optional HUD rows."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.screen_state import propose


class GrenadeProposalTests(unittest.TestCase):
    def image(self, damage_offset=256, slot_shift=0):
        image = np.zeros((720, 1280, 3), np.uint8)
        rng = np.random.default_rng(55)
        image[230:245, 300:320] = 220
        image[78:112, 592 + slot_shift:642 + slot_shift] = rng.integers(0, 255, (34, 50, 3), dtype=np.uint8)
        image[195 + damage_offset:210 + damage_offset, 270:285] = rng.integers(0, 255, (15, 15, 3), dtype=np.uint8)
        image[230 + damage_offset:245 + damage_offset, 300:320] = 220
        return image

    def test_shifted_grenade_slots_and_damage_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / 'sessions/ref'
            (session / 'originals').mkdir(parents=True)
            (session / 'annotations').mkdir()
            cv2.imwrite(str(session / 'originals/screen.png'), self.image())
            path = session / 'annotations/screen.json'
            data = {'split': 'train', 'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'screen_state': {'combat_stats': {'grenades_remaining': 1, 'grenade_damage': 2}, 'reviews': {'combat_stats': True}}}
            path.write_text(json.dumps(data))
            result = propose(self.image(damage_offset=192, slot_shift=3), root)
            self.assertEqual(result['screen_state']['combat_stats']['grenades_remaining'], 1)
            self.assertEqual(result['screen_state']['combat_stats']['grenade_damage'], 2)
            excluded = propose(self.image(), root, exclude=path)
            self.assertIsNone(excluded['screen_state']['combat_stats']['grenade_damage'])
            data['split'] = 'test'
            path.write_text(json.dumps(data))
            held_out = propose(self.image(), root)
            self.assertIsNone(held_out['screen_state']['combat_stats']['grenades_remaining'])
            self.assertIsNone(held_out['screen_state']['combat_stats']['grenade_damage'])

    def test_absent_grenade_displays_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            result = propose(np.zeros((720, 1280, 3), np.uint8), Path(directory))
        self.assertIsNone(result['screen_state']['combat_stats']['grenades_remaining'])
        self.assertIsNone(result['screen_state']['combat_stats']['grenade_damage'])
