"""Non-visible game-state values cannot become numeric image labels."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2

from shotgun_king_rl.perception.visibility import combat_visibility, combat_visual_targets, visual_combat_stats
from shotgun_king_rl.review.server import load_item, save_item, validate_screen_state
from shotgun_king_rl.collection.dataset import build_dataset
from shotgun_king_rl.collection.refresh import fill_unknowns
from tests.support.hud import CASES, screen
from tests.support.collection import piece_annotation, game_screen


class CombatVisibilityTests(unittest.TestCase):
    def test_visibility_and_game_state_value_are_separate(self):
        for status in ('occluded', 'not_displayed', 'unreadable', 'unknown'):
            state = {'combat_stats': {'grenade_damage': 2}, 'combat_visibility': {'grenade_damage': status},
                     'reviews': {'combat_stats': True}}
            self.assertIsNone(visual_combat_stats(state)['grenade_damage'])
            target = combat_visual_targets(state)['grenade_damage']
            self.assertEqual(target, {'visibility': status, 'value': None, 'reviewed': True})
            self.assertEqual(state['combat_stats']['grenade_damage'], 2)

    def test_legacy_labels_are_not_silently_called_visible(self):
        state = {'combat_stats': {'grenade_damage': 2}, 'reviews': {'combat_stats': True}}
        self.assertEqual(visual_combat_stats(state)['grenade_damage'], 2)
        self.assertEqual(combat_visual_targets(state)['grenade_damage'],
                         {'visibility': 'unknown', 'value': None, 'reviewed': False})
        state['combat_visibility'] = {'grenade_damage': 'visible'}
        self.assertEqual(combat_visual_targets(state)['grenade_damage']['value'], 2)
        state['reviews']['combat_stats'] = False
        self.assertIsNone(combat_visual_targets(state)['grenade_damage']['value'])

    def test_bad_visibility_and_visible_without_number_are_rejected(self):
        for labels in ([], {'bad': 'visible'}, {'grenade_damage': 'remembered'}, {'grenade_damage': 'visible'}):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                combat_visibility({'combat_visibility': labels})

    def test_hud_reference_loaders_exclude_explicitly_occluded_labels(self):
        for case in CASES:
            with self.subTest(reader=case.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, status in (('visible', 'visible'), ('hidden', 'occluded')):
                    session = root / 'sessions' / name
                    (session / 'annotations').mkdir(parents=True)
                    (session / 'originals').mkdir()
                    # A hidden annotation must be skipped before image loading.
                    if status == 'visible':
                        cv2.imwrite(str(session / 'originals/screen.png'), screen(case))
                    (session / 'annotations/screen.json').write_text(json.dumps({
                        'split': 'train', 'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720],
                        'screen_state': {'screen_kind': 'gameplay', 'combat_stats': case.labels,
                                         'combat_visibility': {key: status for key in case.labels},
                                         'reviews': {'combat_stats': True, 'screen_kind': True}},
                    }))
                kwargs = {'title_reference': case.patch(screen(case))[:30]} if case.name == 'melee_damage' else {}
                reader = case.reader.from_collection(root, **kwargs)
                self.assertEqual(len(reader.examples), 1)

    def test_save_reload_export_preserve_occlusion_and_require_reapproval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / 'sessions/run'
            (session / 'annotations').mkdir(parents=True)
            (session / 'originals').mkdir()
            cv2.imwrite(str(session / 'originals/screen.png'), game_screen())
            annotation = piece_annotation()
            annotation.update(image='../originals/screen.png', session_id='run', confirmed=True, floor=1)
            annotation['screen_state'] = {'screen_kind': 'gameplay', 'combat_stats': {'grenade_damage': 2},
                                          'reviews': {'screen_kind': True, 'combat_stats': True}}
            path = session / 'annotations/screen.json'
            path.write_text(json.dumps(annotation))
            relative = 'sessions/run/annotations/screen.json'
            _, data, _, revision = load_item(root, relative)
            data['screen_state']['combat_visibility'] = {'grenade_damage': 'occluded'}
            payload = dict(data, id=relative, revision=revision)
            saved = save_item(root, payload)
            self.assertFalse(saved['reviews']['combat_stats'])
            _, data, _, revision = load_item(root, relative)
            self.assertEqual(data['screen_state']['combat_stats']['grenade_damage'], 2)
            self.assertEqual(data['screen_state']['combat_visibility']['grenade_damage'], 'occluded')
            data['screen_state']['reviews']['combat_stats'] = True
            payload = dict(data, id=relative, revision=revision, combat_reviewed=True)
            self.assertTrue(save_item(root, payload)['reviews']['combat_stats'])
            output = root / 'dataset'
            build_dataset(session / 'annotations', output)
            record = json.loads((output / 'manifest.jsonl').read_text().splitlines()[0])
            self.assertEqual(record['combat_visual_targets']['grenade_damage'],
                             {'visibility': 'occluded', 'value': None, 'reviewed': True})
            self.assertEqual(record['screen_state']['combat_stats']['grenade_damage'], 2)

    def test_autofill_cannot_replace_occlusion_with_predicted_number(self):
        state = {'screen_kind': 'gameplay', 'combat_stats': {'grenade_damage': None},
                 'combat_visibility': {'grenade_damage': 'occluded'}}
        proposal = {'floor': None, 'screen_state': {'combat_stats': {'grenade_damage': 2}}}
        fill_unknowns({'confirmed': True, 'screen_state': state}, proposal)
        self.assertIsNone(state['combat_stats']['grenade_damage'])
        normalized = validate_screen_state(state)
        self.assertEqual(normalized['combat_visibility']['grenade_damage'], 'occluded')
