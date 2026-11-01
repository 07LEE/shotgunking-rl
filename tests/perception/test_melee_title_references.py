"""Runtime and collection honor the same explicit melee title selection."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.combat_evaluation import build_combat_reader
from shotgun_king_rl.collection.optional_combat import melee_title_reference
from shotgun_king_rl.perception.title_references import load_melee_title_reference
from shotgun_king_rl.runtime.perception import build_live_pipeline
from tests.support.hud import CASES, screen


MELEE = next(case for case in CASES if case.name == 'melee_damage')


def create_reference(root, name='known', *, seed=7, value=2):
    session = root / 'sessions' / name
    (session / 'annotations').mkdir(parents=True)
    (session / 'originals').mkdir()
    image = screen(MELEE, seed=seed, digits=str(value))
    cv2.imwrite(str(session / 'originals/screen.png'), image)
    path = session / 'annotations/screen.json'
    path.write_text(json.dumps({
        'split': 'train', 'image': '../originals/screen.png',
        'game_crop': [0, 0, 1280, 720], 'screen_state': {
            'screen_kind': 'gameplay', 'combat_stats': {'melee_damage': value},
            'reviews': {'screen_kind': True, 'combat_stats': True},
        },
    }))
    return path, image


def select(root, path):
    (root / 'title_references.json').write_text(json.dumps({
        'melee_damage': path.relative_to(root).as_posix(),
    }))


class MeleeTitleReferenceTests(unittest.TestCase):
    def test_live_factory_and_collection_use_selected_title_without_config_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, image = create_reference(root)
            select(root, path)
            reference, provenance = load_melee_title_reference(root)
            np.testing.assert_array_equal(reference, melee_title_reference(root))
            pipeline, metadata = build_live_pipeline(root)
            reader = pipeline.combat_stats_reader.melee_reader
            self.assertEqual(reader.read(image).value, 2)
            self.assertFalse(reader.read(screen(MELEE, seed=11)).is_accepted)
            self.assertEqual(metadata['melee_title'], provenance)
            self.assertEqual(provenance['selection'], 'collection_manifest')
            self.assertEqual(provenance['annotation_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_explicit_override_takes_precedence_over_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_image = create_reference(root)
            second, second_image = create_reference(root, 'other', seed=11, value=8)
            select(root, first)
            combat, metadata = build_combat_reader(root, second)
            self.assertEqual(combat.melee_reader.read(second_image).value, 8)
            self.assertFalse(combat.melee_reader.read(first_image).is_accepted)
            self.assertEqual(metadata['melee_title']['annotation'], second.relative_to(root).as_posix())
            self.assertEqual(metadata['melee_title']['selection'], 'explicit')

    def test_absent_manifest_does_not_automatically_pick_a_positive_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, image = create_reference(root)
            combat, metadata = build_combat_reader(root)
            self.assertEqual(len(combat.melee_reader.examples), 0)
            self.assertFalse(combat.melee_reader.read(image).is_accepted)
            self.assertIsNone(metadata['melee_title'])

    def test_excluding_selected_source_never_substitutes_another_title(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = create_reference(root)
            create_reference(root, 'other', seed=11, value=8)
            select(root, path)
            self.assertEqual(load_melee_title_reference(root, exclude=path), (None, None))
            self.assertIsNone(melee_title_reference(root, exclude=path))

    def test_manifest_rejects_nontrain_unreviewed_and_nonpositive_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = create_reference(root)
            select(root, path)
            original = json.loads(path.read_text())
            for split, review, value in (
                ('val', True, 2), ('test', True, 2), ('train', False, 2),
                ('train', 1, 2), ('train', True, 0), ('train', True, True),
            ):
                with self.subTest(split=split, review=review, value=value):
                    data = json.loads(json.dumps(original))
                    data['split'] = split
                    data['screen_state']['reviews']['combat_stats'] = review
                    data['screen_state']['combat_stats']['melee_damage'] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        build_combat_reader(root)

    def test_invalid_manifest_path_and_invisible_title_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'title_references.json'
            for data in ([], {'melee_damage': 5}, {'melee_damage': ''},
                         {'melee_damage': '../escape.json'}, {'melee_damage': '/tmp/title.json'}):
                with self.subTest(data=data):
                    manifest.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_melee_title_reference(root)
            path, _ = create_reference(root)
            select(root, path)
            cv2.imwrite(str(path.parent.parent / 'originals/screen.png'), np.zeros((720, 1280, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, 'title is not visible'):
                load_melee_title_reference(root)
