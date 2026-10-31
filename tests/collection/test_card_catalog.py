"""Card names and recognition require versioned provenance and reviewed train slots."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.collection.card_catalog import import_catalog, reference_patch, slot_box
from shotgun_king_rl.collection.card_recognition import propose_cards


class CardCatalogTests(unittest.TestCase):
    def source(self, root, split='train', reviewed=True):
        image = np.zeros((720, 1280, 3), np.uint8)
        left, top, right, bottom = slot_box('choice:0')
        image[top:bottom, left:right] = np.random.default_rng(23).integers(0, 255, (116, 84, 3), dtype=np.uint8)
        session = root / 'sessions/ref'
        (session / 'originals').mkdir(parents=True)
        (session / 'annotations').mkdir()
        original = session / 'originals/screen.png'
        cv2.imwrite(str(original), image)
        annotation = session / 'annotations/screen.json'
        annotation.write_text(json.dumps({'split': split, 'image': '../originals/screen.png', 'game_crop': [0, 0, 1280, 720], 'screen_state': {'screen_kind': 'card_selection', 'card_choices': ['known', None, None, None], 'reviews': {'screen_kind': True, 'cards': reviewed}}}))
        reference = {'annotation': annotation.relative_to(root).as_posix(), 'slot': 'choice:0', 'image_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}
        catalog = {'schema': 'card-catalog-v1', 'revision': 1, 'entries': [{'key': 'known', 'side': 'black', 'names': {'en': 'Known'}, 'references': [reference], 'effects': None}]}
        (root / 'card_catalog.json').write_text(json.dumps(catalog))
        return image, annotation, reference

    def test_import_uses_localized_names_and_does_not_guess_effect_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'game/lang').mkdir(parents=True)
            (root / 'game/lang/english.txt').write_text('-- CARD NAMES\nCrown::Crown\n-- Next are the names for the white cards\nTower::Tower\n-- CARDS DYNAMIC\neffect_test::Add $0\n')
            (root / 'game/lang/korean.txt').write_text('Crown::왕관\nTower::탑\neffect_test::$0 추가\n')
            catalog = import_catalog(root / 'game', root / 'collection')
            self.assertEqual([entry['side'] for entry in catalog['entries']], ['black', 'white'])
            self.assertEqual(catalog['entries'][0]['names']['ko'], '왕관')
            self.assertIsNone(catalog['entries'][0]['effects'])
            self.assertEqual(catalog['entries'][0]['references'], [])
            with self.assertRaises(FileExistsError):
                import_catalog(root / 'game', root / 'collection')

    def test_reviewed_train_slot_matches_without_inventing_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, annotation, _ = self.source(root)
            result = propose_cards(image, root, 'card_selection')
            self.assertEqual(result['card_choices'][0], 'known')
            self.assertIsNone(result['matches'][0]['effects'])
            self.assertIsNone(propose_cards(image, root, 'card_selection', exclude=annotation)['card_choices'][0])
            self.assertIsNone(propose_cards(np.zeros_like(image), root, 'card_selection')['card_choices'][0])

    def test_held_out_unreviewed_and_tampered_sources_are_not_references(self):
        for split, reviewed in (('val', True), ('test', True), ('train', False)):
            with self.subTest(split=split, reviewed=reviewed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image, _, _ = self.source(root, split, reviewed)
                self.assertEqual(propose_cards(image, root, 'card_selection')['reference_count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, reference = self.source(root)
            reference['image_sha256'] = 'wrong'
            self.assertIsNone(reference_patch(root, reference, 'known'))
            reference['annotation'] = '../escape.json'
            with self.assertRaises(ValueError):
                reference_patch(root, reference, 'known')
