"""Human card review creates references without approving unrelated labels."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.review.cards import save_card_review
from shotgun_king_rl.review.server import load_item, save_item, validate_screen_state
from shotgun_king_rl.collection.card_catalog import load_catalog
from shotgun_king_rl.collection.card_recognition import propose_cards, match_slot


class CardReviewTests(unittest.TestCase):
    def fixture(self, root, split='train'):
        path = root / 'sessions/ref/annotations/screen.json'
        path.parent.mkdir(parents=True)
        (path.parent.parent / 'originals').mkdir()
        image = np.zeros((720, 1280, 3), np.uint8)
        image[216:332, 540:624] = np.random.default_rng(44).integers(0, 255, (116, 84, 3), dtype=np.uint8)
        cv2.imwrite(str(path.parent.parent / 'originals/screen.png'), image)
        data = {'schema': 'piece-cells-v1', 'session_id': 'ref', 'split': split, 'confirmed': False, 'board': [['empty'] * 8 for _ in range(8)], 'floor': 1, 'exclude_cells': [], 'game_crop': [0, 0, 1280, 720], 'image': '../originals/screen.png', 'screen_state': {'screen_kind': 'card_selection', 'ammo': {'loaded': 2}, 'reviews': {'ammo': True}}}
        path.write_text(json.dumps(data))
        catalog = {'schema': 'card-catalog-v1', 'revision': 1, 'empty_references': [], 'entries': [{'key': key, 'side': side, 'names': {'en': key}, 'effects': None, 'references': []} for key, side in (('black', 'black'), ('white', 'white'))]}
        (root / 'card_catalog.json').write_text(json.dumps(catalog))
        payload = {'id': path.relative_to(root).as_posix(), 'revision': hashlib.sha256(path.read_bytes()).hexdigest(), 'catalog_revision': 1, 'confirmed': True, 'screen_kind_reviewed': True, 'cards': {'left': ['empty'] * 10, 'right': ['empty'] * 10}, 'card_choices': ['black', 'white', 'black', 'white'], 'effect_descriptions': {'black': 'Verified effect text'}}
        return path, image, payload

    def test_confirmation_creates_references_and_preserves_other_approvals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, image, payload = self.fixture(root)
            save_card_review(root, payload)
            data = json.loads(path.read_text())
            self.assertFalse(data['confirmed'])
            self.assertTrue(data['screen_state']['reviews']['ammo'])
            self.assertTrue(data['screen_state']['reviews']['cards'])
            result = propose_cards(image, root, 'card_selection')
            self.assertEqual(result['card_choices'][0], 'black')
            match = next(match for match in result['matches'] if match['key'] == 'black')
            self.assertEqual(match['effects']['description'], 'Verified effect text')
            _, current, _, revision = load_item(root, payload['id'])
            save_item(root, dict(current, id=payload['id'], revision=revision))
            self.assertEqual(load_item(root, payload['id'])[1]['screen_state']['card_choices'], payload['card_choices'])

    def test_held_out_reviews_never_supply_references_or_effect_metadata(self):
        for split in ('val', 'test'):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, payload = self.fixture(root, split)
                result = save_card_review(root, payload)
                self.assertFalse(result['training_references_added'])
                self.assertTrue(all(not entry['references'] and entry['effects'] is None for entry in load_catalog(root)['entries']))

    def test_unknown_wrong_side_and_stale_revisions_are_rejected(self):
        for mutation in ('unknown_key', 'side', 'revision', 'catalog'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _, payload = self.fixture(root)
                before = path.read_bytes()
                if mutation == 'unknown_key': payload['card_choices'][0] = 'unknown_key'
                if mutation == 'side': payload['card_choices'][0] = 'white'
                if mutation == 'revision': payload['revision'] = 'stale'
                if mutation == 'catalog': payload['catalog_revision'] = 0
                with self.assertRaises(ValueError):
                    save_card_review(root, payload)
                self.assertEqual(path.read_bytes(), before)

    def test_partial_review_supplies_only_explicitly_known_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, image, payload = self.fixture(root)
            payload['card_choices'] = ['black', None, None, None]
            payload['cards'] = {'left': [None] * 10, 'right': [None] * 10}
            save_card_review(root, payload)
            state = json.loads(path.read_text())['screen_state']
            self.assertFalse(state['reviews']['cards'])
            self.assertEqual(state['card_reviewed_slots'], ['choice:0'])
            self.assertEqual(propose_cards(image, root, 'card_selection')['card_choices'][0], 'black')
            self.assertEqual(len(load_catalog(root)['entries'][0]['references']), 1)

    def test_saving_draft_revokes_its_reference_associations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, image, payload = self.fixture(root)
            result = save_card_review(root, payload)
            payload.update(confirmed=False, revision=result['revision'], catalog_revision=result['catalog_revision'])
            save_card_review(root, payload)
            self.assertTrue(all(not entry['references'] and entry['effects'] is None for entry in load_catalog(root)['entries']))
            self.assertEqual(propose_cards(image, root, 'card_selection')['reference_count'], 0)

    def test_conflicting_visual_keys_are_unknown(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        template = np.random.default_rng(42).integers(0, 255, (116, 84), dtype=np.uint8)
        image[216:332, 540:624] = template[:, :, None]
        self.assertIsNone(match_slot(image, 'choice:0', [('a', template), ('b', template)]))
        self.assertEqual(validate_screen_state({'card_choices': [None] * 4})['card_choices'], [None] * 4)
