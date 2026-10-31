"""Shared knockback and penetration behavior checks with per-reader subtests."""

from functools import partial
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from shotgun_king_rl.perception.knockback import KnockbackReader, knockback_patch
from shotgun_king_rl.perception.penetration import PenetrationReader, penetration_patch
from shotgun_king_rl.perception.types import PredictionStatus
from shotgun_king_rl.perception.hud import stat_context_patch


def screen(y=250, seed=7, digits="50%"):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    title = np.random.default_rng(seed).integers(0, 100, (25, 84, 3), dtype=np.uint8)
    image[y + 157:y + 182, 270:354] = title
    cv2.putText(image, '4', (305, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (220, 220, 220), 2)
    cv2.putText(image, digits, (285, y + 212), cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 220, 220), 2)
    return image

READERS = (
    ('knockback', KnockbackReader, knockback_patch, 50),
    ('penetration', PenetrationReader, penetration_patch, 40),
)


class PercentageReaderTests(unittest.TestCase):
    def test_accepts_known_percentage_after_vertical_hud_shift(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                reader = reader_cls([(known, patch_fn(make_screen(250)))])
                result = reader.read(make_screen(290))
                self.assertTrue(result.is_accepted)
                self.assertEqual(result.value, known)

    def test_rejects_conflicting_labels_for_same_patch(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                template = patch_fn(make_screen())
                result = reader_cls([(known, template), (75, template)]).read(make_screen())
                self.assertEqual(result.status, PredictionStatus.UNKNOWN)
                self.assertEqual(result.reason, f'{field}_ambiguous')

    def test_requires_title_context_not_just_matching_digit(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                reader = reader_cls([(known, patch_fn(make_screen()))])
                self.assertFalse(reader.read(make_screen(seed=11)).is_accepted)

    def test_unseen_percentage_is_unknown(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                reader = reader_cls([(known, patch_fn(make_screen()))])
                self.assertFalse(reader.read(make_screen(digits="25%")).is_accepted)

    def test_absent_row_is_unknown_rather_than_zero(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                image = make_screen()
                image[400:600, 270:354] = 0
                reader = reader_cls([(known, patch_fn(make_screen()))])
                result = reader.read(image)
                self.assertFalse(result.is_accepted)
                self.assertIsNone(result.value)

    def test_reviewed_zero_requires_both_known_row_and_panel(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                absent = make_screen()
                absent[400:600, 270:354] = 0
                reader = reader_cls(zero_examples=[(stat_context_patch(absent), patch_fn(absent))])
                self.assertTrue(reader.read(absent).is_accepted)
                self.assertEqual(reader.read(absent).value, 0)
                self.assertFalse(reader.read(make_screen(digits="75%")).is_accepted)
                different = absent.copy()
                different[210:235, 270:354] = np.random.default_rng(77).integers(0, 100, (25, 84, 3), dtype=np.uint8)
                self.assertFalse(reader.read(different).is_accepted)
                self.assertFalse(reader.read(np.zeros_like(absent)).is_accepted)

    def test_reviewed_zero_conflicting_with_positive_is_ambiguous(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                image = make_screen()
                reader = reader_cls(
                    [(known, patch_fn(image))],
                    [(stat_context_patch(image), patch_fn(image))],
                )
                self.assertEqual(reader.read(image).reason, f'{field}_ambiguous')

    def test_missing_reference_is_unknown(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                self.assertEqual(reader_cls().read(make_screen()).reason, f'{field}_no_match')

    def test_invalid_and_blank_images_are_unknown(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                reader = reader_cls()
                for image in (None, np.zeros((720, 1280, 3), dtype=np.uint8)):
                    with self.subTest(image_is_none=image is None):
                        self.assertEqual(reader.read(image).status, PredictionStatus.UNKNOWN)

    def test_only_reviewed_gameplay_train_references_are_loaded(self):
        for field, reader_cls, patch_fn, known in READERS:
            with self.subTest(reader=field):
                make_screen = partial(screen, digits=f"{known}%")
                with tempfile.TemporaryDirectory() as directory:
                    collection = Path(directory)
                    for name, split, reviewed, kind, value in (
                        ('train', 'train', True, 'gameplay', known),
                        ('val', 'val', True, 'gameplay', 8),
                        ('test', 'test', True, 'gameplay', 6),
                        ('unreviewed', 'train', False, 'gameplay', 3),
                        ('card', 'train', True, 'card_selection', 2),
                        ('negative', 'train', True, 'gameplay', -1),
                        ('zero', 'train', True, 'gameplay', 0),
                        ('val_zero', 'val', True, 'gameplay', 0),
                        ('unreviewed_zero', 'train', False, 'gameplay', 0),
                        ('missing', 'train', True, 'gameplay', None),
                        ('overflow', 'train', True, 'gameplay', 101),
                        ('boolean', 'train', True, 'gameplay', True),
                    ):
                        session = collection / 'sessions' / name
                        (session / 'originals').mkdir(parents=True)
                        (session / 'annotations').mkdir()
                        sample = make_screen()
                        if type(value) is int and value == 0:
                            sample[400:600, 270:354] = 0
                        cv2.imwrite(str(session / 'originals/screen.png'), sample)
                        data = {
                            'split': split, 'image': '../originals/screen.png',
                            'game_crop': [0, 0, 1280, 720],
                            'screen_state': {
                                'screen_kind': kind, 'combat_stats': {f'{field}_percent': value},
                                'reviews': {'screen_kind': True, 'combat_stats': reviewed},
                            },
                        }
                        (session / 'annotations/screen.json').write_text(json.dumps(data))
                    reader = reader_cls.from_collection(collection)
                    self.assertEqual(len(reader.examples), 1)
                    self.assertEqual(len(reader.zero_examples), 1)
                    absent = make_screen()
                    absent[400:600, 270:354] = 0
                    self.assertEqual(reader.read(absent).value, 0)
                    self.assertEqual(reader.read(make_screen()).value, known)


if __name__ == '__main__':
    unittest.main()
