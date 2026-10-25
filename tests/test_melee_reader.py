"""Confirmed melee absence remains distinct from unreadable gameplay HUDs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_fixtures import NormalContextReader
from shotgun_king_rl.collection.combat_evaluation import build_combat_reader
from shotgun_king_rl.perception import (
    AmmoReading, MeleeDamageReader, PerceptionPipeline, Prediction, PredictionStatus,
    ScreenKind,
)
from shotgun_king_rl.perception.hud import stat_context_patch
from shotgun_king_rl.perception.melee import melee_damage_patch
from test_combat_reader import make_reader, StubReader
from test_hud_readers import CASES, screen


MELEE_CASE = next(case for case in CASES if case.name == "melee_damage")


def absent_screen(y=250):
    image = screen(MELEE_CASE, y=y)
    image[y + 157:y + 222, 270:354] = 0
    return image


def absence_reader():
    image = absent_screen()
    return MeleeDamageReader(zero_examples=[
        (stat_context_patch(image), melee_damage_patch(image)),
    ])


class MeleeAvailabilityReaderTests(unittest.TestCase):
    def test_confirmed_absence_survives_panel_shift(self):
        reader = absence_reader()
        for image in (absent_screen(), absent_screen(y=290)):
            result = reader.read(image)
            self.assertTrue(result.is_accepted)
            self.assertEqual(result.value, 0)

    def test_absence_requires_known_context_and_exact_optional_row(self):
        reader = absence_reader()
        changed_panel = absent_screen()
        changed_panel[215:240, 270:354] = np.random.default_rng(77).integers(
            0, 100, (25, 84, 3), dtype=np.uint8,
        )
        obscured_row = absent_screen()
        obscured_row[415:430, 275:295] = 50
        for image in (None, np.zeros_like(changed_panel), changed_panel,
                      obscured_row, screen(MELEE_CASE), screen(MELEE_CASE, digits="8")):
            with self.subTest(image_is_none=image is None):
                result = reader.read(image)
                self.assertEqual(result.status, PredictionStatus.UNKNOWN)
                self.assertIsNone(result.value)
        self.assertFalse(MeleeDamageReader().read(absent_screen()).is_accepted)

    def test_conflicting_absence_and_positive_references_remain_unknown(self):
        image = screen(MELEE_CASE)
        reader = MeleeDamageReader(
            [(2, melee_damage_patch(image))],
            [(stat_context_patch(image), melee_damage_patch(image))],
        )
        self.assertEqual(reader.read(image).reason, "melee_damage_ambiguous")

    def test_only_explicit_reviewed_train_zeros_become_absence_references(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            for name, split, kind, screen_review, combat_review, value in (
                ("zero", "train", "gameplay", True, True, 0),
                ("val", "val", "gameplay", True, True, 0),
                ("test", "test", "gameplay", True, True, 0),
                ("screen-unreviewed", "train", "gameplay", False, True, 0),
                ("combat-unreviewed", "train", "gameplay", True, False, 0),
                ("card", "train", "card_selection", True, True, 0),
                ("missing", "train", "gameplay", True, True, None),
                ("negative", "train", "gameplay", True, True, -1),
                ("boolean", "train", "gameplay", True, True, False),
                ("fraction", "train", "gameplay", True, True, 0.5),
                ("positive-no-title", "train", "gameplay", True, True, 2),
            ):
                session = collection / "sessions" / name
                (session / "annotations").mkdir(parents=True)
                (session / "originals").mkdir()
                # Excluded splits have no image: reading them must fail this test.
                if split == "train":
                    cv2.imwrite(str(session / "originals/screen.png"), absent_screen())
                (session / "annotations/screen.json").write_text(json.dumps({
                    "split": split, "image": "../originals/screen.png",
                    "game_crop": [0, 0, 1280, 720], "screen_state": {
                        "screen_kind": kind, "combat_stats": {"melee_damage": value},
                        "reviews": {"screen_kind": screen_review, "combat_stats": combat_review},
                    },
                }))
            reader = MeleeDamageReader.from_collection(collection)
            self.assertEqual(len(reader.examples), 0)
            self.assertEqual(len(reader.zero_examples), 1)
            self.assertEqual(reader.read(absent_screen()).value, 0)

    def test_default_combat_factory_loads_absence_without_positive_title(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            session = collection / "sessions/run"
            (session / "annotations").mkdir(parents=True)
            (session / "originals").mkdir()
            cv2.imwrite(str(session / "originals/screen.png"), absent_screen())
            (session / "annotations/screen.json").write_text(json.dumps({
                "split": "train", "image": "../originals/screen.png",
                "game_crop": [0, 0, 1280, 720], "screen_state": {
                    "screen_kind": "gameplay", "combat_stats": {"melee_damage": 0},
                    "reviews": {"screen_kind": True, "combat_stats": True},
                },
            }))
            combat, metadata = build_combat_reader(collection)
            self.assertEqual(combat.melee_reader.read(absent_screen()).value, 0)
            self.assertIsNone(metadata["melee_title"])
            self.assertEqual(metadata["readers"]["melee_reader"]["references"], 0)
            self.assertEqual(metadata["readers"]["melee_reader"]["zero_references"], 1)
            self.assertFalse(combat.melee_reader.read(screen(MELEE_CASE)).is_accepted)

    def test_pipeline_accepts_confirmed_absence_but_blocks_next_unreadable_frame(self):
        combat = make_reader()
        combat.melee_reader = absence_reader()
        board = np.zeros((8, 8), dtype=int)
        board[4, 4] = 1
        pipeline = PerceptionPipeline(
            StubReader(ScreenKind.GAMEPLAY), StubReader(board),
            StubReader(AmmoReading(1, 4)), combat, NormalContextReader(board),
        )
        accepted = pipeline.observe(absent_screen())
        self.assertTrue(accepted.ready_for_policy)
        self.assertEqual(accepted.combat_stats.melee_damage, 0)
        unknown = pipeline.observe(screen(MELEE_CASE))
        self.assertFalse(unknown.ready_for_policy)
        self.assertIsNone(unknown.combat_stats)
        self.assertEqual(len(unknown.issues), 1)
        self.assertIn("combat_stats:melee_damage:", unknown.issues[0])


if __name__ == "__main__":
    unittest.main()
