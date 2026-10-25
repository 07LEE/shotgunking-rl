"""Offline royal classification and supported-policy boundary checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shotgun_king_rl.perception.readers import TemplatePieceReader
from shotgun_king_rl.perception.royals import RoyalTemplates


class RoyalReaderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.queen = rng.uniform(0, 255, (45, 45)).astype(np.float32)
        self.king = rng.uniform(0, 255, (45, 45)).astype(np.float32)
        self.reader = RoyalTemplates(((8, self.queen), (9, self.king)))

    def test_distinguishes_both_kinds_and_rejects_unseen_signal(self):
        self.assertEqual(self.reader.classify(self.queen), 8)
        self.assertEqual(self.reader.classify(self.king), 9)
        self.assertEqual(self.reader.classify(np.zeros((45, 45))), 6)
        self.assertEqual(self.reader.classify(np.flipud(self.queen)), 6)

    def test_conflicting_or_missing_kind_is_ambiguous(self):
        reader = RoyalTemplates(((8, self.queen), (9, self.queen)))
        self.assertEqual(reader.classify(self.queen), 6)
        self.assertEqual(RoyalTemplates(((8, self.queen),)).classify(self.queen), 6)

    def test_refinement_preserves_other_cells_and_input(self):
        board = np.zeros((8, 8), dtype=int)
        board[0, :3] = (1, 6, 4)
        image = np.full((720, 1280, 3), 220, dtype=np.uint8)
        with patch("shotgun_king_rl.perception.royals.analyzer._extract_signal", return_value=self.queen):
            result = self.reader.refine(image, board)
        self.assertEqual(result[0].tolist(), [1, 8, 4, 0, 0, 0, 0, 0])
        self.assertEqual(board[0, 1], 6)

    def test_distinct_royals_are_accepted_but_ambiguous_royal_stays_rejected(self):
        image = np.full((720, 1280, 3), 220, dtype=np.uint8)
        board = np.zeros((8, 8), dtype=int)
        board[7, 4] = 1
        for kind in (6, 8, 9):
            board[0, 0] = kind
            with patch("shotgun_king_rl.perception.frame.analyzer.get_state_matrices", return_value=(board, board)):
                reader = TemplatePieceReader(self.reader)
                with patch.object(self.reader, "refine", return_value=board):
                    prediction = reader.read(image)
                self.assertEqual(prediction.is_accepted, kind in (8, 9))
                if kind == 6:
                    self.assertEqual(prediction.reason, 'ambiguous_royal_kind')
                self.assertEqual(prediction.value[0, 0], kind)

    def test_collection_excludes_nontraining_and_unconfirmed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            annotations = collection / "sessions/run/annotations"
            annotations.mkdir(parents=True)
            for i, (split, confirmed, reviewed) in enumerate((
                ("val", True, True), ("test", True, True),
                ("train", False, True), ("train", True, False),
            )):
                data = {"split": split, "confirmed": confirmed,
                        "screen_state": {"screen_kind": "gameplay", "reviews": {"screen_kind": reviewed}}}
                (annotations / f"{i}.json").write_text(json.dumps(data))
            with patch("shotgun_king_rl.perception.royals.cv2.imread") as read:
                self.assertEqual(RoyalTemplates.from_collection(collection).examples, ())
                read.assert_not_called()

    def test_collection_uses_confirmed_labels_and_skips_excluded_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            annotations = collection / "sessions/run/annotations"
            annotations.mkdir(parents=True)
            data = {"split": "train", "confirmed": True, "image": "../originals/screen.png",
                    "board": [["queen", "white_king", "queen"]], "exclude_cells": [[0, 2]],
                    "screen_state": {"screen_kind": "gameplay", "reviews": {"screen_kind": True}}}
            (annotations / "screen.json").write_text(json.dumps(data))
            image = np.full((720, 1280, 3), 220, dtype=np.uint8)
            with patch("shotgun_king_rl.perception.royals.cv2.imread", return_value=image), patch(
                "shotgun_king_rl.perception.royals.analyzer._extract_signal",
                side_effect=(self.queen, self.king),
            ):
                reader = RoyalTemplates.from_collection(collection)
            self.assertEqual([kind for kind, _ in reader.examples], [8, 9])
            self.assertEqual(reader.classify(self.king), 9)
