"""Offline metrics for reviewed perception labels."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.collection.evaluation import evaluate_readers
from shotgun_king_rl.perception import AmmoReading, Prediction, ScreenKind


class FakeReader:
    def __init__(self, prediction, source):
        self.prediction = prediction
        self.source = source

    def read(self, image):
        return self.prediction


def create_reviewed_screen(collection, split="train"):
    session = collection / "sessions" / "run-01"
    originals = session / "originals"
    annotations = session / "annotations"
    originals.mkdir(parents=True)
    annotations.mkdir()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imwrite(str(originals / "screen.png"), image)
    board = [["empty" for _ in range(8)] for _ in range(8)]
    board[0][0] = "queen"
    board[7][4] = "player_king"
    annotation = {
        "schema": "piece-cells-v1",
        "image": "../originals/screen.png",
        "session_id": "run-01",
        "split": split,
        "game_crop": [0, 0, 1280, 720],
        "confirmed": True,
        "exclude_cells": [],
        "board": board,
        "screen_state": {
            "screen_kind": "gameplay",
            "ammo": {"loaded": 2, "reserve": 6},
            "reviews": {"screen_kind": True, "ammo": True},
        },
    }
    (annotations / "screen.json").write_text(json.dumps(annotation))


class ReaderEvaluationTests(unittest.TestCase):
    def test_rejected_special_knight_remains_visible_in_accuracy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            path = collection / "sessions/run-01/annotations/screen.json"
            annotation = json.loads(path.read_text())
            annotation["board"][0][0] = "special_knight"
            path.write_text(json.dumps(annotation))
            board = np.zeros((8, 8), dtype=int)
            board[0, 0], board[7, 4] = 7, 1
            report = evaluate_readers(
                collection / "sessions",
                piece_reader=FakeReader(Prediction.rejected(
                    board, source="pieces:test", reason="unsupported_runtime_piece:special_knight"
                ), "pieces:test"),
                ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source="ammo:test"), "ammo:test"),
                screen_reader=FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:test"), "screen:test"),
            )
        overall = report["overall"]
        self.assertEqual(overall["pieces"]["strict"]["accuracy"], 1.0)
        self.assertEqual(overall["pieces"]["strict"]["classes"]["special_knight"]["recall"], 1.0)
        self.assertEqual(overall["combined"]["accepted_rate"], 0.0)

    def test_reports_strict_compatible_and_combined_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            predicted_board = np.zeros((8, 8), dtype=int)
            predicted_board[0][0] = 6
            predicted_board[7][4] = 1
            report = evaluate_readers(
                collection / "sessions",
                piece_reader=FakeReader(Prediction.accepted(predicted_board, source="pieces:test"), "pieces:test"),
                ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source="ammo:test"), "ammo:test"),
                screen_reader=FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:test"), "screen:test"),
            )

        overall = report["overall"]
        self.assertEqual(overall["pieces"]["strict"]["accuracy"], 0.984375)
        self.assertEqual(overall["pieces"]["legacy_compatible"]["accuracy"], 1.0)
        self.assertEqual(overall["pieces"]["legacy_compatible"]["full_64_cell_board_exact_rate"], 1.0)
        self.assertEqual(overall["pieces"]["player_king"]["correct_position_rate"], 1.0)
        self.assertEqual(overall["ammo"]["complete_accuracy"], 1.0)
        self.assertEqual(overall["screen_kind"]["accuracy"], 1.0)
        self.assertEqual(overall["combined"]["correct_and_accepted_rate"], 1.0)
        self.assertEqual(report["splits"]["val"]["annotations"], 0)

    def test_unknown_screen_prevents_combined_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            predicted_board = np.zeros((8, 8), dtype=int)
            predicted_board[0][0] = 6
            predicted_board[7][4] = 1
            report = evaluate_readers(
                collection / "sessions",
                piece_reader=FakeReader(Prediction.accepted(predicted_board, source="pieces:test"), "pieces:test"),
                ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source="ammo:test"), "ammo:test"),
                screen_reader=FakeReader(Prediction.unknown(source="screen:test", reason="unverified"), "screen:test"),
            )

        self.assertEqual(report["overall"]["screen_kind"]["confusion"], {"gameplay": {"unknown": 1}})
        self.assertEqual(report["overall"]["combined"]["accepted_rate"], 0.0)
        self.assertEqual(report["overall"]["combined"]["correct_and_accepted_rate"], 0.0)

    def test_invalid_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection, split="other")
            with self.assertRaisesRegex(ValueError, "invalid split"):
                evaluate_readers(collection / "sessions")


if __name__ == "__main__":
    unittest.main()
