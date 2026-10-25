"""Offline metrics for reviewed perception labels."""

import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.collection.evaluation import evaluate_readers, main
from shotgun_king_rl.perception import AmmoReading, CombatStatsReading, Prediction, ScreenKind
from shotgun_king_rl.collection.combat_evaluation import build_combat_reader


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
    def test_distinct_queen_is_scored_and_accepted_for_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            board = np.zeros((8, 8), dtype=int)
            board[0, 0], board[7, 4] = 8, 1
            report = evaluate_readers(
                collection / "sessions",
                piece_reader=FakeReader(Prediction.accepted(
                    board, source="pieces:test"
                ), "pieces:test"),
                ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source="ammo:test"), "ammo:test"),
                screen_reader=FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:test"), "screen:test"),
            )
        self.assertEqual(report["overall"]["pieces"]["strict"]["classes"]["queen"]["recall"], 1.0)
        self.assertEqual(report["overall"]["combined"]["accepted_rate"], 1.0)

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
            predicted_board[0][0] = 8
            predicted_board[7][4] = 1
            report = evaluate_readers(
                collection / "sessions",
                piece_reader=FakeReader(Prediction.accepted(predicted_board, source="pieces:test"), "pieces:test"),
                ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source="ammo:test"), "ammo:test"),
                screen_reader=FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:test"), "screen:test"),
            )

        overall = report["overall"]
        self.assertEqual(overall["pieces"]["strict"]["accuracy"], 1.0)
        self.assertEqual(overall["pieces"]["legacy_compatible"]["accuracy"], 1.0)
        self.assertEqual(overall["pieces"]["legacy_compatible"]["full_64_cell_board_exact_rate"], 1.0)
        self.assertEqual(overall["pieces"]["player_king"]["correct_position_rate"], 1.0)
        self.assertEqual(overall["ammo"]["complete_accuracy"], 1.0)
        self.assertEqual(overall["screen_kind"]["accuracy"], 1.0)
        self.assertEqual(overall["combined"]["correct_and_accepted_rate"], 1.0)
        self.assertEqual(report["splits"]["val"]["annotations"], 0)

    def test_wrong_royal_kind_is_not_correct_in_combined_policy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            board = np.zeros((8, 8), dtype=int)
            board[0, 0], board[7, 4] = 9, 1  # Label is queen; prediction is white king.
            report = evaluate_readers(
                collection / 'sessions',
                piece_reader=FakeReader(Prediction.accepted(board, source='pieces:test'), 'pieces:test'),
                ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source='ammo:test'), 'ammo:test'),
                screen_reader=FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen:test'), 'screen:test'),
            )
        overall = report['overall']
        self.assertEqual(overall['pieces']['strict']['accuracy'], 63 / 64)
        self.assertEqual(overall['pieces']['legacy_compatible']['accuracy'], 1.0)
        self.assertEqual(overall['combined']['accepted_rate'], 1.0)
        self.assertEqual(overall['combined']['correct_and_accepted_rate'], 0.0)

    def test_unknown_screen_prevents_combined_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            predicted_board = np.zeros((8, 8), dtype=int)
            predicted_board[0][0] = 8
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

    def test_combat_metrics_require_complete_valid_labels_for_correctness(self):
        stats = CombatStatsReading(4, 3, 5, 65, 0, 0, 0, 2, 0)
        cases = (
            (asdict(stats), Prediction.accepted(stats, source="combat:test"), 1, 1, 1.0),
            (asdict(stats), Prediction.accepted(replace(stats, attack=8), source="combat:test"), 1, 1, 0.0),
            (asdict(stats), Prediction.unknown(source="combat:test", reason="melee_damage:missing@melee:test"), 0, 1, 0.0),
            (asdict(stats), Prediction.accepted(replace(stats, melee_damage=-1), source="combat:test"), 0, 1, 0.0),
            (asdict(stats) | {"melee_damage": None}, Prediction.accepted(stats, source="combat:test"), 1, 0, None),
            (asdict(stats) | {"melee_damage": True}, Prediction.accepted(stats, source="combat:test"), 1, 0, None),
            (asdict(stats) | {"grenade_damage": None}, Prediction.unknown(source="combat:test", reason="grenade_damage:missing@grenade:test"), 0, 0, None),
        )
        for labels, prediction, accepted, complete, rate in cases:
            with self.subTest(labels=labels, status=prediction.status), tempfile.TemporaryDirectory() as directory:
                collection = Path(directory)
                create_reviewed_screen(collection)
                path = collection / "sessions/run-01/annotations/screen.json"
                data = json.loads(path.read_text())
                data["screen_state"]["reviews"]["combat_stats"] = True
                data["screen_state"]["combat_stats"] = labels
                path.write_text(json.dumps(data))
                board = np.zeros((8, 8), dtype=int)
                board[0, 0], board[7, 4] = 8, 1
                report = evaluate_readers(
                    collection / "sessions",
                    piece_reader=FakeReader(Prediction.accepted(board, source="pieces:test"), "pieces:test"),
                    ammo_reader=FakeReader(Prediction.accepted(AmmoReading(2, 6), source="ammo:test"), "ammo:test"),
                    screen_reader=FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source="screen:test"), "screen:test"),
                    combat_reader=FakeReader(prediction, "combat:test"), splits=("train", "val"),
                )
                metrics = report["overall"]["combat_stats"]
                self.assertEqual(metrics["accepted"], accepted)
                self.assertEqual(metrics["complete_valid_labels"], complete)
                self.assertEqual(metrics["correct_and_accepted_rate"], rate)
                self.assertEqual(report["overall"]["combined"]["accepted_rate"], float(accepted))
                self.assertEqual(report["overall"]["combined"]["correct_and_accepted_rate"], rate)
                self.assertEqual(metrics["accepted_without_complete_label"], accepted if not complete else 0)
                if prediction.reason:
                    self.assertEqual(metrics["blocking_reasons"], {prediction.reason: 1})
                self.assertEqual(report["splits"]["test"]["combat_stats"]["screens"], 0)

    def test_unreviewed_combat_labels_do_not_enter_metrics_or_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            combat = FakeReader(Prediction.unknown(source="combat:test", reason="missing"), "combat:test")
            report = evaluate_readers(collection / "sessions", combat_reader=combat)
            self.assertEqual(report["overall"]["combat_stats"]["screens"], 0)
            self.assertEqual(report["overall"]["combined"]["eligible_gameplay_screens"], 0)

    def test_excluded_test_image_is_not_opened_or_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection, split="test")
            image = collection / "sessions/run-01/originals/screen.png"
            image.unlink()
            report = evaluate_readers(collection / "sessions", splits=("train", "val"))
            path = collection / "sessions/run-01/annotations/screen.json"
            data = json.loads(path.read_text())
            data["board"] = "changed excluded test data"
            path.write_text(json.dumps(data))
            again = evaluate_readers(collection / "sessions", splits=("train", "val"))
            self.assertEqual(report["dataset_sha256"], again["dataset_sha256"])
            self.assertEqual(report["overall"]["annotations"], 0)
            for splits in ((), ("train", "train"), ("other",)):
                with self.assertRaisesRegex(ValueError, "unique valid splits"):
                    evaluate_readers(collection / "sessions", splits=splits)

    def test_combat_factory_refuses_unreviewed_or_nontrain_melee_title(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            path = collection / "sessions/run-01/annotations/screen.json"
            original = json.loads(path.read_text())
            for split, reviewed, value in (("test", True, 2), ("val", True, 2), ("train", False, 2), ("train", True, 0)):
                data = json.loads(json.dumps(original))
                data["split"] = split
                data["screen_state"]["reviews"]["combat_stats"] = reviewed
                data["screen_state"]["combat_stats"] = {"melee_damage": value}
                path.write_text(json.dumps(data))
                with self.subTest(split=split, reviewed=reviewed, value=value), self.assertRaisesRegex(ValueError, "reviewed positive gameplay train"):
                    build_combat_reader(collection, path)

    def test_combat_cli_defaults_to_train_val_and_protects_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("original report")
            with (patch.object(sys, "argv", ["evaluate", "--combat-stats", "--output", str(output)]),
                  patch("shotgun_king_rl.collection.combat_evaluation.build_combat_reader", return_value=(object(), {})),
                  patch("shotgun_king_rl.collection.evaluation.evaluate_readers", return_value={}) as evaluate,
                  patch.object(sys, "stderr")):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(evaluate.call_args.kwargs["splits"], ("train", "val"))
            self.assertEqual(output.read_text(), "original report")
        with (patch.object(sys, "argv", ["evaluate", "--melee-title-annotation", "some.json"]),
              patch.object(sys, "stderr")):
            with self.assertRaises(SystemExit):
                main()

    def test_invalid_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection, split="other")
            with self.assertRaisesRegex(ValueError, "invalid split"):
                evaluate_readers(collection / "sessions")


if __name__ == "__main__":
    unittest.main()
