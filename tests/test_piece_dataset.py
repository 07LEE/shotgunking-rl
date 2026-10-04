"""Small dataset integrity checks using one saved screen."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/preprocess"))
from build_piece_dataset import build_dataset


@unittest.skipUnless((ROOT / "data/test_fixtures/piece_annotations/board_01.json").exists() and (ROOT / "data/test_fixtures/board_01.png").exists(), "Private fixtures are required")
class PieceDatasetTests(unittest.TestCase):
    def test_provisional_labels_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cells"
            with self.assertRaisesRegex(ValueError, "No usable annotations"):
                build_dataset(ROOT / "data/test_fixtures/piece_annotations", output)
            self.assertFalse(output.exists())
            summary = build_dataset(ROOT / "data/test_fixtures/piece_annotations", output, True)
            self.assertEqual(summary["cells"], 64)
            self.assertEqual(summary["counts"]["train"]["white_king"], 1)
            self.assertEqual(summary["missing_classes"], ["rook", "queen", "special_knight"])
            self.assertFalse(summary["evaluation_ready"])
            records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual({r["split"] for r in records}, {"train"})
            self.assertTrue(all((output / r["path"]).exists() for r in records))

    def test_session_cannot_span_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            annotations = base / "annotations"
            annotations.mkdir()
            data = json.loads((ROOT / "data/test_fixtures/piece_annotations/board_01.json").read_text())
            data["image"] = str(ROOT / "data/test_fixtures/board_01.png")
            for split in ("train", "test"):
                data["split"] = split
                (annotations / f"{split}.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "one session cannot span splits"):
                build_dataset(annotations, base / "output", True)
            self.assertFalse((base / "output").exists())

    def test_window_crop_and_excluded_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            annotations = base / "annotations"
            annotations.mkdir()
            data = json.loads((ROOT / "data/test_fixtures/piece_annotations/board_01.json").read_text())
            image = cv2.imread(str(ROOT / "data/test_fixtures/board_01.png"))
            padded = cv2.copyMakeBorder(image, 20, 10, 5, 5, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(annotations / "window.png"), padded)
            data.update(image="window.png", game_crop=[5, 20, 1285, 740], exclude_cells=[[7, 3]], floor=2)
            (annotations / "board.json").write_text(json.dumps(data))
            output = base / "output"
            summary = build_dataset(annotations, output, True)
            self.assertEqual(summary["cells"], 63)
            self.assertEqual(summary["counts"]["train"]["player_king"], 0)
            records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["floor"] == 2 for r in records))
