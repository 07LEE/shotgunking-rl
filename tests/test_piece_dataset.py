"""Dataset integrity checks using synthetic images and labels."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotgun_king_rl.collection.dataset import build_dataset
from collection_fixtures import game_screen, piece_annotation


class PieceDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixtures = Path(temporary.name)
        self.annotations = self.fixtures / 'annotations'
        self.annotations.mkdir()
        self.image = self.fixtures / 'screen.png'
        cv2.imwrite(str(self.image), game_screen())
        annotation = piece_annotation()
        annotation['image'] = str(self.image)
        self.annotation = self.annotations / 'screen.json'
        self.annotation.write_text(json.dumps(annotation))

    def test_provisional_labels_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cells"
            with self.assertRaisesRegex(ValueError, "No usable annotations"):
                build_dataset(self.annotations, output)
            self.assertFalse(output.exists())
            summary = build_dataset(self.annotations, output, True)
            self.assertEqual(summary["cells"], 64)
            self.assertEqual(summary["counts"]["train"]["white_king"], 1)
            self.assertEqual(summary["missing_classes"], ["rook", "queen", "special_knight"])
            self.assertFalse(summary["evaluation_ready"])
            records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual({r["split"] for r in records}, {"train"})
            self.assertTrue(all((output / r["path"]).exists() for r in records))

    def test_ambiguous_royals_never_become_training_classes(self):
        data = json.loads(self.annotation.read_text())
        data['confirmed'] = True
        data['board'][0][0] = 'ambiguous_royal'
        self.annotation.write_text(json.dumps(data))
        output = self.fixtures / 'cells'
        with self.assertRaisesRegex(ValueError, 'unknown class label'):
            build_dataset(self.annotations, output)
        self.assertFalse(output.exists())
        data['exclude_cells'] = [[0, 0]]
        self.annotation.write_text(json.dumps(data))
        summary = build_dataset(self.annotations, output)
        self.assertEqual(summary['cells'], 63)
        self.assertNotIn('ambiguous_royal', summary['classes'])

    def test_session_cannot_span_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            annotations = base / "annotations"
            annotations.mkdir()
            data = json.loads(self.annotation.read_text())
            data["image"] = str(self.image)
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
            data = json.loads(self.annotation.read_text())
            image = cv2.imread(str(self.image))
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

    def test_non_gameplay_screen_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            annotations = base / "annotations"
            annotations.mkdir()
            data = json.loads(self.annotation.read_text())
            data.update(image=str(self.image), confirmed=True)
            gameplay = {**data, "screen_state": {"screen_kind": "gameplay", "reviews": {"screen_kind": True}}}
            cards = {**data, "screen_state": {"screen_kind": "card_selection", "reviews": {"screen_kind": True}}}
            (annotations / "gameplay.json").write_text(json.dumps(gameplay))
            (annotations / "cards.json").write_text(json.dumps(cards))

            summary = build_dataset(annotations, base / "output")

        self.assertEqual(summary["screens"], 1)
        self.assertEqual(summary["cells"], 64)
        self.assertEqual(summary["skipped_non_gameplay"], 1)
