"""Session-level dataset split assignment checks."""

import json
from pathlib import Path
import tempfile
import unittest

from shotgun_king_rl.collection.dataset import choose_session_split


def add_annotations(collection, session_id, split, count=1):
    annotations = collection / "sessions" / session_id / "annotations"
    annotations.mkdir(parents=True)
    for index in range(count):
        (annotations / f"{index}.json").write_text(json.dumps({
            "session_id": session_id,
            "split": split,
        }))


class DatasetSplitTests(unittest.TestCase):
    def test_missing_splits_are_filled_before_ratio_balancing(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            self.assertEqual(choose_session_split(collection, "run-01", 4), "train")
            add_annotations(collection, "run-01", "train", 4)
            self.assertEqual(choose_session_split(collection, "run-02", 4), "val")
            add_annotations(collection, "run-02", "val", 4)
            self.assertEqual(choose_session_split(collection, "run-03", 4), "test")

    def test_existing_session_keeps_its_saved_split(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            add_annotations(collection, "run-01", "val", 2)
            self.assertEqual(choose_session_split(collection, "run-01", 20), "val")

    def test_screen_count_deficit_guides_later_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            add_annotations(collection, "run-train", "train", 16)
            add_annotations(collection, "run-val", "val")
            add_annotations(collection, "run-test", "test")
            self.assertEqual(choose_session_split(collection, "run-next", 1), "val")

    def test_mixed_existing_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            add_annotations(collection, "run-01", "train")
            annotation = collection / "sessions" / "run-01" / "annotations" / "other.json"
            annotation.write_text(json.dumps({"session_id": "run-01", "split": "test"}))
            with self.assertRaisesRegex(ValueError, "one session cannot span splits"):
                choose_session_split(collection, "run-02")


if __name__ == "__main__":
    unittest.main()
