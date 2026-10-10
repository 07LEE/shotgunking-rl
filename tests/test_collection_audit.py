"""Offline checks for collection review and split audits."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.collection.audit import audit_collection


BOARD = [["empty"] * 8 for _ in range(8)]


def add_annotation(root, session_id, split, name, *, complete=True, cards_reviewed=None, screen_kind="gameplay", confirmed=True, image=b"image"):
    session = root / session_id
    originals = session / "originals"
    annotations = session / "annotations"
    originals.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(exist_ok=True)
    (originals / f"{name}.png").write_bytes(image)
    reviews = {
        "screen_kind": complete,
        "ammo": complete,
        "combat_stats": complete,
        "cards": complete if cards_reviewed is None else cards_reviewed,
        "soul": False,
    }
    data = {
        "schema": "piece-cells-v1",
        "image": f"../originals/{name}.png",
        "session_id": session_id,
        "split": split,
        "confirmed": confirmed,
        "board": BOARD,
        "screen_state": {
            "screen_kind": screen_kind,
            "ammo": {"loaded": 2, "reserve": 6},
            "combat_stats": {
                "attack": 4,
                "range_min": 3,
                "range_max": 5,
                "spread_degrees": 65,
                "penetration_percent": 0,
                "knockback_percent": 0,
                "grenades_remaining": 0,
                "grenade_damage": 0,
                "melee_damage": 0,
            },
            "cards": {"left": [], "right": []},
            "reviews": reviews,
        },
    }
    (annotations / f"{name}.json").write_text(json.dumps(data))


class CollectionAuditTests(unittest.TestCase):
    def test_complete_independent_splits_are_evaluation_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            for index, split in enumerate(("train", "val", "test")):
                add_annotation(
                    sessions,
                    f"run-{split}",
                    split,
                    "screen",
                    image=f"image-{index}".encode(),
                )

            report = audit_collection(sessions)

        self.assertTrue(report["evaluation_ready"])
        self.assertEqual(report["screens"], 3)
        self.assertEqual(report["page_info_complete"], 3)
        self.assertEqual(report["splits"]["test"], {"sessions": 1, "screens": 1})
        self.assertEqual(report["usable_labels"]["val"]["combat_stats"], 1)
        self.assertEqual(report["combat_values"]["attack"], [4])

    def test_card_review_is_not_required_for_page_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            for index, split in enumerate(("train", "val", "test")):
                add_annotation(sessions, f"run-{split}", split, "screen", cards_reviewed=False, image=f"image-{index}".encode())

            report = audit_collection(sessions)

        self.assertTrue(report["evaluation_ready"])
        self.assertEqual(report["page_info_complete"], 3)
        self.assertEqual(report["reviews"]["cards"], 0)
        self.assertEqual(report["usable_labels"]["train"]["cards"], 0)

    def test_soul_review_is_independent_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            for index, split in enumerate(("train", "val", "test")):
                add_annotation(sessions, f"run-{split}", split, "screen", image=f"image-{index}".encode())
            path = sessions / "run-train/annotations/screen.json"
            data = json.loads(path.read_text())
            data["screen_state"]["soul_piece"] = "knight"
            data["screen_state"]["reviews"]["soul"] = True
            path.write_text(json.dumps(data))

            report = audit_collection(sessions)

        self.assertTrue(report["evaluation_ready"])
        self.assertEqual(report["reviews"]["soul"], 1)
        self.assertEqual(report["usable_labels"]["train"]["soul"], 1)
        self.assertEqual(report["soul_values"], {"knight": 1})

    def test_grenade_damage_is_required_only_when_grenades_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            add_annotation(sessions, "run-zero", "train", "zero", image=b"zero")
            add_annotation(sessions, "run-live", "val", "live", image=b"live")
            for session, name, remaining in (("run-zero", "zero", 0), ("run-live", "live", 1)):
                path = sessions / session / "annotations" / f"{name}.json"
                data = json.loads(path.read_text())
                data["screen_state"]["combat_stats"]["grenades_remaining"] = remaining
                data["screen_state"]["combat_stats"].pop("grenade_damage")
                path.write_text(json.dumps(data))

            report = audit_collection(sessions)

        self.assertEqual(report["usable_labels"]["train"]["combat_stats"], 1)
        self.assertEqual(report["usable_labels"]["val"]["combat_stats"], 0)

    def test_non_gameplay_screen_does_not_require_piece_or_stat_review(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            add_annotation(sessions, "run-01", "train", "gameplay", image=b"gameplay")
            add_annotation(sessions, "run-01", "train", "cards", screen_kind="card_selection", confirmed=False, image=b"cards")

            report = audit_collection(sessions)

        self.assertEqual(report["screens"], 2)
        self.assertEqual(report["piece_screens"], 1)
        self.assertEqual(report["pieces_confirmed"], 1)
        self.assertEqual(report["page_info_complete"], 2)
        self.assertNotIn("piece_review_incomplete:1", report["issues"])

    def test_missing_reviews_splits_and_duplicate_images_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            add_annotation(sessions, "run-01", "train", "one", complete=False)
            add_annotation(sessions, "run-01", "train", "two", image=b"image")

            report = audit_collection(sessions)

        self.assertFalse(report["evaluation_ready"])
        self.assertIn("missing_split:val", report["issues"])
        self.assertIn("missing_split:test", report["issues"])
        self.assertIn("page_review_incomplete:1", report["issues"])
        self.assertIn("missing_usable_labels:val:combat_stats", report["issues"])
        self.assertTrue(any(issue.startswith("duplicate_image:") for issue in report["issues"]))
        self.assertEqual(report["reviews"]["combat_stats"], 1)

    def test_one_session_cannot_span_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory)
            add_annotation(sessions, "run-01", "train", "one", image=b"one")
            add_annotation(sessions, "run-01", "test", "two", image=b"two")

            report = audit_collection(sessions)

        self.assertIn("mixed_session_split:run-01", report["issues"])
        self.assertIsNone(report["sessions"][0]["split"])


if __name__ == "__main__":
    unittest.main()
