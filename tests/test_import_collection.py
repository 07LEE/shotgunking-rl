"""Check inbox imports preserve files and reject collisions."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from import_collection import import_images


class CollectionImportTests(unittest.TestCase):
    def test_import_and_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            inbox = collection / "inbox"
            inbox.mkdir()
            (inbox / "a.png").write_bytes(b"saved image")
            (inbox / "notes.txt").write_text("retain")
            self.assertEqual(import_images(collection, "session-01"), 1)
            target = collection / "sessions/session-01/originals/a.png"
            self.assertEqual(target.read_bytes(), b"saved image")
            self.assertTrue((inbox / "notes.txt").exists())
            (inbox / "a.png").write_bytes(b"new image")
            (inbox / "b.png").write_bytes(b"another image")
            with self.assertRaisesRegex(ValueError, "Already exists"):
                import_images(collection, "session-01")
            self.assertTrue((inbox / "b.png").exists())
            self.assertEqual(target.read_bytes(), b"saved image")

    def test_invalid_session_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                import_images(Path(directory), "../escape")
