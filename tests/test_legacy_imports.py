"""Legacy runtime modules must remain importable without desktop access."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class LegacyImportTests(unittest.TestCase):
    def test_environment_import_does_not_require_display(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, DISPLAY=":invalid", PYTHONPATH=str(root / "src"))
        result = subprocess.run(
            [sys.executable, "-c", "import env; print(env.ShotgunKingEnv.__name__)"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ShotgunKingEnv")
