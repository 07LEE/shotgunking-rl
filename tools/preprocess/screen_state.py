"""Compatibility exports for conservative screen-state proposals."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.collection.screen_state import *  # noqa: F403
