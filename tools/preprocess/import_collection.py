"""Compatibility entry point for private screenshot imports."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.collection.importer import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
