"""Compatibility entry point for the local collection review server."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.review.server import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
