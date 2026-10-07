"""Compatibility entry point for collection integrity audits."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.collection.audit import main


if __name__ == "__main__":
    main()
