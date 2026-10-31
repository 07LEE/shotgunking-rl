"""Refresh supported unknown fields in an existing collection session."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from shotgun_king_rl.collection.refresh import main

if __name__ == '__main__':
    main()
