"""Import versioned card names from the user's installed game."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from shotgun_king_rl.collection.card_catalog import main

if __name__ == '__main__':
    main()
