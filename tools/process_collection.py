"""Process newly saved screenshots without game control or training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shotgun_king_rl.collection.pipeline import main

if __name__ == '__main__':
    main()
