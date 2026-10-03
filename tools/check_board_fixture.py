"""Compare a saved board with visual annotations without game interaction."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import analyzer

NAMES = {0: "empty", 1: "player", 2: "pawn", 3: "knight", 4: "bishop", 5: "rook", 6: "queen/king"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", type=Path, default=ROOT / "data/test_fixtures/board_01.json")
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    annotation = json.loads(fixture.read_text())
    expected = np.asarray(annotation["board"], dtype=int)
    if expected.shape != (8, 8) or not np.isin(expected, list(NAMES)).all():
        parser.error("Annotation must contain an 8x8 board with IDs 0..6")
    img = cv2.imread(str(fixture.parent / annotation["image"]))
    if img is None:
        parser.error("Fixture image could not be read")
    # Existing analyzer resolves its template directory from the working directory.
    os.chdir(ROOT)
    analyzer.reset_analyzer_cache()
    actual = analyzer.get_state_matrix(img)
    mismatches = np.argwhere(actual != expected)
    occupied = expected != 0
    matched_pieces = int(np.sum((actual == expected) & occupied))
    print(f"Annotation user-confirmed: {annotation.get('user_confirmed', False)}")
    print(f"Cells correct: {64 - len(mismatches)}/64")
    print(f"Occupied cells correctly classified: {matched_pieces}/{int(occupied.sum())}")
    print(f"Enemies: expected={int((expected >= 2).sum())}, detected={int((actual >= 2).sum())}")
    print("Coordinates use chess files a..h and ranks 8..1 (top to bottom)")
    for row, col in mismatches:
        square = f"{chr(ord('a') + int(col))}{8 - int(row)}"
        print(f"{square}: expected={NAMES[int(expected[row, col])]}, actual={NAMES[int(actual[row, col])]}")
    return 1 if len(mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
