import sys
import os
import cv2
import numpy as np

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import analyzer as _analyzer

TEMPLATES_DIR = "data/templates"
CELL_SIZE = 65
CROP_INNER = (10, 55)

def extract_piece_signal(patch):
    cropped = patch[CROP_INNER[0]:CROP_INNER[1], CROP_INNER[0]:CROP_INNER[1]]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(float)
    corners = [gray[:4,:4], gray[:4,-4:], gray[-4:,:4], gray[-4:,-4:]]
    bg = np.mean([c.mean() for c in corners])
    signal = np.abs(gray - bg)
    mx = signal.max()
    if mx > 1e-3:
        signal = (signal / mx * 255).clip(0, 255).astype(np.uint8)
    else:
        signal = np.zeros((45, 45), dtype=np.uint8)
    return signal

def main():
    img_path = "data/screenshot.png"
    if not os.path.exists(img_path):
        print(f"Screenshot not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    board = _analyzer.crop_chessboard(img)
    if board is None:
        print("Failed to crop board")
        sys.exit(1)

    # Mapping of (row, col) to template filename
    mapping = {
        (0, 1): "knight_3_0.png",
        (0, 2): "bishop_4_0.png",
        (0, 3): "queen_6_0.png",
        (0, 4): "rook_5_0.png",
        (0, 5): "bishop_4_1.png",
        (0, 6): "knight_3_1.png",
        (1, 1): "pawn_2_0.png",
        (1, 2): "pawn_2_1.png",
        (1, 3): "pawn_2_2.png",
        (1, 4): "pawn_2_3.png",
        (1, 5): "pawn_2_4.png",
        (7, 3): "king_1_0.png"
    }

    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    print("Extracting and updating templates from current screenshot...")

    for (r, c), filename in mapping.items():
        patch = board[r*CELL_SIZE:(r+1)*CELL_SIZE, c*CELL_SIZE:(c+1)*CELL_SIZE]
        signal = extract_piece_signal(patch)
        out_path = os.path.join(TEMPLATES_DIR, filename)
        cv2.imwrite(out_path, signal)
        print(f"Saved: {out_path} from cell ({r}, {c})")

if __name__ == "__main__":
    main()
