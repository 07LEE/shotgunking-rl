"""
In-game live capture based piece template extractor tool.

Usage:
  1. Run Shotgun King and start a game
  2. Run this script while pieces are on the board
  3. Automatically takes a screenshot, detects pieces, and saves templates
  4. Visually inspect saved PNGs to verify correct labeling

  uv run python scripts/capture_templates.py
"""
import subprocess
import sys
import time
import os

# Add src path (to allow running without PYTHONPATH)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

try:
    import mss
except ImportError:
    print("mss not installed. Install and rerun: uv add mss")
    sys.exit(1)

# Import dynamic board detection function from analyzer
try:
    import analyzer as _analyzer
    from capture import capture_screen
except ImportError as e:
    print(f"[Error] Failed to load analyzer/capture module: {e}")
    sys.exit(1)

TEMPLATES_DIR = "data/templates"
CELL_SIZE = 65          # 520 / 8
CROP_INNER = (10, 55)   # Crop inner piece (remove grid lines)


# Window capture helper functions removed in favor of src/capture.py reuse.


def crop_board(img):
    """Crop dynamic board using analyzer.crop_chessboard and return 520x520.

    ROI is cached in data/board_roi.json to use the same coordinates on every run.
    """
    return _analyzer.crop_chessboard(img)


def extract_piece_signal(patch):
    """Extract pure piece signal by subtracting corner background (45x45)."""
    cropped = patch[CROP_INNER[0]:CROP_INNER[1],
                    CROP_INNER[0]:CROP_INNER[1]]
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


def detect_pieces(board):
    """Return piece candidate positions and areas based on geometry.

    Returns:
        list of (row, col, area_white, area_black, h_white)
    """
    pieces = []
    for r in range(8):
        for c in range(8):
            patch = board[r*CELL_SIZE:(r+1)*CELL_SIZE,
                          c*CELL_SIZE:(c+1)*CELL_SIZE]
            cropped = patch[CROP_INNER[0]:CROP_INNER[1],
                            CROP_INNER[0]:CROP_INNER[1]]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # Detect white pieces
            _, tw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            cw, _ = cv2.findContours(tw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_w, h_w = 0, 0
            if cw:
                cx = max(cw, key=cv2.contourArea)
                area_w = cv2.contourArea(cx)
                if 75 < area_w < 1700:
                    _, _, ww, hh = cv2.boundingRect(cx)
                    h_w = hh

            # Detect black pieces (player king and other black pieces)
            _, tb = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            cb, _ = cv2.findContours(tb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_b = 0
            if cb:
                cbx = max(cb, key=cv2.contourArea)
                area_b = cv2.contourArea(cbx)

            # Determine piece detection validity
            is_white_piece = 75 < area_w < 1700
            is_black_piece = area_b > 50

            if is_white_piece or is_black_piece:
                effective_w = area_w if is_white_piece else 0
                effective_b = area_b if is_black_piece else 0
                pieces.append((r, c, effective_w, effective_b, h_w))

    return pieces


def main():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    print("=" * 60)
    print("  Shotgun King Piece Template Live Capture Tool")
    print("=" * 60)
    print()
    print("Run the game and press Enter when pieces are visible on the board.")
    print("(In-game state only, do not run on GameOver screen)")
    input(">>> Press Enter when ready: ")

    print("Capturing in 3 seconds...", end="", flush=True)
    for i in range(3, 0, -1):
        print(f" {i}", end="", flush=True)
        time.sleep(1)
    print()

    success = capture_screen("data/screenshot.png", "Shotgun King")
    if not success:
        print("[Error] Failed to capture game window.")
        sys.exit(1)
    img = cv2.imread("data/screenshot.png")
    if img is None:
        print("[Error] Failed to read captured screenshot.")
        sys.exit(1)
    print(f"Screenshot saved: data/screenshot.png ({img.shape[1]}x{img.shape[0]})")

    board = crop_board(img)
    pieces = detect_pieces(board)

    print(f"\nDetected piece candidates: {len(pieces)}")
    print(f"{'Position':10s} {'WhiteArea':>12s} {'BlackArea':>12s} {'h':>6s}")
    print("-" * 45)
    for r, c, aw, ab, hw in pieces:
        kind = "BLACK_PIECE?" if ab > 50 else f"WHITE(h={hw})"
        print(f"  ({r},{c})    aw={aw:7.0f}   ab={ab:7.0f}   h={hw:3d}   {kind}")

    # List of templates to save (visually verify labels)
    print()
    print("Saving detected piece patches to data/templates/.")
    print("After saving, visually check PNG files and rename them to correct piece names.")
    print()
    print("  Piece ID Guide:")
    print("    1: king_1*.png    (Player Black King, e.g. king_1_0.png, king_1_1.png)")
    print("    2: pawn_2*.png    (White Pawn, e.g. pawn_2_0.png, pawn_2_1.png)")
    print("    3: knight_3*.png  (White Knight, e.g. knight_3_0.png, knight_3_1.png)")
    print("    4: bishop_4*.png  (White Bishop, e.g. bishop_4_0.png, bishop_4_1.png)")
    print("    5: rook_5*.png    (White Rook, e.g. rook_5_0.png, rook_5_1.png)")
    print("    6: queen_6*.png   (White Queen, e.g. queen_6_0.png, queen_6_1.png)")
    print()
    print("  Use the command below to copy candidate files as templates:")
    print("    cp data/templates/<candidate_file> data/templates/pawn_2_0.png")
    print("    cp data/templates/<candidate_file> data/templates/knight_3_0.png")
    print("    ... etc. (Use suffixes like _0, _1, _2 to maintain multiple templates)")
    print()

    saved = {}
    for r, c, aw, ab, hw in pieces:
        patch = board[r*CELL_SIZE:(r+1)*CELL_SIZE,
                      c*CELL_SIZE:(c+1)*CELL_SIZE]
        signal = extract_piece_signal(patch)

        # Save as temporary filename (manually copy to correct name later)
        fname = f"candidate_r{r}_c{c}_aw{int(aw)}_h{hw}.png"
        out = os.path.join(TEMPLATES_DIR, fname)
        cv2.imwrite(out, signal)
        saved[(r, c)] = fname

    print(f"Total {len(saved)} patches successfully saved.")
    print(f"\n-> Check the PNG files in data/templates/ directory")
    print("  If a piece is correct, copy or rename it using the command below:")
    print("    cp data/templates/<candidate_file> data/templates/pawn_2.png")
    print("    cp data/templates/<candidate_file> data/templates/knight_3.png")
    print("    ... etc.")


if __name__ == "__main__":
    main()
