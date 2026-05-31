"""Chessboard segmentation utility to slice the board into 8x8 individual cells.

Saves each sliced cell image under data/debug_cells/ and outputs the real-time
classification matrix for easy visual verification of vision detection accuracy.
"""

import os
import sys

# Ensure src path is importable
project_root = "/home/lee/Documents/pl"
sys.path.append(os.path.join(project_root, "src"))

try:
    import cv2
    import numpy as np
    from analyzer import crop_chessboard, classify_patch
except ImportError as e:
    print(f"Failed to import dependencies: {e}")
    sys.exit(1)


def main():
    screenshot_path = os.path.join(project_root, "data/screenshot.png")
    if not os.path.exists(screenshot_path):
        print(f"Screenshot file '{screenshot_path}' not found. Please run environment first to capture.")
        return

    img = cv2.imread(screenshot_path)
    if img is None:
        print("Failed to read screenshot image.")
        return

    # Extract 8x8 board grid
    board_img = crop_chessboard(img)
    if board_img is None:
        print("Failed to crop chessboard.")
        return

    output_dir = os.path.join(project_root, "data/debug_cells")
    os.makedirs(output_dir, exist_ok=True)

    cell_size = 65
    state_matrix = np.zeros((8, 8), dtype=int)
    piece_names = {
        0: "Empty", 1: "Black King (Player)",
        2: "Pawn", 3: "Knight", 4: "Bishop", 5: "Rook", 6: "Queen/King"
    }

    print("\n=== Slicing Chessboard & Classifying Cells ===")
    for r in range(8):
        for c in range(8):
            y_start = r * cell_size
            y_end = y_start + cell_size
            x_start = c * cell_size
            x_end = x_start + cell_size
            
            patch = board_img[y_start:y_end, x_start:x_end]
            cell_val = classify_patch(patch)
            state_matrix[r, c] = cell_val

            # Save individual cell BGR patch
            cell_path = os.path.join(output_dir, f"cell_{r}_{c}.png")
            cv2.imwrite(cell_path, patch)

    print("\n=== DETECTED SHOTGUN KING CHESSBOARD STATE MATRIX (8x8) ===")
    print("Values: 0:Empty, 1:Player King, 2:Pawn, 3:Knight, 4:Bishop, 5:Rook, 6:Queen/King")
    print("=========================================================")
    for r in range(8):
        row_str = "  ".join(str(state_matrix[r, c]) for c in range(8))
        print(f"Row {r + 1} (Index {r}):  [{row_str}]")
    print("=========================================================")

    print("\n=== Piece Location Audit ===")
    for r in range(8):
        for c in range(8):
            val = state_matrix[r, c]
            if val != 0:
                print(f"Cell ({r}, {c}) -> Type {val}: {piece_names[val]}")

    print(f"\nSuccessfully segmented and saved all 64 cell patches to: '{output_dir}'")


if __name__ == "__main__":
    main()
