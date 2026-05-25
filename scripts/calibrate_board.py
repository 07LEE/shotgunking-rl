"""
Automatically detect chessboard boundaries to calibrate crop_chessboard coordinates.

Analyzes checkerboard alternating patterns (mint/cream tiles vs brown/red tiles)
using column average brightness variance to find left/right boundaries,
and row average brightness variance to find top/bottom boundaries.

Usage:
    PYTHONPATH=src uv run python scripts/calibrate_board.py
"""
import cv2
import numpy as np
import os


SCREENSHOT_PATH = "data/screenshot.png"
DEBUG_OUT = "data/board_calibration_debug.png"


def find_board_roi(img):
    """Automatically detect chessboard ROI and return (y1, y2, x1, x2).

    Strategy:
    1. Detect green border lines to find board inner boundaries
    2. Calculate cell size from detected inner boundaries and expand to outer boundaries
    3. Return optimal ROI completing the 8x8 grid.
    """
    h, w = img.shape[:2]

    # -- Green Line Detection --
    b, g, r = cv2.split(img)
    green_mask = (g.astype(int) - b.astype(int) > 30) & \
                 (g.astype(int) - r.astype(int) > 30) & \
                 (g > 80)
    green_mask = green_mask.astype(np.uint8) * 255

    hor_proj = green_mask.sum(axis=1).astype(float)
    ver_proj = green_mask.sum(axis=0).astype(float)

    hor_thresh = max(hor_proj.max() * 0.3, 1)
    ver_thresh = max(ver_proj.max() * 0.3, 1)

    hor_lines = np.where(hor_proj > hor_thresh)[0]
    ver_lines = np.where(ver_proj > ver_thresh)[0]

    if len(hor_lines) < 2 or len(ver_lines) < 2:
        # Fallback: variance-based
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_std = np.convolve(np.std(gray, axis=0),
                              np.ones(15)/15, mode='same')
        row_std = np.convolve(np.std(gray, axis=1),
                              np.ones(15)/15, mode='same')
        c_active = np.where(col_std > col_std.max() * 0.5)[0]
        r_active = np.where(row_std > row_std.max() * 0.5)[0]
        if len(c_active) == 0 or len(r_active) == 0:
            return None
        y1_inner, y2_inner = r_active.min(), r_active.max()
        x1_inner, x2_inner = c_active.min(), c_active.max()
    else:
        y1_inner, y2_inner = hor_lines.min(), hor_lines.max()
        x1_inner, x2_inner = ver_lines.min(), ver_lines.max()

    print(f"[Inner Boundary Detected] y:{y1_inner}~{y2_inner}, x:{x1_inner}~{x2_inner}")

    # -- Expand to Outer Boundaries by Estimating Cell Size --
    # Inner boundaries correspond to 7x7 cells (green lines between cells) or 8x8 cells.
    # Estimate cell size from inner width/height since actual board is 8x8.
    inner_w = x2_inner - x1_inner
    inner_h = y2_inner - y1_inner

    # Divide by 8 or 7 to select more accurate value
    cell_w8 = inner_w / 8.0
    cell_h8 = inner_h / 8.0

    # Actual cell size (integer)
    cw = round(cell_w8)
    ch = round(cell_h8)

    print(f"[Estimated Cell Size] {cw}w x {ch}h px")

    # Check if inner boundaries already cover 8 cells.
    # Since x1_inner is the first inner green line position,
    # board starts at x1_inner - cw (one more cell to the left).
    x1 = max(0, x1_inner - cw)
    x2 = min(w, x1_inner + 8 * cw)
    y1 = max(0, y1_inner)
    y2 = min(h, y1_inner + 8 * ch)

    # Verification: Check if size is reasonable (between 200 and 800px)
    if not (200 < x2-x1 < 800 and 200 < y2-y1 < 800):
        # Assume inner boundary already includes 8 cells
        x1, x2 = x1_inner, x2_inner
        y1, y2 = y1_inner, y2_inner

    print(f"[Final ROI] y:{y1}~{y2}, x:{x1}~{x2}")
    print(f"  -> Board size: {x2-x1} x {y2-y1}")
    return y1, y2, x1, x2


def main():
    if not os.path.exists(SCREENSHOT_PATH):
        print(f"Screenshot not found: {SCREENSHOT_PATH}")
        return

    img = cv2.imread(SCREENSHOT_PATH)
    if img is None:
        print("Image load failed")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    result = find_board_roi(img)
    if result is None:
        print("Board boundary detection failed")
        return

    y1, y2, x1, x2 = result

    # Draw detected boundaries on image
    debug = img.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 8x8 grid overlay
    bw = x2 - x1
    bh = y2 - y1
    cell_w = bw / 8
    cell_h = bh / 8
    for i in range(1, 8):
        cx = int(x1 + i * cell_w)
        cy = int(y1 + i * cell_h)
        cv2.line(debug, (cx, y1), (cx, y2), (255, 0, 0), 1)
        cv2.line(debug, (x1, cy), (x2, cy), (255, 0, 0), 1)

    cv2.imwrite(DEBUG_OUT, debug)
    print(f"\nDebug image saved: {DEBUG_OUT}")

    # crop_chessboard Modification Proposal
    print("\n=== crop_chessboard Modification Proposal ===")
    print(f"  chessboard_roi = img[{y1}:{y2}, {x1}:{x2}]")
    print(f"  (Current code: img[120:640, 380:900])")
    print()
    print(f"Cell size: {cell_w:.1f}w x {cell_h:.1f}h px -> 65px after resize to 520")

if __name__ == "__main__":
    main()
