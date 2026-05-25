"""Chess grid segmentation and state analysis module for Shotgun King.

This module processes captured game screens, segments the 8x8 chessboard,
and extracts a clean numerical state matrix representing piece positions.
"""

import os

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# Piece template paths and cache
# If template file is missing, fallback to geometry classification
TEMPLATES_DIR = "data/templates"
_TEMPLATES = {}   # {piece_id: list of 45x45 uint8 grayscale signals}
_TEMPLATES_LOADED = False


def _load_templates():
    """Load all matching piece template PNGs from templates directory to allow multi-template matching."""
    global _TEMPLATES, _TEMPLATES_LOADED
    if _TEMPLATES_LOADED:
        return
    _TEMPLATES_LOADED = True
    if cv2 is None or np is None:
        return

    # Initialize lists for each piece ID
    for piece_id in range(1, 7):
        _TEMPLATES[piece_id] = []

    if not os.path.exists(TEMPLATES_DIR):
        return

    try:
        # Scan directory for all PNG files matching patterns like 'pawn_2*.png'
        for filename in os.listdir(TEMPLATES_DIR):
            if not filename.endswith(".png"):
                continue
            # Parse piece_id safely by splitting filename with underscore (e.g. 'bishop_4_1.png' -> parts[1] is '4')
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 2:
                try:
                    piece_id = int(parts[1])
                    if 1 <= piece_id <= 6:
                        path = os.path.join(TEMPLATES_DIR, filename)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            _TEMPLATES[piece_id].append(cv2.resize(img, (45, 45)))
                except ValueError:
                    pass
        # for pid in range(1, 7):
        #     print(f"Piece ID {pid}: loaded {len(_TEMPLATES[pid])} templates")
    except Exception as e:
        print(f"Failed to load templates: {e}")


def _extract_signal(patch):
    """Extract 45x45 piece signal by subtracting corner background from 65x65 patch.

    Removes background tile colors to emphasize the piece silhouette.
    Works identically on both light and dark tiles.

    Args:
        patch: 65x65 BGR image patch.

    Returns:
        45x45 uint8 grayscale signal (piece=bright, background=dark).
    """
    cropped = patch[10:55, 10:55]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(float)
    corners = [
        gray[:4, :4], gray[:4, -4:],
        gray[-4:, :4], gray[-4:, -4:],
    ]
    bg_mean = np.mean([c.mean() for c in corners])
    signal = np.abs(gray - bg_mean)
    max_val = signal.max()
    # print(f"  [DEBUG extract] max_val: {max_val}")
    if max_val > 20.0:
        signal = (signal / max_val * 255.0).clip(0, 255).astype(np.uint8)
    else:
        signal = np.zeros((45, 45), dtype=np.uint8)
    return signal


import json

# Board ROI cache (prevents recalculation on every frame)
_BOARD_ROI = None   # (y1, y2, x1, x2) or None
_ROI_FILE = "data/board_roi.json"  # Disk cache file


def _save_roi(roi):
    """Save ROI to disk to reuse in future runs."""
    try:
        os.makedirs("data", exist_ok=True)
        with open(_ROI_FILE, "w") as f:
            json.dump(list(roi), f)
    except Exception:
        pass


def _load_roi():
    """Load ROI disk cache. Return None if not found."""
    try:
        if os.path.exists(_ROI_FILE):
            with open(_ROI_FILE) as f:
                data = json.load(f)
            return tuple(int(v) for v in data)
    except Exception:
        pass
    return None


def _detect_board_roi(img):
    """Automatically detect chessboard ROI from green grid lines active range.

    The active range (span) of green grid lines represents 7 interior cell intervals.
    Thus cell_size = span / 7, and the full board = 8 * cell_size.

    ROI calculation:
        y1 = y_active.min  (top border position)
        y2 = y1 + 8 * cell_size
        x1 = x_active.min - cell_size  (include leftmost column)
        x2 = x1 + 8 * cell_size

    Returns:
        (y1, y2, x1, x2) or None if detection fails.
    """
    h, w = img.shape[:2]

    b, g, r = cv2.split(img)
    green_mask = ((g.astype(int) - b.astype(int) > 30) &
                  (g.astype(int) - r.astype(int) > 30) &
                  (g > 80)).astype(np.uint8) * 255

    hor_proj = green_mask.sum(axis=1).astype(float)
    ver_proj = green_mask.sum(axis=0).astype(float)

    hor_max = hor_proj.max()
    ver_max = ver_proj.max()
    if hor_max < 1 or ver_max < 1:
        return None

    # Active range of green grid (using 0.25 threshold to remove noise)
    hor_active = np.where(hor_proj > hor_max * 0.25)[0]
    ver_active = np.where(ver_proj > ver_max * 0.25)[0]
    if len(hor_active) < 2 or len(ver_active) < 2:
        return None

    y_min = int(hor_active.min())
    x_min = int(ver_active.min())
    span_h = int(hor_active.max() - y_min)
    span_w = int(ver_active.max() - x_min)

    # Detected span = 7 interior intervals -> cell_size = span / 7
    cell_h = round(span_h / 7)
    cell_w = round(span_w / 7)

    # Validate reasonable cell size (approx 55~90px based on 1280x720)
    if not (50 <= cell_h <= 100 and 50 <= cell_w <= 100):
        return None

    # ROI = 8 * cell_size (extend 1 cell to the left and bottom)
    y1 = max(0, y_min)
    y2 = min(h, y1 + 8 * cell_h)
    x1 = max(0, x_min - cell_w)   # Include first column
    x2 = min(w, x1 + 8 * cell_w)

    if (y2 - y1) < 100 or (x2 - x1) < 100:
        return None

    return y1, y2, x1, x2


def crop_chessboard(img):
    """Crops the primary chessboard area from the full 1280x720 game screen.

    Uses calibrated fixed coordinates for consistent 8x8 slicing.

    Args:
        img: A numpy array representing the 1280x720 BGR image.

    Returns:
        A resized 520x520 BGR image of the chessboard, or None if crop fails.
    """
    if img is None:
        return None

    height, width = img.shape[:2]
    if height != 720 or width != 1280:
        img = cv2.resize(img, (1280, 720))

    try:
        # Enforce calibrated fixed coordinates to prevent miscalibration due to restricted green line range
        y1, y2, x1, x2 = 119, 631, 383, 895
        chessboard_roi = img[y1:y2, x1:x2]

        # Resize to exactly 520x520 for robust 8x8 slicing (520 / 8 = 65 pixels per cell)
        resized_board = cv2.resize(chessboard_roi, (520, 520))
        return resized_board
    except Exception as e:
        print(f"Failed to crop chessboard: {e}")
        return None


def classify_patch(patch):
    """Classifies a single 65x65 cell patch on the board to identify its piece.

    Prioritize template matching, fallback to geometry-based logic if templates are missing.

    Args:
        patch: A 65x65 BGR image representing a single cell.

    Returns:
        An integer representing the cell state:
        0: Empty, 1: Black King (Player),
        2: Pawn, 3: Knight, 4: Bishop, 5: Rook, 6: Queen/King.
    """
    if patch is None or cv2 is None or np is None:
        return 0

    try:
        _load_templates()

        cropped = patch[10:55, 10:55]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # -- 1. Player King detection (dark piece) --
        if 1 in _TEMPLATES and _TEMPLATES[1]:
            signal = _extract_signal(patch)
            patch_f = signal.astype(np.float32)
            max_score = -1.0
            for tmpl in _TEMPLATES[1]:
                res = cv2.matchTemplate(
                    patch_f,
                    tmpl.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
                score = float(res[0][0])
                if score > max_score:
                    max_score = score
            if max_score >= 0.55:
                # Color Guard: Player King is dark, reject if the detected region is too bright (white pieces)
                piece_pixels = gray[signal > 100]
                if len(piece_pixels) > 0 and np.mean(piece_pixels) < 110.0:
                    return 1
        else:
            # geometry fallback: based on dark contour area
            _, thresh_black = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            contours_black, _ = cv2.findContours(
                thresh_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours_black:
                area_black = cv2.contourArea(
                    max(contours_black, key=cv2.contourArea)
                )
                if area_black > 50.0:
                    return 1

        # Check signal strength to bypass empty tiles regardless of background tile color
        signal = _extract_signal(patch)
        # print(f"  [DEBUG check] signal max: {signal.max()}")
        if signal.max() == 0:
            return 0

        # Use background-subtracted signal image for contour extraction to eliminate tile color interference
        _, thresh_white = cv2.threshold(signal, 100, 255, cv2.THRESH_BINARY)
        contours_white, _ = cv2.findContours(
            thresh_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours_white:
            return 0

        c_white = max(contours_white, key=cv2.contourArea)
        area_white = cv2.contourArea(c_white)

        # Empty tile is already filtered by signal.max() == 0, check only for small noise
        if area_white < 30.0:    # noise
            return 0

        # -- 3. Template matching (loaded pieces only) --
        has_any_template = any(len(_TEMPLATES[piece_id]) > 0 for piece_id in range(2, 7))
        if has_any_template:
            signal = _extract_signal(patch)
            patch_f = signal.astype(np.float32)

            best_score = -1.0
            best_piece = 6  # fallback to Queen if matching fails

            for piece_id in range(2, 7):
                templates = _TEMPLATES[piece_id]
                for tmpl in templates:
                    res = cv2.matchTemplate(
                        patch_f,
                        tmpl.astype(np.float32),
                        cv2.TM_CCOEFF_NORMED,
                    )
                    score = float(res[0][0])
                    if score > best_score:
                        best_score = score
                        best_piece = piece_id

            if best_score >= 0.50:      # confidence threshold
                # print(f"  [Template Match] Found piece {best_piece} with score {best_score:.3f}")
                return best_piece
            # else:
            #     print(f"  [Template Match Failed] Best guess was {best_piece} with score {best_score:.3f}. Fallback to geometry.")

        # -- 4. Geometry fallback (missing templates or low confidence) --
        x, y, w, h = cv2.boundingRect(c_white)

        # [GUARD] scanline noise: Empty tile artifact if h is too small
        if h <= 5:
            return 0

        # [GUARD] Empty light/dark tile artifact if width is almost cell size
        if w >= 40 and area_white < 300:
            return 0

        M = cv2.moments(c_white)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + w / 2.0
        bias = abs(cx - (x + w / 2.0))

        # [FIX] check bias (Knight asymmetry) before height threshold
        if h <= 25:
            return 2    # Pawn
        elif bias >= 2.0:
            return 3    # Knight (asymmetric horse head)
        elif h >= 40:
            return 6    # Queen / King (very large piece)
        elif w / float(h) >= 1.1:
            return 5    # Rook (wide and flat castle shape)
        elif 26 <= h <= 39:
            return 4    # Bishop (symmetric medium height)
        else:
            return 6    # safe fallback

    except Exception as e:
        print(f"Failed to classify patch: {e}")
        return 0


def get_state_matrix(img):
    """Segments the chessboard image and returns an 8x8 integer state matrix.

    Args:
        img: Full 1280x720 BGR game screen screenshot.

    Returns:
        An 8x8 numpy array containing state values (0: Empty, 1: Player, 2: Enemy).
    """
    state_matrix = np.zeros((8, 8), dtype=int)
    
    board_img = crop_chessboard(img)
    if board_img is None:
        print("Error: Could not crop chessboard for state extraction.")
        return state_matrix

    # Slicing width and height (520 / 8 = 65)
    cell_size = 65

    for row in range(8):
        for col in range(8):
            y_start = row * cell_size
            y_end = y_start + cell_size
            x_start = col * cell_size
            x_end = x_start + cell_size
            
            patch = board_img[y_start:y_end, x_start:x_end]
            state_matrix[row, col] = classify_patch(patch)
            
    return state_matrix


def extract_ammo_count(img):
    """Analyze UI pixels at top-left of screenshot to return real-time loaded and reserve ammo count.

    Args:
        img: 1280x720 BGR image.

    Returns:
        A tuple containing (loaded_ammo, reserve_ammo).
    """
    if img is None or cv2 is None or np is None:
        return 2, 8

    try:
        height, width, _ = img.shape
        if height != 720 or width != 1280:
            img = cv2.resize(img, (1280, 720))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Calculate loaded ammo (Y: 52~72)
        loaded = 0
        for i in range(10):
            start_x = 384 + i * 16
            end_x = start_x + 8
            
            red_pixels = 0
            for y in range(52, 72):
                for x in range(start_x, end_x):
                    r, g, b = rgb[y, x]
                    if r > 180 and g < 60 and b < 80:
                        red_pixels += 1
            if red_pixels >= 15:
                loaded += 1
            else:
                break

        # 2. Calculate reserve ammo (Y: 89~108)
        reserve = 0
        for i in range(20):
            start_x = 384 + i * 16
            end_x = start_x + 8
            
            red_pixels = 0
            for y in range(89, 108):
                for x in range(start_x, end_x):
                    r, g, b = rgb[y, x]
                    if r > 180 and g < 60 and b < 80:
                        red_pixels += 1
            if red_pixels >= 15:
                reserve += 1
            else:
                break

        return max(0, loaded), max(0, reserve)
    except Exception as e:
        print(f"Failed to extract ammo count: {e}")
        return 2, 8


def check_retry_popup(img):
    """Checks whether the captured screen contains the retry/game-over popup.

    It uses pixel statistics (variance and mean) in target button regions to detect
    if the game-over 'YES / NO' popup is currently active.

    Args:
        img: Full 1280x720 BGR game screen screenshot.

    Returns:
        True if the retry popup is active, False otherwise.
    """
    if img is None or cv2 is None or np is None:
        return False

    try:
        height, width, _ = img.shape
        if height != 720 or width != 1280:
            img = cv2.resize(img, (1280, 720))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Left button region (X: 500-580, Y: 380-440)
        left_patch = gray[380:440, 500:580]
        left_var = np.var(left_patch)

        # 2. Right button region (X: 700-780, Y: 380-440)
        right_patch = gray[380:440, 700:780]
        right_var = np.var(right_patch)

        # 3. Background center region (X: 620-660, Y: 340-380)
        center_patch = gray[340:380, 620:660]
        center_mean = np.mean(center_patch)

        # Active popup characteristics: High variance in button regions, dark background
        if left_var > 2000.0 and right_var > 2000.0 and center_mean < 40.0:
            return True

        return False
    except Exception as e:
        print(f"Failed to check retry popup: {e}")
        return False


if __name__ == "__main__":
    print("Testing chessboard grid segmentation and analysis...")
    screenshot_path = "data/screenshot.png"
    
    if cv2 is not None and np is not None:
        if os.path.exists(screenshot_path):
            img_data = cv2.imread(screenshot_path)
            print("Loaded screenshot, extracting state matrix...")
            
            state = get_state_matrix(img_data)
            loaded, reserve = extract_ammo_count(img_data)
            print(f"Detected Ammo Info -> Loaded: {loaded}, Reserve: {reserve}")
            
            print("\n=== DETECTED SHOTGUN KING CHESSBOARD STATE MATRIX (8x8) ===")
            print("Values: 0 = Empty, 1 = Black King (Player), 2 = White Piece (Enemy)")
            print("=========================================================")
            for row_idx in range(8):
                row_str = "  ".join(str(val) for val in state[row_idx])
                print(f"Row {row_idx + 1}:  [{row_str}]")
            print("=========================================================")
            
            # Save visual grid division for manual inspection/debugging
            board = crop_chessboard(img_data)
            if board is not None:
                # Draw white grid division lines
                for idx in range(1, 8):
                    coord = idx * 65
                    cv2.line(board, (coord, 0), (coord, 520), (255, 255, 255), 1)
                    cv2.line(board, (0, coord), (520, coord), (255, 255, 255), 1)
                
                debug_output = "data/grid_sliced_debug.png"
                cv2.imwrite(debug_output, board)
                print(f"Saved visual slicing grid for debugging: {debug_output}")
        else:
            print(f"Error: Screenshot {screenshot_path} not found. Please run capture.py first.")
    else:
        print("Error: cv2 or numpy is not installed.")
