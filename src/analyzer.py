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


def crop_chessboard(img):
    """Crops the primary chessboard area from the full 1280x720 game screen.

    Args:
        img: A numpy array representing the 1280x720 BGR image.

    Returns:
        A resized 520x520 BGR image of the chessboard, or None if crop fails.
    """
    if img is None:
        return None

    # Expected dimensions of the screenshot
    height, width, _ = img.shape
    if height != 720 or width != 1280:
        # Auto-resize if it varies slightly to match coordinate grid
        img = cv2.resize(img, (1280, 720))

    try:
        # Primary chessboard region coordinates [ymin:ymax, xmin:xmax]
        # Evaluated precisely from the game screenshot
        chessboard_roi = img[120:640, 380:900]
        
        # Resize to exactly 520x520 for robust 8x8 slicing (520 / 8 = 65 pixels per cell)
        resized_board = cv2.resize(chessboard_roi, (520, 520))
        return resized_board
    except Exception as e:
        print(f"Failed to crop chessboard: {e}")
        return None


def classify_patch(patch):
    """Classifies a single 65x65 cell patch on the board to identify its piece.

    Args:
        patch: A 65x65 BGR image representing a single cell.

    Returns:
        An integer representing the cell state:
        0: Empty, 1: Black King (Player), 2: White Piece (Enemy).
    """
    if patch is None or cv2 is None or np is None:
        return 0

    try:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Focus on the center area of the patch (45x45 pixels) to bypass boundary lines
        center_patch = gray[10:55, 10:55]
        
        # Calculate pixel variance in the center area
        # Empty cells have very low variance (either uniform yellow or uniform purple)
        # Cells with pieces have high variance due to diverse details and outlines
        variance = np.var(center_patch)
        
        # Threshold for detecting presence of any piece
        if variance > 250.0:
            # Detect whether it is the Black King or a White Piece
            # Black King is predominantly dark gray/black with a high amount of low-intensity pixels
            # White pieces have predominantly high-intensity pixels and far fewer dark pixels
            dark_pixels = np.sum(center_patch < 80)
            
            if dark_pixels >= 350:
                # Black King (Player)
                return 1
            else:
                # White Enemy Piece
                return 2
        return 0
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
