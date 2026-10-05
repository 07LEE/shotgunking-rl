"""Top-level popup and card-screen recognition."""

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def check_retry_popup(img):
    """Return whether the retry popup geometry is present."""
    if img is None or cv2 is None or np is None:
        return False

    try:
        height, width = img.shape[:2]
        if height != 720 or width != 1280:
            img = cv2.resize(img, (1280, 720))

        from .analyzer import crop_chessboard

        board = crop_chessboard(img)
        if board is None:
            return False
        board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        if board_gray.mean() > 60.0:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        buttons = []
        for contour in contours:
            x, y, button_width, button_height = cv2.boundingRect(contour)
            if (
                110 <= button_width <= 130
                and 30 <= button_height <= 45
                and 345 <= y <= 385
            ):
                buttons.append((x, y, button_width, button_height))

        if len(buttons) != 2:
            return False
        left, right = sorted(buttons, key=lambda button: button[0])
        left_center = left[0] + left[2] / 2
        right_center = right[0] + right[2] / 2
        return (
            abs(left[1] - right[1]) < 5
            and abs((left_center + right_center) / 2 - 640.0) < 10.0
        )
    except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
        return False


def check_card_selection_screen(img):
    """Return whether all known card-border sample points are bright."""
    if img is None or cv2 is None or np is None:
        return False

    try:
        height, width = img.shape[:2]
        if height != 720 or width != 1280:
            img = cv2.resize(img, (1280, 720))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        card_borders = (
            (220, 545), (220, 619), (328, 545), (328, 619),
            (220, 656), (220, 730), (328, 656), (328, 730),
            (420, 545), (420, 619), (528, 545), (528, 619),
            (420, 656), (420, 730), (528, 656), (528, 730),
        )
        return all(gray[y, x] >= 215 for y, x in card_borders)
    except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
        return False
