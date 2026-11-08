"""Find and normalize the game viewport from differently sized captures."""

import cv2
import numpy as np


TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT


def detect_game_crop(image):
    """Return the 16:9 game viewport inside a captured window."""
    if not _valid_image(image):
        raise ValueError("Image must be a non-empty BGR array")
    height, width = image.shape[:2]
    if (height, width) == (TARGET_HEIGHT, TARGET_WIDTH):
        return [0, 0, width, height]

    channels = cv2.split(image)
    brightest = channels[0]
    darkest = channels[0]
    for channel in channels[1:]:
        brightest = cv2.max(brightest, channel)
        darkest = cv2.min(darkest, channel)
    colored = ((brightest - darkest > 8) & (brightest > 20))
    content_rows = np.flatnonzero(colored.any(axis=1))
    content_cols = np.flatnonzero(colored.any(axis=0))
    content_width = int(content_cols[-1] - content_cols[0] + 1) if len(content_cols) else width
    content_height = int(content_rows[-1] - content_rows[0] + 1) if len(content_rows) else height
    # Portal frames can surround a small window with a much larger black canvas.
    rows = np.flatnonzero(colored.sum(axis=1) > content_width * 0.5)
    cols = np.flatnonzero(colored.sum(axis=0) > content_height * 0.5)
    if len(rows) and len(cols):
        crop = [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]
        if _near_target_ratio(crop):
            return crop

    mask = (brightest > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, crop_width, crop_height = cv2.boundingRect(contour)
        crop = [x, y, x + crop_width, y + crop_height]
        if crop_width > content_width * 0.5 and _near_target_ratio(crop):
            candidates.append((crop_width * crop_height, crop))
    if not candidates:
        raise ValueError("Game area not detected")
    return max(candidates)[1]


def normalize_game_screen(image, crop=None):
    """Crop the game viewport and resize it to the calibrated 1280x720 frame."""
    if not _valid_image(image):
        raise ValueError("Image must be a non-empty BGR array")
    crop = detect_game_crop(image) if crop is None else crop
    if not isinstance(crop, list) or len(crop) != 4 or any(type(value) is not int for value in crop):
        raise ValueError("Crop must contain four integer coordinates")
    left, top, right, bottom = crop
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        raise ValueError("Crop exceeds image bounds")
    return cv2.resize(image[top:bottom, left:right], (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)


def _near_target_ratio(crop):
    left, top, right, bottom = crop
    return abs((right - left) / (bottom - top) - TARGET_RATIO) < 0.08


def _valid_image(image):
    return (
        isinstance(image, np.ndarray)
        and image.ndim == 3
        and image.shape[0] > 0
        and image.shape[1] > 0
        and image.shape[2] >= 3
    )
