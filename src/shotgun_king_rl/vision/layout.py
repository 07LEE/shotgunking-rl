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

    colored = (
        (image.max(axis=2).astype(int) - image.min(axis=2).astype(int) > 8)
        & (image.max(axis=2) > 20)
    )
    rows = np.flatnonzero(colored.sum(axis=1) > width * 0.5)
    cols = np.flatnonzero(colored.sum(axis=0) > height * 0.5)
    if len(rows) and len(cols):
        crop = [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]
        if _near_target_ratio(crop):
            return crop

    mask = (image.max(axis=2) > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, crop_width, crop_height = cv2.boundingRect(contour)
        crop = [x, y, x + crop_width, y + crop_height]
        if crop_width > width * 0.5 and _near_target_ratio(crop):
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
