"""Normalize captured game windows without changing their originals."""

import cv2
import numpy as np


def detect_crop(image):
    if image.shape[:2] == (720, 1280):
        return [0, 0, 1280, 720]
    colored = (image.max(axis=2).astype(int) - image.min(axis=2).astype(int) > 8) & (image.max(axis=2) > 20)
    rows = np.flatnonzero(colored.sum(axis=1) > image.shape[1] * .5)
    cols = np.flatnonzero(colored.sum(axis=0) > image.shape[0] * .5)
    if len(rows) and len(cols):
        crop = [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]
        if abs((crop[2] - crop[0]) / (crop[3] - crop[1]) - 16 / 9) < .08:
            return crop
    mask = (image.max(axis=2) > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > image.shape[1] * .5 and abs(width / height - 16 / 9) < .08:
            candidates.append((width * height, [x, y, x + width, y + height]))
    if not candidates:
        raise ValueError("Game area not detected; supply --crop LEFT TOP RIGHT BOTTOM")
    return max(candidates)[1]


def normalize(image, crop):
    if not isinstance(crop, list) or len(crop) != 4 or any(type(value) is not int for value in crop):
        raise ValueError("Crop must contain four integer coordinates")
    left, top, right, bottom = crop
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        raise ValueError("Crop exceeds image bounds")
    return cv2.resize(image[top:bottom, left:right], (1280, 720), interpolation=cv2.INTER_AREA)
