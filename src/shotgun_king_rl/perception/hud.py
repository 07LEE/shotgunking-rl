"""Shared crop alignment for the calibrated gameplay stats panel."""

import cv2
import numpy as np


def stat_patch(image, row_offset=0):
    """Locate a stat relative to the first number in the shifting HUD panel.

    Keep its title as context so another stat cannot match by its digit alone.
    Coordinates refer to a normalized 1280x720 game viewport.
    """
    region = image[180:335, 290:340]
    bright = region.max(axis=2) > 150
    rows = np.flatnonzero(bright.sum(axis=1) >= 2)
    if not len(rows):
        return None
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 6) + 1)
    for group in groups:
        top, bottom = int(group[0]), int(group[-1]) + 1
        if bottom - top < 10 or np.count_nonzero(bright[top:bottom]) < 20:
            continue
        y = 180 + top + row_offset
        # Anchor on the number, retaining the title above it and no next stat.
        return cv2.cvtColor(image[y - 35:y + 30, 270:354], cv2.COLOR_BGR2GRAY)
    return None

