"""Shared crop alignment for the calibrated gameplay stats panel."""

import cv2
import numpy as np


def _stat_anchor(image):
    """Find the first numeric row shared by the calibrated stat crops."""
    region = image[180:335, 290:340]
    bright = region.max(axis=2) > 150
    rows = np.flatnonzero(bright.sum(axis=1) >= 2)
    if not len(rows):
        return None
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 6) + 1)
    for group in groups:
        top, bottom = int(group[0]), int(group[-1]) + 1
        if bottom - top >= 10 and np.count_nonzero(bright[top:bottom]) >= 20:
            return 180 + top
    return None


def stat_patch(image, row_offset=0):
    """Locate a stat relative to the first number in the shifting HUD panel.

    Keep its title as context so another stat cannot match by its digit alone.
    Coordinates refer to a normalized 1280x720 game viewport.
    """
    anchor = _stat_anchor(image)
    if anchor is None:
        return None
    y = anchor + row_offset
    return cv2.cvtColor(image[y - 35:y + 30, 270:354], cv2.COLOR_BGR2GRAY)


def stat_context_patch(image):
    """Include the main stats and optional rows when recognizing a reviewed zero."""
    anchor = _stat_anchor(image)
    if anchor is None or anchor - 35 < 0 or anchor + 286 > image.shape[0]:
        return None
    return cv2.cvtColor(image[anchor - 35:anchor + 286, 270:354], cv2.COLOR_BGR2GRAY)


def reviewed_zero_score(image, query, examples, deviations):
    """Require the known row exactly, plus a strong match of its panel context.

    A new optional stat must not disappear inside a high whole-panel score.
    Unseen blank panels and shifted or occluded row patterns stay unknown.
    """
    if query is None or not examples:
        return -1.0
    context = stat_context_patch(image)
    if context is None or context.std() < 5:
        return -1.0
    best = -1.0
    for (template, row), deviation in zip(examples, deviations):
        if (template.shape != context.shape or deviation < 5
                or not np.array_equal(query, row)):
            continue
        score = float(cv2.matchTemplate(context, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        best = max(best, score)
    return best

