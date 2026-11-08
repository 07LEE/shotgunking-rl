"""Explain small boundary fragments with reviewed floors and visible neighbors."""

import cv2
import numpy as np

from ..state_schema import SIDE_EMPTY


class NeighborOverlapReader:
    """Require independent occupancy, floor evidence, and connected sprite pixels."""

    source = 'piece-sides:reviewed-neighbor-overlap-v2'

    def __init__(self, references=()):
        patches = []
        for image, sides in references:
            for row, col in np.argwhere(sides == SIDE_EMPTY):
                patch = image[row*65:(row+1)*65, col*65:(col+1)*65]
                patches.append(patch.copy())
        # Empty labels can include spill from another tile. Keep complete raw
        # reference patches only when they agree everywhere with the robust
        # floor appearance of at least three similarly colored reviewed cells.
        self._empty_floors = np.stack(list({p.tobytes(): p for p in patches}.values())).astype(np.int16) if patches else None
        groups = {}
        for patch in patches:
            key = tuple((np.median(patch[22:43, 22:43], axis=(0, 1)) // 16).astype(int))
            groups.setdefault(key, []).append(patch)
        clean = []
        for group in groups.values():
            if len(group) < 3:
                continue
            pixels = np.stack(group).astype(np.int16)
            median = np.median(pixels, axis=0)
            clean.extend(pixels[np.max(np.abs(pixels-median), axis=(1, 2, 3)) <= 24])
        unique = {patch.tobytes(): patch for patch in clean}
        self._floors = np.stack(list(unique.values())) if unique else None

    def _empty_residuals(self, patch):
        if self._empty_floors is None:
            return
        error = np.abs(self._empty_floors-patch.astype(np.int16)).max(axis=3)
        core = error[:, 22:43, 22:43]
        masks = error > 24
        areas = masks.sum(axis=(1, 2))
        candidates = np.flatnonzero((core.mean((1, 2)) <= 2) & (core.max((1, 2)) <= 24)
                                    & (areas >= 1) & (areas <= 400))
        seen = set()
        for index in sorted(candidates, key=lambda i: (areas[i], error[i].mean())):
            mask = masks[index]
            key = mask.tobytes()
            if key not in seen:
                seen.add(key)
                yield mask

    def _floor(self, patch, *, empty):
        if empty:
            return next(self._empty_residuals(patch), None)
        if self._floors is None:
            return None
        error = np.abs(self._floors-patch.astype(np.int16)).max(axis=3)
        # Occupied neighbors expose floor around the sprite. Require at least
        # one quarter of all pixels to agree with a clean reviewed floor.
        scores = np.quantile(error, .25, axis=(1, 2))
        index = int(scores.argmin())
        if scores[index] > 24:
            return None
        return error[index] > 24

    def accepts_empty(self, image, row, col, board, predictions):
        if board[row, col] != 0:
            return False
        patch = image[row*65:(row+1)*65, col*65:(col+1)*65]
        # Every unexplained component must continue into the central foreground
        # of a separately recognized occupied neighbor. Unknown neighbors cannot
        # bootstrap each other into acceptance.
        canvas = np.zeros((195, 195), np.uint8)
        seeds = np.zeros_like(canvas)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = row + dy, col + dx
                if (dy == dx == 0 or not (0 <= y < 8 and 0 <= x < 8)
                        or board[y, x] == 0 or not predictions[y][x].is_accepted
                        or predictions[y][x].value == SIDE_EMPTY):
                    continue
                neighbor = image[y*65:(y+1)*65, x*65:(x+1)*65]
                foreground = self._floor(neighbor, empty=False)
                if foreground is None:
                    continue
                sy, sx = (dy+1)*65, (dx+1)*65
                canvas[sy:sy+65, sx:sx+65] = foreground
                seeds[sy+22:sy+43, sx+22:sx+43] = foreground[22:43, 22:43]
        # A reviewed empty cell may itself contain a neighbor's head. Selecting
        # just the nearest floor can introduce a disconnected negative ghost.
        # Accept only a complete floor hypothesis explaining every changed pixel.
        for residual in self._empty_residuals(patch):
            pixels = patch[residual].astype(np.int16)
            if np.any((pixels[:, 2]-pixels[:, 1] > 100) & (pixels[:, 2]-pixels[:, 0] > 100)):
                continue
            canvas[65:130, 65:130] = residual
            _, components = cv2.connectedComponents(canvas, connectivity=8)
            supported = set(components[seeds.astype(bool)]) - {0}
            needed = set(components[65:130, 65:130][residual]) - {0}
            if needed and needed <= supported:
                return True
        return False
