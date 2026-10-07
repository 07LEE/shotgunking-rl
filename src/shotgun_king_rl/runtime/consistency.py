"""Pre-action checks that prevent decisions from using stale observations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationConsistency:
    """Differences between the policy observation and the execution screen."""

    consistent: bool
    reason: str
    changed_cells: tuple[tuple[int, int, int, int], ...] = ()
    expected_ammo: tuple[int, int] | None = None
    actual_ammo: tuple[int, int] | None = None

    def as_info(self):
        return {
            "consistent": self.consistent,
            "reason": self.reason,
            "changed_cells": [
                {
                    "square": f"{chr(ord('a') + col)}{8 - row}",
                    "expected": expected,
                    "actual": actual,
                }
                for row, col, expected, actual in self.changed_cells
            ],
            "expected_ammo": (
                {"loaded": self.expected_ammo[0], "reserve": self.expected_ammo[1]}
                if self.expected_ammo is not None
                else None
            ),
            "actual_ammo": (
                {"loaded": self.actual_ammo[0], "reserve": self.actual_ammo[1]}
                if self.actual_ammo is not None
                else None
            ),
        }


def compare_policy_observation(expected, actual):
    """Compare policy-critical board and ammunition immediately before input."""
    if expected is None or actual is None:
        return ObservationConsistency(False, "observation_missing")
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.size < 130 or actual.size < 130:
        return ObservationConsistency(False, "observation_shape_invalid")

    expected_board = expected[:64].reshape(8, 8).astype(int)
    actual_board = actual[:64].reshape(8, 8).astype(int)
    changed = tuple(
        (int(row), int(col), int(expected_board[row, col]), int(actual_board[row, col]))
        for row, col in np.argwhere(expected_board != actual_board)
    )
    expected_ammo = (int(expected[128]), int(expected[129]))
    actual_ammo = (int(actual[128]), int(actual[129]))
    reasons = []
    if changed:
        reasons.append("board_changed")
    if expected_ammo != actual_ammo:
        reasons.append("ammo_changed")
    return ObservationConsistency(
        consistent=not reasons,
        reason="match" if not reasons else "+".join(reasons),
        changed_cells=changed,
        expected_ammo=expected_ammo,
        actual_ammo=actual_ammo,
    )
