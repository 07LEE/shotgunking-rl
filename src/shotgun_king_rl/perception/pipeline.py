"""Gate independent screen, board, and HUD readers before policy use."""

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np

from .types import AmmoReading, FusedPerception, Prediction, ScreenKind


T = TypeVar("T")


class Reader(Protocol[T]):
    """Interface implemented by a rule, template, or learned reader."""

    def read(self, image: Any) -> Prediction[T]: ...


@dataclass
class PerceptionPipeline:
    """Run only readers relevant to the accepted top-level screen kind."""

    screen_reader: Reader[ScreenKind]
    board_reader: Reader[np.ndarray]
    ammo_reader: Reader[AmmoReading]

    def observe(self, image: Any) -> FusedPerception:
        screen = self.screen_reader.read(image)
        if not screen.is_accepted:
            return FusedPerception(
                screen_kind=None,
                board=None,
                ammo=None,
                ready_for_policy=False,
                issues=(_issue("screen", screen),),
                sources=(screen.source,),
            )

        if screen.value is not ScreenKind.GAMEPLAY:
            return FusedPerception(
                screen_kind=screen.value,
                board=None,
                ammo=None,
                ready_for_policy=False,
                issues=(f"screen:{screen.value.value}",),
                sources=(screen.source,),
            )

        board = self.board_reader.read(image)
        ammo = self.ammo_reader.read(image)
        issues = []
        accepted_board = None
        accepted_ammo = None

        if board.is_accepted and _valid_board(board.value):
            accepted_board = board.value
        elif board.is_accepted:
            issues.append("board:invalid_shape_or_values")
        else:
            issues.append(_issue("board", board))

        if ammo.is_accepted and _valid_ammo(ammo.value):
            accepted_ammo = ammo.value
        elif ammo.is_accepted:
            issues.append("ammo:invalid_values")
        else:
            issues.append(_issue("ammo", ammo))

        return FusedPerception(
            screen_kind=screen.value,
            board=accepted_board,
            ammo=accepted_ammo,
            ready_for_policy=not issues,
            issues=tuple(issues),
            sources=(screen.source, board.source, ammo.source),
        )


def _issue(name: str, prediction: Prediction[Any]) -> str:
    detail = prediction.reason or prediction.status.value
    return f"{name}:{detail}"


def _valid_board(board: np.ndarray | None) -> bool:
    if not isinstance(board, np.ndarray) or board.shape != (8, 8):
        return False
    if not np.issubdtype(board.dtype, np.number):
        return False
    return bool(np.all(np.isfinite(board)) and np.all(board >= 0))


def _valid_ammo(ammo: AmmoReading | None) -> bool:
    if ammo is None:
        return False
    return (
        isinstance(ammo.loaded, int)
        and not isinstance(ammo.loaded, bool)
        and isinstance(ammo.reserve, int)
        and not isinstance(ammo.reserve, bool)
        and ammo.loaded >= 0
        and ammo.reserve >= 0
    )
