"""Gate independent screen, board, and HUD readers before policy use."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

import numpy as np

from ..state_schema import SUPPORTED_BOARD_IDS
from .frame import PreparedFrame, read_frame
from .validation import valid_combat_stats
from .context import encode_sided_board, valid_board_context

from .types import (
    AmmoReading,
    BoardContextReading,
    CombatStatsReading,
    FusedPerception,
    Prediction,
    ScreenKind,
)


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
    combat_stats_reader: Reader[CombatStatsReading]
    context_reader: Reader[BoardContextReading] | None = None

    def observe_live(
        self,
        capture: Callable[[], Any],
        clear_hover: Callable[[], bool],
        wait_for_tooltip: Callable[[], None],
    ) -> FusedPerception:
        """Retry one gameplay HUD failure after clearing a possible hover overlay.

        Capture must return a fresh image, or None on failure. The caller owns
        desktop permissions, pointer placement and the tooltip dismissal delay.
        Offline observe never moves the pointer. No values survive a recapture.
        """
        image = capture()
        if image is None:
            return FusedPerception(
                screen_kind=None, board=None, ammo=None, combat_stats=None,
                ready_for_policy=False, issues=("recovery:capture_failed",), sources=(),
            )
        result = self.observe(image)
        if result.screen_kind is not ScreenKind.GAMEPLAY or not any(
            issue.startswith("combat_stats:") for issue in result.issues
        ):
            return result
        if not clear_hover():
            return _recovery_failure(result, "hover_clear_failed")
        wait_for_tooltip()
        image = capture()
        if image is None:
            return _recovery_failure(result, "recapture_failed")
        return self.observe(image)

    def observe(self, image: Any) -> FusedPerception:
        frame = PreparedFrame(image)
        screen = read_frame(self.screen_reader, frame)
        if not screen.is_accepted:
            return FusedPerception(
                screen_kind=None,
                board=None,
                ammo=None,
                combat_stats=None,
                ready_for_policy=False,
                issues=(_issue("screen", screen),),
                sources=(screen.source,),
            )

        if screen.value is not ScreenKind.GAMEPLAY:
            return FusedPerception(
                screen_kind=screen.value,
                board=None,
                ammo=None,
                combat_stats=None,
                ready_for_policy=False,
                issues=(f"screen:{screen.value.value}",),
                sources=(screen.source,),
            )

        board = read_frame(self.board_reader, frame)
        ammo = read_frame(self.ammo_reader, frame)
        combat_stats = read_frame(self.combat_stats_reader, frame)
        issues = []
        accepted_board = None
        accepted_ammo = None
        accepted_combat_stats = None
        accepted_context = None

        if board.is_accepted and _valid_board(board.value):
            accepted_board = board.value
        elif board.is_accepted:
            issues.append("board:invalid_shape_or_values")
        else:
            issues.append(_issue("board", board))

        context_source = "board-context:unavailable"
        if accepted_board is not None:
            context = (read_frame(self.context_reader, frame) if self.context_reader is not None
                       else Prediction.unknown(source=context_source, reason="reader_unavailable"))
            context_source = context.source
            if context.is_accepted and valid_board_context(context.value, accepted_board):
                accepted_context = BoardContextReading(context.value.piece_sides.copy(), context.value.moat_active)
                accepted_board = encode_sided_board(accepted_board, accepted_context)
            elif context.is_accepted:
                issues.append("board_context:invalid_values")
            else:
                issues.append(_issue("board_context", context))

        if ammo.is_accepted and _valid_ammo(ammo.value):
            accepted_ammo = ammo.value
        elif ammo.is_accepted:
            issues.append("ammo:invalid_values")
        else:
            issues.append(_issue("ammo", ammo))

        if combat_stats.is_accepted and valid_combat_stats(combat_stats.value):
            accepted_combat_stats = combat_stats.value
        elif combat_stats.is_accepted:
            issues.append("combat_stats:invalid_values")
        else:
            issues.append(_issue("combat_stats", combat_stats))

        return FusedPerception(
            screen_kind=screen.value,
            board=accepted_board,
            ammo=accepted_ammo,
            combat_stats=accepted_combat_stats,
            ready_for_policy=not issues,
            issues=tuple(issues),
            sources=(screen.source, board.source, ammo.source, combat_stats.source, context_source),
            board_context=accepted_context,
        )


def _issue(name: str, prediction: Prediction[Any]) -> str:
    detail = prediction.reason or prediction.status.value
    return f"{name}:{detail}"


def _recovery_failure(result: FusedPerception, reason: str) -> FusedPerception:
    return FusedPerception(
        screen_kind=result.screen_kind,
        board=None,
        ammo=None,
        combat_stats=None,
        ready_for_policy=False,
        issues=(*result.issues, f"recovery:{reason}"),
        sources=result.sources,
        board_context=result.board_context,
    )


def _valid_board(board: np.ndarray | None) -> bool:
    if not isinstance(board, np.ndarray) or board.shape != (8, 8):
        return False
    if not (np.issubdtype(board.dtype, np.integer) or np.issubdtype(board.dtype, np.floating)):
        return False
    return bool(np.all(np.isfinite(board)) and np.all(np.isin(board, SUPPORTED_BOARD_IDS)))


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
