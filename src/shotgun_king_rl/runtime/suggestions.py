"""Inspectable suggest-mode summaries built without game input."""

from dataclasses import dataclass

import numpy as np

from .actions import action_destination, select_shot_target


ACTION_NAMES = {
    0: "Move Up-Left",
    1: "Move Up",
    2: "Move Up-Right",
    3: "Move Left",
    4: "Move Right",
    5: "Move Down-Left",
    6: "Move Down",
    7: "Move Down-Right",
    8: "Reload",
    9: "Shoot",
    10: "Jump Up-Left",
    11: "Jump Up",
    12: "Jump Up-Right",
    13: "Jump Left",
    14: "Jump Right",
    15: "Jump Down-Left",
    16: "Jump Down",
    17: "Jump Down-Right",
}

ACTION_KEYS = {
    0: "7",
    1: "8",
    2: "9",
    3: "4",
    4: "6",
    5: "1",
    6: "2",
    7: "3",
    8: "5 or r",
    9: "0",
}

PIECE_MARKS = {
    0: ".",
    1: "K",
    2: "P",
    3: "N",
    4: "B",
    5: "R",
    6: "X",
}

PIECE_NAMES = {
    2: "pawn",
    3: "knight",
    4: "bishop",
    5: "rook",
    6: "queen or white king",
}


@dataclass(frozen=True)
class SuggestedAction:
    """One permitted action with its observable reason and optional Q-value."""

    action: int
    name: str
    key: str
    reason: str
    q_value: float | None


@dataclass(frozen=True)
class SuggestionView:
    """All state details required for a human to inspect one recommendation."""

    board_rows: tuple[str, ...]
    ammo: str
    valid_actions: tuple[SuggestedAction, ...]
    recommendations: tuple[SuggestedAction, ...]


def build_suggestion_view(
    observation,
    action_mask,
    q_values,
    *,
    loaded_ammo,
    max_ammo,
    reserve_ammo,
    range_limit,
    royal_guard_active=False,
    melee_damage=0.0,
    recommendation_count=3,
):
    """Build a stable view from the exact observation used by the policy."""
    board = np.asarray(observation[:64]).reshape(8, 8)
    threat = np.asarray(observation[64:128]).reshape(8, 8)
    king_positions = np.argwhere(board == 1)
    king_position = None
    if len(king_positions) == 1:
        king_position = tuple(int(value) for value in king_positions[0])

    scores = None if q_values is None else np.asarray(q_values, dtype=float)
    valid = []
    for action in np.flatnonzero(np.asarray(action_mask) > 0):
        action = int(action)
        q_value = None
        if scores is not None and action < len(scores) and np.isfinite(scores[action]):
            q_value = float(scores[action])
        valid.append(
            SuggestedAction(
                action=action,
                name=ACTION_NAMES.get(action, f"Action {action}"),
                key=ACTION_KEYS.get(action, str(action)),
                reason=_action_reason(
                    action,
                    board,
                    threat,
                    king_position,
                    loaded_ammo,
                    max_ammo,
                    reserve_ammo,
                    range_limit,
                    royal_guard_active,
                    melee_damage,
                ),
                q_value=q_value,
            )
        )

    ranked = sorted(
        (item for item in valid if item.q_value is not None),
        key=lambda item: item.q_value,
        reverse=True,
    )[:recommendation_count]
    return SuggestionView(
        board_rows=_format_board(board),
        ammo=f"loaded {loaded_ammo}/{max_ammo} | reserve {reserve_ammo}",
        valid_actions=tuple(valid),
        recommendations=tuple(ranked),
    )


def format_suggestion_view(view):
    """Render a suggestion view without presenting Q-values as confidence."""
    lines = ["=== OBSERVED STATE ===", *view.board_rows, f"Ammo: {view.ammo}"]
    lines.append("Valid actions:")
    if view.valid_actions:
        for item in view.valid_actions:
            lines.append(f"  [{item.key}] {item.name}: {item.reason}")
    else:
        lines.append("  none")
    lines.append("Recommendations (model Q-values, not probabilities):")
    lines.append("  Reasons explain validity and targeting rules, not the model's internal logic.")
    if view.recommendations:
        for rank, item in enumerate(view.recommendations, 1):
            lines.append(
                f"  {rank}. [{item.key}] {item.name} | "
                f"Q={item.q_value:.4f} | {item.reason}"
            )
    else:
        lines.append("  unavailable")
    return "\n".join(lines)


def _format_board(board):
    rows = ["    a b c d e f g h"]
    for row in range(8):
        marks = " ".join(PIECE_MARKS.get(int(value), "?") for value in board[row])
        rows.append(f"{8 - row} | {marks}")
    rows.append("Legend: K player | P N B R enemies | X queen/white king | . empty")
    return tuple(rows)


def _action_reason(
    action,
    board,
    threat,
    king_position,
    loaded_ammo,
    max_ammo,
    reserve_ammo,
    range_limit,
    royal_guard_active,
    melee_damage,
):
    if action in range(8) or action in range(10, 18):
        if king_position is None:
            return "king position unavailable"
        destination = action_destination(king_position, action)
        safety = "safe" if threat[destination] == 0 else "threatened; no safe move available"
        return f"{safety} empty square {_square_name(destination)}"
    if action == 8:
        needed = max(0, max_ammo - loaded_ammo)
        return f"reload {min(needed, reserve_ammo)} shell(s) from reserve"
    if action == 9:
        if king_position is None:
            return "king position unavailable"
        target = select_shot_target(
            board,
            king_position,
            royal_guard_active=royal_guard_active,
            range_limit=range_limit,
        )
        if target is None:
            return "target unavailable"
        row = king_position[0] + target.direction[0] * target.distance
        col = king_position[1] + target.direction[1] * target.distance
        piece = PIECE_NAMES.get(target.piece, f"piece {target.piece}")
        if loaded_ammo == 0 and target.distance == 1 and melee_damage > 0:
            return f"melee {piece} on {_square_name((row, col))}"
        priority = "threatening " if target.threatens_king else ""
        return (
            f"target {priority}{piece} on {_square_name((row, col))} "
            f"at {target.effective_distance:.2f} tiles"
        )
    return "permitted by action mask"


def _square_name(position):
    row, col = position
    return f"{chr(ord('a') + col)}{8 - row}"
