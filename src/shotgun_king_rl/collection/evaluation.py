"""Evaluate current perception readers against reviewed collection labels."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .audit import SPLITS
from .images import normalize
from ..perception import OverlayScreenReader, RedPixelAmmoReader, ScreenKind, TemplatePieceReader
from ..perception.frame import PreparedFrame, read_frame
from ..vision import analyzer


ROOT = Path(__file__).resolve().parents[3]
PIECE_LABELS = {
    0: "empty",
    1: "player_king",
    2: "pawn",
    3: "knight",
    4: "bishop",
    5: "rook",
    6: "queen_or_white_king",
    7: "special_knight",
    8: "queen",
    9: "white_king",
}


def _rate(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


class ClassificationMetrics:
    def __init__(self):
        self.expected = Counter()
        self.predicted = Counter()
        self.true_positive = Counter()
        self.confusion = defaultdict(Counter)

    def add(self, expected, predicted):
        self.expected[expected] += 1
        self.predicted[predicted] += 1
        self.confusion[expected][predicted] += 1
        if expected == predicted:
            self.true_positive[expected] += 1

    def report(self):
        labels = sorted(set(self.expected) | set(self.predicted))
        correct = sum(self.true_positive.values())
        total = sum(self.expected.values())
        return {
            "samples": total,
            "accuracy": _rate(correct, total),
            "classes": {
                label: {
                    "support": self.expected[label],
                    "predicted": self.predicted[label],
                    "precision": _rate(self.true_positive[label], self.predicted[label]),
                    "recall": _rate(self.true_positive[label], self.expected[label]),
                }
                for label in labels
            },
            "confusion": {
                expected: dict(sorted(predictions.items()))
                for expected, predictions in sorted(self.confusion.items())
            },
        }


class PieceMetrics:
    def __init__(self):
        self.statuses = Counter()
        self.exact = ClassificationMetrics()
        self.compatible = ClassificationMetrics()
        self.screens = 0
        self.exact_screens = 0
        self.compatible_screens = 0
        self.complete_screens = 0
        self.complete_exact_screens = 0
        self.occupied_cells = 0
        self.occupied_correct = 0
        self.king_expected = 0
        self.king_exactly_one = 0
        self.king_position_correct = 0

    def add(self, expected, prediction, excluded):
        self.screens += 1
        self.statuses[prediction.status.value] += 1
        board = prediction.value
        valid_board = isinstance(board, np.ndarray) and board.shape == (8, 8)
        predicted = [
            [PIECE_LABELS.get(int(board[row, col]), "unknown") if valid_board else "unknown" for col in range(8)]
            for row in range(8)
        ]
        exact_screen = True
        compatible_screen = True
        excluded_set = {tuple(cell) for cell in excluded}
        for row in range(8):
            for col in range(8):
                if (row, col) in excluded_set:
                    continue
                truth = expected[row][col]
                guess = predicted[row][col]
                self.exact.add(truth, guess)
                truth_compatible = _compatible_piece(truth)
                guess_compatible = _compatible_piece(guess)
                self.compatible.add(truth_compatible, guess_compatible)
                exact_screen &= truth == guess
                compatible_screen &= truth_compatible == guess_compatible
                if truth != "empty":
                    self.occupied_cells += 1
                    self.occupied_correct += truth == guess
        self.exact_screens += exact_screen
        self.compatible_screens += compatible_screen
        if not excluded_set:
            self.complete_screens += 1
            self.complete_exact_screens += compatible_screen

        expected_kings = {
            (row, col)
            for row in range(8)
            for col in range(8)
            if expected[row][col] == "player_king" and (row, col) not in excluded_set
        }
        predicted_kings = {
            (row, col)
            for row in range(8)
            for col in range(8)
            if predicted[row][col] == "player_king" and (row, col) not in excluded_set
        }
        if len(expected_kings) == 1:
            self.king_expected += 1
            self.king_exactly_one += len(predicted_kings) == 1
            self.king_position_correct += predicted_kings == expected_kings

    def report(self):
        return {
            "screens": self.screens,
            "reader_status": dict(sorted(self.statuses.items())),
            "strict": {
                **self.exact.report(),
                "screen_exact_rate": _rate(self.exact_screens, self.screens),
            },
            "legacy_compatible": {
                **self.compatible.report(),
                "screen_exact_rate": _rate(self.compatible_screens, self.screens),
                "full_64_cell_boards": self.complete_screens,
                "full_64_cell_board_exact_rate": _rate(self.complete_exact_screens, self.complete_screens),
            },
            "occupied_cell_accuracy": _rate(self.occupied_correct, self.occupied_cells),
            "player_king": {
                "screens_with_one_expected": self.king_expected,
                "exactly_one_detected_rate": _rate(self.king_exactly_one, self.king_expected),
                "correct_position_rate": _rate(self.king_position_correct, self.king_expected),
            },
        }


class AmmoMetrics:
    def __init__(self):
        self.statuses = Counter()
        self.screens = 0
        self.loaded_correct = 0
        self.reserve_correct = 0
        self.complete_correct = 0

    def add(self, expected, prediction):
        self.screens += 1
        self.statuses[prediction.status.value] += 1
        reading = prediction.value
        loaded = getattr(reading, "loaded", None)
        reserve = getattr(reading, "reserve", None)
        self.loaded_correct += loaded == expected["loaded"]
        self.reserve_correct += reserve == expected["reserve"]
        self.complete_correct += loaded == expected["loaded"] and reserve == expected["reserve"]

    def report(self):
        return {
            "screens": self.screens,
            "reader_status": dict(sorted(self.statuses.items())),
            "loaded_accuracy": _rate(self.loaded_correct, self.screens),
            "reserve_accuracy": _rate(self.reserve_correct, self.screens),
            "complete_accuracy": _rate(self.complete_correct, self.screens),
        }


class CombinedMetrics:
    def __init__(self):
        self.eligible = 0
        self.accepted = 0
        self.correct = 0

    def add(self, screen_prediction, piece_prediction, ammo_prediction, piece_correct, ammo_correct):
        self.eligible += 1
        accepted = (
            screen_prediction.is_accepted
            and screen_prediction.value is ScreenKind.GAMEPLAY
            and piece_prediction.is_accepted
            and ammo_prediction.is_accepted
        )
        self.accepted += accepted
        self.correct += accepted and piece_correct and ammo_correct

    def report(self):
        return {
            "eligible_gameplay_screens": self.eligible,
            "accepted_rate": _rate(self.accepted, self.eligible),
            "correct_and_accepted_rate": _rate(self.correct, self.eligible),
            "required_readers": ["screen_kind", "pieces", "ammo"],
            "combat_stats_excluded": "reader_not_implemented",
        }


class EvaluationBucket:
    def __init__(self):
        self.annotations = 0
        self.pieces = PieceMetrics()
        self.ammo = AmmoMetrics()
        self.screen_kind = ClassificationMetrics()
        self.screen_statuses = Counter()
        self.combined = CombinedMetrics()

    def report(self):
        return {
            "annotations": self.annotations,
            "pieces": self.pieces.report(),
            "ammo": self.ammo.report(),
            "screen_kind": {
                **self.screen_kind.report(),
                "reader_status": dict(sorted(self.screen_statuses.items())),
            },
            "combined": self.combined.report(),
        }


def evaluate_readers(sessions_dir, piece_reader=None, ammo_reader=None, screen_reader=None):
    """Return deterministic metrics for reviewed labels without changing data."""
    sessions_dir = Path(sessions_dir)
    piece_reader = piece_reader or TemplatePieceReader.from_collection(Path(sessions_dir).parent)
    ammo_reader = ammo_reader or RedPixelAmmoReader()
    screen_reader = screen_reader or OverlayScreenReader()
    buckets = {name: EvaluationBucket() for name in ("overall", *SPLITS)}
    digest = hashlib.sha256()
    paths = sorted(sessions_dir.glob("*/annotations/*.json"))

    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        split = data.get("split")
        if split not in SPLITS:
            raise ValueError(f"{path}: invalid split")
        relative = path.relative_to(sessions_dir).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(raw)
        state = data.get("screen_state") or {}
        reviews = state.get("reviews") or {}
        eligible = (
            reviews.get("screen_kind") is True
            or reviews.get("ammo") is True
            or data.get("confirmed") is True
        )
        if not eligible:
            continue
        image_path = (path.parent / data.get("image", "")).resolve()
        originals = (path.parent.parent / "originals").resolve()
        if image_path.parent != originals or not image_path.is_file():
            raise ValueError(f"{path}: invalid image")
        image_raw = image_path.read_bytes()
        digest.update(hashlib.sha256(image_raw).digest())
        image = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{path}: unreadable image")
        screen = normalize(image, data.get("game_crop"))
        frame = PreparedFrame(screen, normalized=True)
        selected = (buckets["overall"], buckets[split])
        for bucket in selected:
            bucket.annotations += 1

        screen_prediction = None
        expected_kind = state.get("screen_kind")
        if reviews.get("screen_kind") is True and expected_kind is not None:
            analyzer.reset_analyzer_cache()
            screen_prediction = read_frame(screen_reader, frame)
            predicted_kind = _screen_label(screen_prediction)
            for bucket in selected:
                bucket.screen_statuses[screen_prediction.status.value] += 1
                bucket.screen_kind.add(expected_kind, predicted_kind)

        piece_prediction = None
        piece_correct = False
        if data.get("confirmed") is True and expected_kind == "gameplay":
            analyzer.reset_analyzer_cache()
            piece_prediction = read_frame(piece_reader, frame)
            board = data.get("board")
            if not _valid_expected_board(board):
                raise ValueError(f"{path}: board must contain 8x8 string labels")
            excluded = data.get("exclude_cells") or []
            piece_correct = _piece_prediction_matches(board, piece_prediction.value, excluded)
            for bucket in selected:
                bucket.pieces.add(board, piece_prediction, excluded)

        ammo_prediction = None
        ammo_correct = False
        ammo = state.get("ammo") or {}
        if reviews.get("ammo") is True and _valid_expected_ammo(ammo):
            ammo_prediction = read_frame(ammo_reader, frame)
            reading = ammo_prediction.value
            ammo_correct = (
                getattr(reading, "loaded", None) == ammo["loaded"]
                and getattr(reading, "reserve", None) == ammo["reserve"]
            )
            for bucket in selected:
                bucket.ammo.add(ammo, ammo_prediction)

        if screen_prediction is not None and piece_prediction is not None and ammo_prediction is not None:
            for bucket in selected:
                bucket.combined.add(screen_prediction, piece_prediction, ammo_prediction, piece_correct, ammo_correct)

    return {
        "schema": "reader-evaluation-v1",
        "dataset_sha256": digest.hexdigest(),
        "readers": {
            "pieces": getattr(piece_reader, "source", piece_reader.__class__.__name__),
            "ammo": getattr(ammo_reader, "source", ammo_reader.__class__.__name__),
            "screen_kind": getattr(screen_reader, "source", screen_reader.__class__.__name__),
            "combat_stats": "not_implemented",
        },
        "piece_label_note": "Strict metrics preserve queen (8), white_king (9), and special_knight (7). Unresolved royals remain ID 6. Legacy-compatible metrics merge royal kinds; extended IDs are rejected for legacy policy use.",
        "overall": buckets["overall"].report(),
        "splits": {split: buckets[split].report() for split in SPLITS},
    }


def _compatible_piece(label):
    return "royal" if label in ("queen", "white_king", "queen_or_white_king") else label


def _screen_label(prediction):
    value = prediction.value
    if isinstance(value, ScreenKind):
        return value.value
    if isinstance(value, str):
        return value
    return prediction.status.value


def _valid_expected_board(board):
    return (
        isinstance(board, list)
        and len(board) == 8
        and all(isinstance(row, list) and len(row) == 8 and all(isinstance(value, str) for value in row) for row in board)
    )


def _valid_expected_ammo(ammo):
    return all(type(ammo.get(field)) is int and ammo[field] >= 0 for field in ("loaded", "reserve"))


def _piece_prediction_matches(expected, predicted, excluded):
    if not isinstance(predicted, np.ndarray) or predicted.shape != (8, 8):
        return False
    excluded_set = {tuple(cell) for cell in excluded}
    for row in range(8):
        for col in range(8):
            if (row, col) in excluded_set:
                continue
            if _compatible_piece(expected[row][col]) != _compatible_piece(PIECE_LABELS.get(int(predicted[row, col]), "unknown")):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=ROOT / "data/collection/sessions")
    parser.add_argument("--output", type=Path, help="Write the JSON report to a new file")
    args = parser.parse_args()
    try:
        report = evaluate_readers(args.sessions)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists():
                raise ValueError(f"Output already exists: {args.output}; choose a new path")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Reader evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
