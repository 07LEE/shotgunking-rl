"""Distinguish royal pieces using confirmed training cells only."""

import json
from pathlib import Path

import cv2
import numpy as np

from ..vision import analyzer
from ..vision.layout import normalize_game_screen


ROYAL_IDS = {"queen": 8, "white_king": 9}


class RoyalTemplates:
    """Keep ambiguous royals at legacy ID 6; never infer kind from position."""

    def __init__(self, examples=()):
        self.examples = tuple((kind, signal.copy()) for kind, signal in examples)
        self._template_stds = tuple(template.std() for _, template in self.examples)

    @classmethod
    def from_collection(cls, collection):
        examples = []
        for path in sorted((Path(collection) / "sessions").glob("*/annotations/*.json")):
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            if (data.get("split") != "train" or data.get("confirmed") is not True
                    or state.get("screen_kind") != "gameplay"
                    or (state.get("reviews") or {}).get("screen_kind") is not True):
                continue
            image_path = (path.parent / data.get("image", "")).resolve()
            if image_path.parent != (path.parent.parent / "originals").resolve():
                raise ValueError(f"{path}: invalid image path")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"{path}: unreadable image")
            board = analyzer.crop_chessboard(normalize_game_screen(image, data.get("game_crop")))
            gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
            excluded = data.get("exclude_cells") or []
            for row, labels in enumerate(data["board"]):
                for col, label in enumerate(labels):
                    if label not in ROYAL_IDS or [row, col] in excluded:
                        continue
                    signal = analyzer._extract_signal(gray[row*65:(row+1)*65, col*65:(col+1)*65]).astype(np.float32)
                    if signal.std() >= 5:
                        examples.append((ROYAL_IDS[label], signal))
        return cls(examples)

    def classify(self, signal):
        query = signal.astype(np.float32)
        if query.shape != (45, 45) or query.std() < 5:
            return 6
        scores = {}
        for (kind, template), template_std in zip(self.examples, self._template_stds):
            if template.shape != query.shape or template_std < 5:
                continue
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[kind] = max(scores.get(kind, -1), score)
        # Both kinds must be represented to measure cross-kind ambiguity.
        if set(scores) != set(ROYAL_IDS.values()):
            return 6
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        if ranked[0][1] < .90 or ranked[0][1] - ranked[1][1] < .10:
            return 6
        return ranked[0][0]

    def refine(self, image, board, *, board_gray=None):
        result = board.copy()
        gray = board_gray if board_gray is not None else cv2.cvtColor(analyzer.crop_chessboard(image), cv2.COLOR_BGR2GRAY)
        for row, col in np.argwhere(board == 6):
            signal = analyzer._extract_signal(gray[row*65:(row+1)*65, col*65:(col+1)*65])
            result[row, col] = self.classify(signal)
        return result
