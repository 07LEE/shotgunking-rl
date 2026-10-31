"""Reviewed synthetic collection samples for offline metrics."""

import json

import cv2
import numpy as np


def create_reviewed_screen(collection, split="train"):
    session = collection / "sessions" / "run-01"
    originals = session / "originals"
    annotations = session / "annotations"
    originals.mkdir(parents=True)
    annotations.mkdir()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imwrite(str(originals / "screen.png"), image)
    board = [["empty" for _ in range(8)] for _ in range(8)]
    board[0][0] = "queen"
    board[7][4] = "player_king"
    annotation = {
        "schema": "piece-cells-v1",
        "image": "../originals/screen.png",
        "session_id": "run-01",
        "split": split,
        "game_crop": [0, 0, 1280, 720],
        "confirmed": True,
        "exclude_cells": [],
        "board": board,
        "screen_state": {
            "screen_kind": "gameplay",
            "ammo": {"loaded": 2, "reserve": 6},
            "reviews": {"screen_kind": True, "ammo": True},
        },
    }
    (annotations / "screen.json").write_text(json.dumps(annotation))
