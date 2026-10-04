"""Build labeled cell crops from saved, manually annotated game screens."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from analyzer import crop_chessboard

CLASSES = ("empty", "player_king", "pawn", "knight", "bishop", "rook", "queen", "white_king", "special_knight")


def group_split(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def build_dataset(annotation_dir, output, allow_provisional=False):
    # Validate every source before writing anything. Never replace a dataset.
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new directory")
    sources = []
    groups = {}
    image_groups = {}
    skipped = 0
    paths = list(annotation_dir.glob("*.json")) + list(annotation_dir.glob("*/annotations/*.json"))
    for path in sorted(paths):
        data = json.loads(path.read_text())
        if data.get("schema") != "piece-cells-v1":
            raise ValueError(f"{path}: expected piece-cells-v1 schema")
        if data.get("confirmed") is not True and not allow_provisional:
            skipped += 1
            continue
        board = data["board"]
        if len(board) != 8 or any(len(row) != 8 for row in board):
            raise ValueError(f"{path}: board must be 8x8")
        if any(label not in CLASSES for row in board for label in row):
            raise ValueError(f"{path}: unknown class label")
        group = data.get("session_id")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"{path}: session_id is required")
        split = data.get("split", group_split(group))
        if split not in ("train", "val", "test"):
            raise ValueError(f"{path}: invalid split")
        if group in groups and groups[group] != split:
            raise ValueError(f"{path}: one session cannot span splits")
        groups[group] = split
        image_path = (path.parent / data["image"]).resolve()
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in image_groups:
            raise ValueError(f"{path}: duplicate image already present in {image_groups[digest]}")
        image_groups[digest] = str(path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"{path}: unreadable image")
        crop = data.get("game_crop")
        if crop is not None:
            if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop):
                raise ValueError(f"{path}: game_crop must be [left, top, right, bottom]")
            left, top, right, bottom = crop
            if not (0 <= left < right <= img.shape[1] and 0 <= top < bottom <= img.shape[0]):
                raise ValueError(f"{path}: game_crop is outside image bounds")
            img = cv2.resize(img[top:bottom, left:right], (1280, 720), interpolation=cv2.INTER_AREA)
        elif img.shape[:2] != (720, 1280):
            raise ValueError(f"{path}: non-1280x720 image requires game_crop")
        excluded = data.get("exclude_cells", [])
        if not isinstance(excluded, list) or any(not isinstance(cell, list) or len(cell) != 2 or any(type(v) is not int or not 0 <= v < 8 for v in cell) for cell in excluded):
            raise ValueError(f"{path}: exclude_cells must contain [row, col] pairs in 0..7")
        sources.append((path, data, digest, split, crop_chessboard(img)))
    if not sources:
        raise ValueError(f"No usable annotations ({skipped} unconfirmed skipped)")
    output.mkdir(parents=True)
    records = []
    counts = {split: Counter() for split in ("train", "val", "test")}
    for path, data, digest, split, board_img in sources:
        for row in range(8):
            for col in range(8):
                if [row, col] in data.get("exclude_cells", []):
                    continue
                label = data["board"][row][col]
                relative = Path(split) / label / f"{digest}_r{row}_c{col}.png"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                cell = board_img[row * 65:(row + 1) * 65, col * 65:(col + 1) * 65]
                if not cv2.imwrite(str(target), cell):
                    raise OSError(f"Could not write {target}")
                counts[split][label] += 1
                records.append({"path": relative.as_posix(), "label": label, "split": split, "session_id": data["session_id"], "floor": data.get("floor"), "game_crop": data.get("game_crop"), "source_annotation": str(path.resolve()), "source_sha256": digest, "confirmed": data.get("confirmed") is True, "row": row, "col": col, "locked": [row, col] in data.get("screen_state", {}).get("locked_cells", []), "screen_state": data.get("screen_state")})
    (output / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    summary = {"classes": list(CLASSES), "screens": len(sources), "cells": len(records), "skipped_unconfirmed": skipped, "counts": {split: {label: counts[split][label] for label in CLASSES} for split in counts}, "missing_classes": [label for label in CLASSES if not any(counts[s][label] for s in counts)], "evaluation_ready": all(sum(counts[s].values()) > 0 for s in counts) and all(counts["train"][label] > 0 for label in CLASSES) and all(r["confirmed"] for r in records)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/collection/sessions", help="An annotation directory or a sessions directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-provisional", action="store_true", help="Include unconfirmed labels for inspection only")
    args = parser.parse_args()
    try:
        print(json.dumps(build_dataset(args.annotations, args.output, args.allow_provisional), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Dataset build failed: {exc}\n")


if __name__ == "__main__":
    main()
