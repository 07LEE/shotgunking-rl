"""Construct combat readers for offline evaluation using train references only."""

import hashlib
import json
from pathlib import Path

import cv2

from ..perception import (
    AttackReader, CombatStatsReader, GrenadeCountReader, GrenadeDamageReader,
    KnockbackReader, MeleeDamageReader, PenetrationReader, RangeReader, SpreadReader,
)
from ..perception.melee import melee_damage_patch
from ..vision.layout import normalize_game_screen


def build_combat_reader(collection, title_annotation=None):
    collection = Path(collection).resolve()
    metadata = {"melee_title": None}
    melee = None
    if title_annotation is not None:
        path = Path(title_annotation).resolve()
        if not path.is_relative_to(collection / "sessions"):
            raise ValueError("Title reference must belong to the collection")
        raw = path.read_bytes()
        data = json.loads(raw)
        state = data.get("screen_state") or {}
        reviews = state.get("reviews") or {}
        value = (state.get("combat_stats") or {}).get("melee_damage")
        if (data.get("split") != "train" or state.get("screen_kind") != "gameplay"
                or reviews.get("screen_kind") is not True or reviews.get("combat_stats") is not True
                or type(value) is not int or value <= 0):
            raise ValueError("Title reference must be reviewed positive gameplay train data")
        image_path = (path.parent / data.get("image", "")).resolve()
        if image_path.parent != (path.parent.parent / "originals").resolve():
            raise ValueError("Invalid title reference image path")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Unreadable title reference image")
        patch = melee_damage_patch(normalize_game_screen(image, data.get("game_crop")))
        melee = MeleeDamageReader.from_collection(collection, title_reference=None if patch is None else patch[:30])
        metadata["melee_title"] = {
            "annotation": path.relative_to(collection).as_posix(),
            "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        }
    readers = {
        "attack_reader": AttackReader.from_collection(collection),
        "range_reader": RangeReader.from_collection(collection),
        "spread_reader": SpreadReader.from_collection(collection),
        "penetration_reader": PenetrationReader.from_collection(collection),
        "knockback_reader": KnockbackReader.from_collection(collection),
        "grenade_count_reader": GrenadeCountReader.from_collection(collection),
        "grenade_damage_reader": GrenadeDamageReader.from_collection(collection),
        "melee_reader": melee,
    }
    metadata["readers"] = {
        name: {"source": reader.source, "references": len(reader.examples)} if reader is not None else None
        for name, reader in readers.items()
    }
    digest = hashlib.sha256()
    for path in sorted((collection / "sessions").glob("*/annotations/*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        state = data.get("screen_state") or {}
        reviews = state.get("reviews") or {}
        if (data.get("split") != "train" or state.get("screen_kind") != "gameplay"
                or reviews.get("screen_kind") is not True or reviews.get("combat_stats") is not True):
            continue
        image_path = (path.parent / data.get("image", "")).resolve()
        if image_path.parent != (path.parent.parent / "originals").resolve():
            raise ValueError("Invalid training reference image path")
        digest.update(path.relative_to(collection).as_posix().encode())
        digest.update(b"\0")
        digest.update(raw)
        digest.update(hashlib.sha256(image_path.read_bytes()).digest())
    metadata["reviewed_train_sha256"] = digest.hexdigest()
    return CombatStatsReader(**readers), metadata
