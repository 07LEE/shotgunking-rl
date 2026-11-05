"""Construct combat readers for offline evaluation using train references only."""

import hashlib
import json
from pathlib import Path

from ..collection_layout import annotation_paths
from ..perception import (
    AttackReader, CombatStatsReader, GrenadeCountReader, GrenadeDamageReader,
    KnockbackReader, MeleeDamageReader, PenetrationReader, RangeReader, SpreadReader,
)
from ..perception.title_references import load_melee_title_reference


def build_combat_reader(collection, title_annotation=None):
    collection = Path(collection).resolve()
    title_reference, title_metadata = load_melee_title_reference(collection, title_annotation)
    metadata = {"melee_title": title_metadata}
    melee = MeleeDamageReader.from_collection(collection, title_reference=title_reference)
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
    for name, reader in readers.items():
        if hasattr(reader, 'numeric_examples'):
            metadata['readers'][name]['numeric_references'] = len(reader.numeric_examples)
    metadata["readers"]["melee_reader"]["zero_references"] = len(melee.zero_examples)
    metadata["readers"]["grenade_count_reader"]["zero_references"] = len(readers["grenade_count_reader"].zero_examples)
    metadata["readers"]["grenade_damage_reader"]["title_references"] = len(readers["grenade_damage_reader"].titles)
    metadata["readers"]["grenade_damage_reader"]["integer_references"] = len(readers["grenade_damage_reader"].integers)
    digest = hashlib.sha256()
    for path in annotation_paths(collection):
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
