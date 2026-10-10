"""Report whether reviewed collection sessions can support offline evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "val", "test")
PAGE_INFO_SECTIONS = ("screen_kind", "ammo", "combat_stats")
REVIEW_SECTIONS = (*PAGE_INFO_SECTIONS, "cards", "soul")
COMBAT_FIELDS = (
    "attack",
    "range_min",
    "range_max",
    "spread_degrees",
    "penetration_percent",
    "knockback_percent",
    "grenades_remaining",
    "grenade_damage",
    "melee_damage",
)


def audit_collection(sessions_dir: Path) -> dict:
    """Return review coverage and split-integrity findings without mutation."""
    paths = sorted(sessions_dir.glob("*/annotations/*.json"))
    split_screens = Counter()
    split_sessions = {split: set() for split in SPLITS}
    review_counts = Counter()
    piece_classes = {split: Counter() for split in SPLITS}
    screen_kinds = Counter()
    ammo_values = {"loaded": set(), "reserve": set()}
    combat_values = {field: set() for field in COMBAT_FIELDS}
    soul_values = Counter()
    soul_slot_counts = Counter()
    usable_labels = {
        split: Counter() for split in SPLITS
    }
    sessions = {}
    image_digests = {}
    issues = []
    confirmed = 0
    piece_screens = 0
    page_complete = 0

    for path in paths:
        data = json.loads(path.read_text())
        session_id = data.get("session_id") or path.parent.parent.name
        split = data.get("split")
        session = sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "splits": set(),
                "screens": 0,
                "piece_screens": 0,
                "pieces_confirmed": 0,
                "page_info_complete": 0,
                "reviews": Counter(),
            },
        )
        session["screens"] += 1
        if split in SPLITS:
            session["splits"].add(split)
            split_screens[split] += 1
            split_sessions[split].add(session_id)
        else:
            issues.append(f"invalid_split:{path.relative_to(sessions_dir)}")

        state = data.get("screen_state") or {}
        reviews = state.get("reviews") or {}
        piece_applicable = not (
            reviews.get("screen_kind") is True
            and state.get("screen_kind") not in (None, "gameplay")
        )
        if piece_applicable:
            piece_screens += 1
            session["piece_screens"] += 1
            if data.get("confirmed") is True:
                confirmed += 1
                session["pieces_confirmed"] += 1
                if split in SPLITS:
                    for row in data.get("board") or []:
                        piece_classes[split].update(row)
        for section in REVIEW_SECTIONS:
            if reviews.get(section) is True:
                review_counts[section] += 1
                session["reviews"][section] += 1
        required_page_sections = ("screen_kind",) if state.get("screen_kind") not in (None, "gameplay") else PAGE_INFO_SECTIONS
        if all(reviews.get(section) is True for section in required_page_sections):
            page_complete += 1
            session["page_info_complete"] += 1
        if reviews.get("screen_kind") is True and state.get("screen_kind") is not None:
            screen_kinds[state["screen_kind"]] += 1
            if split in SPLITS:
                usable_labels[split]["screen_kind"] += 1
        if reviews.get("ammo") is True:
            ammo = state.get("ammo") or {}
            for field in ammo_values:
                value = ammo.get(field)
                if value is not None:
                    ammo_values[field].add(value)
            if split in SPLITS and all(ammo.get(field) is not None for field in ammo_values):
                usable_labels[split]["ammo"] += 1
        if reviews.get("combat_stats") is True:
            combat = state.get("combat_stats") or {}
            for field in COMBAT_FIELDS:
                value = combat.get(field)
                if value is not None:
                    combat_values[field].add(value)
            required = tuple(field for field in COMBAT_FIELDS if field != "grenade_damage")
            grenade_complete = combat.get("grenades_remaining") == 0 or combat.get("grenade_damage") is not None
            if split in SPLITS and all(combat.get(field) is not None for field in required) and grenade_complete:
                usable_labels[split]["combat_stats"] += 1
        if reviews.get("cards") is True and split in SPLITS:
            usable_labels[split]["cards"] += 1
        if reviews.get("soul") is True and state.get("screen_kind") in (None, "gameplay"):
            souls = state.get("souls") or []
            soul_values.update(souls or ["none"])
            soul_slot_counts[len(souls)] += 1
            if split in SPLITS:
                usable_labels[split]["soul"] += 1

        image = (path.parent / data.get("image", "")).resolve()
        originals = (path.parent.parent / "originals").resolve()
        if image.parent != originals or not image.is_file():
            issues.append(f"invalid_image:{path.relative_to(sessions_dir)}")
        else:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            image_digests.setdefault(digest, []).append(path)

    for session_id, session in sessions.items():
        if len(session["splits"]) > 1:
            issues.append(f"mixed_session_split:{session_id}")
    for digest, duplicates in image_digests.items():
        if len(duplicates) > 1:
            issues.append(f"duplicate_image:{digest}:{len(duplicates)}")
    if not paths:
        issues.append("no_annotations")
    for split in SPLITS:
        if not split_sessions[split]:
            issues.append(f"missing_split:{split}")
        for section in PAGE_INFO_SECTIONS:
            if usable_labels[split][section] == 0:
                issues.append(f"missing_usable_labels:{split}:{section}")
    if confirmed != piece_screens:
        issues.append(f"piece_review_incomplete:{piece_screens - confirmed}")
    if page_complete != len(paths):
        issues.append(f"page_review_incomplete:{len(paths) - page_complete}")

    return {
        "screens": len(paths),
        "piece_screens": piece_screens,
        "pieces_confirmed": confirmed,
        "page_info_complete": page_complete,
        "splits": {
            split: {
                "sessions": len(split_sessions[split]),
                "screens": split_screens[split],
            }
            for split in SPLITS
        },
        "reviews": {section: review_counts[section] for section in REVIEW_SECTIONS},
        "usable_labels": {
            split: {
                section: usable_labels[split][section] for section in REVIEW_SECTIONS
            }
            for split in SPLITS
        },
        "screen_kinds": dict(sorted(screen_kinds.items())),
        "ammo_values": {field: sorted(values) for field, values in ammo_values.items()},
        "combat_values": {field: sorted(values) for field, values in combat_values.items()},
        "soul_values": dict(sorted(soul_values.items())),
        "soul_slot_counts": {str(count): screens for count, screens in sorted(soul_slot_counts.items())},
        "piece_classes": {
            split: dict(sorted(piece_classes[split].items())) for split in SPLITS
        },
        "sessions": [
            {
                **{key: value for key, value in session.items() if key not in ("splits", "reviews")},
                "split": next(iter(session["splits"])) if len(session["splits"]) == 1 else None,
                "reviews": {
                    section: session["reviews"][section] for section in REVIEW_SECTIONS
                },
            }
            for session in sorted(sessions.values(), key=lambda item: item["session_id"])
        ],
        "issues": sorted(issues),
        "evaluation_ready": not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        type=Path,
        default=ROOT / "data/collection/sessions",
        help="Collection sessions directory",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit unsuccessfully when the collection is not evaluation-ready",
    )
    args = parser.parse_args()
    try:
        report = audit_collection(args.sessions)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Collection audit failed: {exc}\n")
    print(json.dumps(report, indent=2))
    if args.require_ready and not report["evaluation_ready"]:
        parser.exit(1, "Collection is not evaluation-ready.\n")


if __name__ == "__main__":
    main()
