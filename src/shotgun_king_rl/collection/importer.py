"""Move saved inbox images into a named session without overwriting files."""

import argparse
from pathlib import Path
import re

from .locking import serialized

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / "data" / "collection"


@serialized
def import_images(collection, session_id, image_names=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", session_id):
        raise ValueError("Session ID must contain only letters, digits, '-' and '_'")
    inbox = collection / "inbox"
    session = collection / "sessions" / session_id
    images = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")) if inbox.exists() else []
    if image_names is not None:
        images = [image for image in images if image.name in image_names]
    for image in images:
        if (session / "originals" / image.name).exists():
            raise ValueError(f"Already exists: {image.name}; no images were moved")
    inbox.mkdir(parents=True, exist_ok=True)
    for name in ("originals", "annotations", "review"):
        (session / name).mkdir(parents=True, exist_ok=True)
    for image in images:
        image.rename(session / "originals" / image.name)
    return len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Use the same ID for all floors of one play session")
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    args = parser.parse_args()
    try:
        count = import_images(args.collection, args.session)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    print(f"Moved {count} images into session {args.session}; annotations still require review")


if __name__ == "__main__":
    main()
