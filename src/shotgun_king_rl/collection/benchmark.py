"""Reproduce offline reader timings and compare predictions with a prior run."""

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
from statistics import mean, median
import tempfile
from time import perf_counter

import cv2
import numpy as np

from .evaluation import ROOT, evaluate_readers
from .images import normalize
from .. import perception
from ..perception.frame import PreparedFrame, read_frame
from ..vision import analyzer


SCHEMA = "reader-benchmark-v1"


def _digest(paths):
    digest = hashlib.sha256()
    for name, path in paths:
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _prediction(prediction):
    value = prediction.value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "value"):
        value = value.value
    return {"value": value, "status": prediction.status.value,
            "reason": prediction.reason, "confidence": prediction.confidence}


def _summary(values):
    return {"calls": len(values), "mean_ms": mean(values), "median_ms": median(values),
            "p95_ms": float(np.percentile(values, 95))}


def compare_reports(baseline, current):
    """Require identical inputs before attributing differences to implementation."""
    for key in ("schema", "dataset_sha256", "templates_sha256", "readers"):
        if baseline.get(key) != current.get(key):
            raise ValueError(f"Baseline {key} differs; use the same inputs and readers")
    return {
        "predictions_identical": baseline["predictions"] == current["predictions"],
        "evaluation_identical": baseline["evaluation"] == current["evaluation"],
        "environment_identical": baseline["environment"] == current["environment"],
        "timing_mean_change_percent": {
            key: (values["mean_ms"] / baseline["timings"][key]["mean_ms"] - 1) * 100
            for key, values in current["timings"].items()
            if key in baseline["timings"] and baseline["timings"][key]["mean_ms"] > 0
        },
    }


def _benchmark(collection, repeats, include_hud):
    readers = {
        "screen_kind": perception.OverlayScreenReader(),
        "pieces": perception.TemplatePieceReader.from_collection(collection),
        "ammo": perception.RedPixelAmmoReader(),
    }
    if include_hud:
        for name in ("Attack", "Range", "Spread", "Knockback", "Penetration", "GrenadeDamage", "GrenadeCount"):
            readers[name] = getattr(perception, name + "Reader").from_collection(collection)
    evaluation = evaluate_readers(collection / "sessions", piece_reader=readers["pieces"])
    samples = []
    for path in sorted((collection / "sessions").glob("*/annotations/*.json")):
        data = json.loads(path.read_text())
        state = data.get("screen_state") or {}
        if (state.get("reviews") or {}).get("screen_kind") is not True:
            continue
        image_path = (path.parent / data.get("image", "")).resolve()
        if image_path.parent != (path.parent.parent / "originals").resolve():
            raise ValueError(f"{path}: invalid image path")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"{path}: unreadable image")
        samples.append((path.relative_to(collection / "sessions").as_posix(),
                        normalize(image, data.get("game_crop"))))
    if not samples:
        raise ValueError("No screen-reviewed train/val samples")

    timings = {"observation": []}
    predictions = {}
    # One untimed warm-up per sample; I/O, reference construction and explicit
    # annotation crop normalization stay outside the timed observation.
    for iteration in range(repeats + 1):
        for name, image in samples:
            analyzer.reset_analyzer_cache()
            start = perf_counter()
            frame = PreparedFrame(image, normalized=True)
            row = {}
            elapsed = {}
            screen = None
            for key, reader in readers.items():
                if key != "screen_kind" and not (
                    screen is not None and screen.is_accepted and screen.value is perception.ScreenKind.GAMEPLAY
                ):
                    break
                reader_start = perf_counter()
                result = read_frame(reader, frame)
                elapsed[key] = (perf_counter() - reader_start) * 1000
                row[key] = result
                if key == "screen_kind":
                    screen = result
            duration = (perf_counter() - start) * 1000
            rendered = {key: _prediction(value) for key, value in row.items()}
            if iteration == 0:
                predictions[name] = rendered
            else:
                if predictions[name] != rendered:
                    raise ValueError(f"Nondeterministic predictions: {name}")
                timings["observation"].append(duration)
                for key, value in elapsed.items():
                    timings.setdefault(key, []).append(value)
    return {
        "schema": SCHEMA, "dataset_sha256": evaluation["dataset_sha256"],
        "readers": list(readers), "evaluation": evaluation,
        "samples": len(samples), "repeats": repeats, "warmup_per_sample": 1,
        "predictions": predictions, "timings": {k: _summary(v) for k, v in timings.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "opencv": cv2.__version__, "machine": platform.machine(),
                        "opencv_threads": cv2.getNumThreads()},
        "source_sha256": _digest((p.relative_to(ROOT / "src").as_posix(), p)
                                  for p in sorted((ROOT / "src").rglob("*.py"))),
        "templates_sha256": _digest((p.name, p) for p in sorted(Path(analyzer.PIECES_DIR).glob("*.png"))),
        "limitations": "Offline CPU timings; excludes I/O, reference loading and annotation crop normalization. "
                       "Reader timings depend on order because frame work is shared. "
                       "Gameplay gates remaining readers; no complete combat stats, game input or training. "
                       "Train references are reused; timings do not establish recognition generalization.",
    }


def benchmark_readers(collection, repeats=3, include_hud=False):
    """Build a train/val-only view; never decode held-out test images."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    collection = Path(collection)
    with tempfile.TemporaryDirectory(prefix="shotgun-reader-benchmark-") as directory:
        view = Path(directory)
        (view / "sessions").mkdir()
        for session in sorted((collection / "sessions").iterdir()):
            if not session.is_dir():
                continue
            paths = sorted((session / "annotations").glob("*.json"))
            splits = {json.loads(path.read_text()).get("split") for path in paths}
            if not splits:
                continue
            if len(splits) != 1 or not splits <= {"train", "val", "test"}:
                raise ValueError(f"{session.name}: invalid or mixed session splits")
            if splits == {"test"}:
                continue
            (view / "sessions" / session.name).symlink_to(session.resolve(), target_is_directory=True)
        return _benchmark(view, repeats, include_hud)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=ROOT / "data/collection")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-hud", action="store_true")
    parser.add_argument("--baseline", type=Path, help="Compare with a previous benchmark JSON")
    parser.add_argument("--output", type=Path, required=True, help="New private report path")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new report path")
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        report = benchmark_readers(args.collection, args.repeats, args.include_hud)
        if baseline is not None:
            report["comparison"] = compare_reports(baseline, report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("samples", "repeats", "timings")} |
                         {"comparison": report.get("comparison"), "output": str(args.output)}, indent=2))
        if baseline is not None and not all(report["comparison"][key] for key in
                                            ("predictions_identical", "evaluation_identical")):
            parser.exit(2, "Reader results changed; inspect the saved comparison report\n")
    except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
