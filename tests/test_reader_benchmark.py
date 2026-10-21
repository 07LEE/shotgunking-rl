"""Small offline checks for repeatable comparisons and held-out isolation."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotgun_king_rl.collection.benchmark import benchmark_readers, compare_reports, main
from test_reader_evaluation import create_reviewed_screen


def report():
    return {"schema": "reader-benchmark-v1", "dataset_sha256": "data",
            "templates_sha256": "templates", "readers": ["screen_kind"],
            "predictions": {"screen": {"value": "gameplay"}}, "evaluation": {"accuracy": 1},
            "environment": {"opencv": "test"}, "timings": {"observation": {"mean_ms": 2}}}


class ReaderBenchmarkTests(unittest.TestCase):
    def test_compares_predictions_evaluation_and_relative_time(self):
        before = report()
        after = deepcopy(before)
        after["timings"]["observation"]["mean_ms"] = 1
        result = compare_reports(before, after)
        self.assertTrue(result["predictions_identical"])
        self.assertTrue(result["evaluation_identical"])
        self.assertEqual(result["timing_mean_change_percent"]["observation"], -50)
        after["predictions"]["screen"]["value"] = "retry"
        self.assertFalse(compare_reports(before, after)["predictions_identical"])

    def test_refuses_comparisons_with_different_inputs(self):
        for key in ("schema", "dataset_sha256", "templates_sha256", "readers"):
            after = report()
            after[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                compare_reports(report(), after)

    def test_benchmarks_reviewed_samples_without_opening_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            test = collection / "sessions/held-out/annotations"
            test.mkdir(parents=True)
            (test / "screen.json").write_text(json.dumps({"split": "test", "image": "nonexistent.png"}))
            result = benchmark_readers(collection, repeats=2)
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["timings"]["observation"]["calls"], 2)
            self.assertEqual(list(result["predictions"]), ["run-01/annotations/screen.json"])
            self.assertEqual(result["evaluation"]["splits"]["test"]["annotations"], 0)
            self.assertNotIn("pieces", result["timings"])  # Blank screen is unknown.

    def test_rejects_mixed_session_splits_and_invalid_repeat_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            create_reviewed_screen(collection)
            (collection / "sessions/run-01/annotations/other.json").write_text(json.dumps({"split": "test"}))
            with self.assertRaisesRegex(ValueError, "mixed session splits"):
                benchmark_readers(collection)
            for repeats in (0, -1, True):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    benchmark_readers(collection, repeats=repeats)

    def test_existing_output_is_preserved_before_work_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("existing")
            with patch.object(sys, "argv", ["benchmark", "--output", str(output)]), patch(
                "shotgun_king_rl.collection.benchmark.benchmark_readers"
            ) as benchmark, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.read_text(), "existing")
            benchmark.assert_not_called()

    def test_changed_predictions_save_report_and_return_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            output = Path(directory) / "current.json"
            before = report() | {"samples": 1, "repeats": 1}
            after = deepcopy(before)
            after["predictions"]["screen"]["value"] = "retry"
            baseline.write_text(json.dumps(before))
            with patch.object(sys, "argv", ["benchmark", "--baseline", str(baseline), "--output", str(output)]), patch(
                "shotgun_king_rl.collection.benchmark.benchmark_readers", return_value=after
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(json.loads(output.read_text())["comparison"]["predictions_identical"])
            self.assertEqual(json.loads(baseline.read_text()), before)
