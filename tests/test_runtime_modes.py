"""Offline checks for explicit runtime mode permissions."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotgun_king_rl.runtime.modes import resolve_runtime_mode


class RuntimeModeTests(unittest.TestCase):
    def test_only_train_can_learn_or_write_checkpoints(self):
        for name in ("suggest", "collect", "evaluate"):
            mode = resolve_runtime_mode(name)
            self.assertFalse(mode.learning_enabled)
            self.assertFalse(mode.checkpoint_writes)

        training = resolve_runtime_mode("train")
        self.assertTrue(training.learning_enabled)
        self.assertTrue(training.checkpoint_writes)
        self.assertTrue(training.decay_epsilon)

    def test_suggest_and_evaluate_require_a_checkpoint(self):
        self.assertTrue(resolve_runtime_mode("suggest").requires_checkpoint)
        self.assertTrue(resolve_runtime_mode("evaluate").requires_checkpoint)
        self.assertFalse(resolve_runtime_mode("collect").requires_checkpoint)
        self.assertFalse(resolve_runtime_mode("train").requires_checkpoint)

    def test_modes_use_explicit_exploration(self):
        self.assertEqual(resolve_runtime_mode("suggest").initial_epsilon, 0.0)
        self.assertEqual(resolve_runtime_mode("evaluate").initial_epsilon, 0.0)
        self.assertEqual(resolve_runtime_mode("collect", 0.2).initial_epsilon, 0.2)
        self.assertEqual(resolve_runtime_mode("train").initial_epsilon, 1.0)

    def test_ambiguous_autonomous_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_runtime_mode("autonomous")


if __name__ == "__main__":
    unittest.main()
