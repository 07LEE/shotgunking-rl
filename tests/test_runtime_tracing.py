"""Offline checks for private runtime action traces."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import train
from shotgun_king_rl.runtime.suggestions import SuggestedAction
from shotgun_king_rl.runtime.tracing import (
    ActionTraceWriter,
    build_action_trace,
    file_sha256,
)


def runtime_doubles():
    """One verified transition; all environment and learning methods are mocked."""
    state = np.zeros(281, dtype=np.float32)
    state[7 * 8 + 4] = 1
    action_mask = np.zeros(10, dtype=np.float32)
    action_mask[4] = 1
    environment = MagicMock()
    environment.action_space = SimpleNamespace(n=10)
    environment.loaded_ammo = environment.max_ammo = 2
    environment.reserve_ammo = 6
    environment.range_limit = 5
    environment.royal_guard_active = False
    environment.melee_damage = 0.0
    environment.reset.return_value = (state.copy(), {"reset_ready": True})
    environment.get_action_mask.return_value = action_mask.copy()
    environment.step.return_value = (
        state.copy(), 0.5, True, False,
        {"action": 4, "input_succeeded": True, "transition_ready": True},
    )
    agent = MagicMock()
    agent.policy_net = None
    agent.act.return_value = 4
    agent.load.return_value = True
    agent.replay.return_value = 0.0
    return environment, agent


class RuntimeTracingTests(unittest.TestCase):
    def test_suggest_loop_does_not_learn_or_save(self):
        import torch

        environment, agent = runtime_doubles()

        class Policy:
            def __call__(self, tensor):
                return torch.arange(10, dtype=torch.float32).unsqueeze(0)

        agent.policy_net = Policy()
        agent.device = torch.device("cpu")

        with TemporaryDirectory() as directory, patch.object(
            train, "ShotgunKingEnv", return_value=environment
        ), patch.object(train, "DQNAgent", return_value=agent), patch.object(
            train, "SummaryWriter", None
        ), patch("builtins.input", return_value=""):
            train.train_dqn(
                episodes=1,
                mode="suggest",
                trace_directory=directory,
                verbose=False,
            )

        agent.remember.assert_not_called()
        agent.replay.assert_not_called()
        agent.update_target_network.assert_not_called()
        agent.save.assert_not_called()

    def test_train_loop_learns_and_saves(self):
        environment, agent = runtime_doubles()

        with TemporaryDirectory() as directory, patch.object(
            train, "ShotgunKingEnv", return_value=environment
        ), patch.object(train, "DQNAgent", return_value=agent), patch.object(
            train, "SummaryWriter", None
        ):
            train.train_dqn(
                episodes=1,
                mode="train",
                trace_directory=directory,
                model_path=str(Path(directory) / "model.pth"),
                verbose=False,
            )

        agent.remember.assert_called_once()
        agent.replay.assert_called_once()
        agent.save.assert_called_once()

    def test_evaluate_loop_is_greedy_and_does_not_learn(self):
        environment, agent = runtime_doubles()

        with TemporaryDirectory() as directory, patch.object(
            train, "ShotgunKingEnv", return_value=environment
        ), patch.object(train, "DQNAgent", return_value=agent), patch.object(
            train, "SummaryWriter", None
        ):
            train.train_dqn(
                episodes=1,
                mode="evaluate",
                trace_directory=directory,
                verbose=False,
            )
            payload = json.loads(
                next(Path(directory).glob("*.jsonl")).read_text(encoding="utf-8")
            )

        agent.act.assert_called_once()
        self.assertEqual(agent.act.call_args.kwargs["epsilon"], 0.0)
        agent.remember.assert_not_called()
        agent.replay.assert_not_called()
        agent.update_target_network.assert_not_called()
        agent.save.assert_not_called()
        self.assertEqual(payload["decision"]["selection_source"], "greedy_policy")

    def test_evaluate_stops_before_reset_without_checkpoint(self):
        environment, agent = runtime_doubles()
        agent.load.return_value = False

        with TemporaryDirectory() as directory, patch.object(
            train, "ShotgunKingEnv", return_value=environment
        ), patch.object(train, "DQNAgent", return_value=agent), patch.object(
            train, "SummaryWriter", None
        ):
            started = train.train_dqn(
                episodes=1,
                mode="evaluate",
                trace_directory=directory,
                model_path=str(Path(directory) / "missing.pth"),
                verbose=False,
            )

        self.assertFalse(started)
        environment.reset.assert_not_called()
        agent.remember.assert_not_called()
        agent.save.assert_not_called()

    def test_manifest_records_mode_and_model_identity(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / "model.pth"
            model.write_bytes(b"model")
            writer = ActionTraceWriter(directory, session_id="session-01")
            writer.write_manifest({
                "mode": "collect",
                "learning_enabled": False,
                "model": {"path": str(model), "sha256": file_sha256(model)},
            })
            payload = json.loads(writer.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "runtime-run-manifest-v1")
        self.assertEqual(payload["mode"], "collect")
        self.assertFalse(payload["learning_enabled"])
        self.assertEqual(len(payload["model"]["sha256"]), 64)

    def test_collect_loop_records_verified_transition_without_learning(self):
        environment, agent = runtime_doubles()
        agent.load.return_value = None

        with TemporaryDirectory() as directory, patch.object(
            train, "ShotgunKingEnv", return_value=environment
        ), patch.object(train, "DQNAgent", return_value=agent), patch.object(
            train, "SummaryWriter", None
        ):
            train.train_dqn(
                episodes=1,
                mode="collect",
                trace_directory=directory,
                model_path=str(Path(directory) / "new-model.pth"),
                verbose=False,
            )
            paths = list(Path(directory).glob("*.jsonl"))
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest = json.loads(
                next(Path(directory).glob("*.manifest.json")).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(paths), 1)
        self.assertEqual(payload["mode"], "collect")
        self.assertEqual(payload["decision"]["selected_action"], 4)
        self.assertEqual(payload["decision"]["selection_source"], "epsilon_greedy")
        self.assertTrue(payload["result"]["transition_verified"])
        self.assertFalse(payload["result"]["accepted_for_replay"])
        self.assertFalse(manifest["learning_enabled"])
        self.assertEqual(manifest['observation']['schema'], 'distinct-royals-281-v1')
        self.assertEqual(manifest['observation']['size'], 281)
        agent.remember.assert_not_called()
        agent.replay.assert_not_called()
        agent.update_target_network.assert_not_called()
        agent.save.assert_not_called()

    def test_existing_incompatible_checkpoint_blocks_all_modes_before_input_or_save(self):
        for mode in ('suggest', 'evaluate', 'collect', 'train'):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                environment, agent = runtime_doubles()
                agent.load.return_value = False
                model = Path(directory) / 'legacy.pth'
                model.write_bytes(b'legacy weights must be preserved')
                with patch.object(train, 'ShotgunKingEnv', return_value=environment), patch.object(train, 'DQNAgent', return_value=agent), patch.object(train, 'SummaryWriter', None):
                    started = train.train_dqn(episodes=1, mode=mode, trace_directory=directory, model_path=str(model), verbose=False)
                self.assertFalse(started)
                environment.reset.assert_not_called()
                environment.step.assert_not_called()
                agent.save.assert_not_called()
                self.assertEqual(model.read_bytes(), b'legacy weights must be preserved')

    def test_writer_appends_json_safe_action_record(self):
        observation = np.zeros(281, dtype=np.float32)
        observation[60] = 1
        next_observation = observation.copy()
        next_observation[60] = 0
        next_observation[52] = 1
        mask = np.zeros(10, dtype=np.float32)
        mask[1] = 1
        recommendation = SuggestedAction(1, "Move Up", "8", "safe empty square e2", 0.5)
        record = build_action_trace(
            episode=1,
            step=2,
            mode="collect",
            rank=3,
            floor=4,
            weapon_type=1,
            weapon_name="Victoria",
            observation=observation,
            action_mask=mask,
            q_values=np.array([np.nan, 0.5]),
            recommendations=(recommendation,),
            selected_action=1,
            selection_source="epsilon_greedy",
            loaded_ammo=1,
            max_ammo=1,
            reserve_ammo=3,
            next_observation=next_observation,
            reward=0.25,
            terminated=False,
            truncated=False,
            info={"action": np.int64(1), "transition_ready": True},
            accepted_for_replay=True,
        )

        with TemporaryDirectory() as directory:
            writer = ActionTraceWriter(directory, session_id="session-01")
            writer.write(record)
            payload = json.loads(writer.path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "runtime-action-trace-v1")
        self.assertEqual(payload["session_id"], "session-01")
        self.assertEqual(payload["scope"]["weapon_name"], "Victoria")
        self.assertEqual(payload["decision"]["q_values"], [None, 0.5])
        self.assertEqual(payload["decision"]["selected_name"], "Move Up")
        self.assertTrue(payload["result"]["accepted_for_replay"])
        self.assertEqual(payload["result"]["next_board"][6][4], 1.0)

    def test_writer_keeps_failed_transition_for_diagnosis(self):
        observation = np.zeros(281, dtype=np.float32)
        record = build_action_trace(
            episode=1,
            step=1,
            mode="suggest",
            rank=1,
            floor=1,
            weapon_type=0,
            weapon_name="Solomon",
            observation=observation,
            action_mask=np.ones(10),
            q_values=None,
            recommendations=(),
            selected_action=8,
            selection_source="user_override",
            loaded_ammo=1,
            max_ammo=2,
            reserve_ammo=4,
            next_observation=observation,
            reward=0,
            terminated=False,
            truncated=False,
            info={"action": None, "input_succeeded": False},
            accepted_for_replay=False,
        )

        self.assertFalse(record["result"]["accepted_for_replay"])
        self.assertFalse(record["result"]["info"]["input_succeeded"])


if __name__ == "__main__":
    unittest.main()
