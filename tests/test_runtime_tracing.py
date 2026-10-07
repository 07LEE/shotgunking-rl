"""Offline checks for private runtime action traces."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import train
from shotgun_king_rl.runtime.suggestions import SuggestedAction
from shotgun_king_rl.runtime.tracing import ActionTraceWriter, build_action_trace


class RuntimeTracingTests(unittest.TestCase):
    def test_autonomous_loop_records_verified_transition(self):
        state = np.zeros(281, dtype=np.float32)
        state[7 * 8 + 4] = 1
        action_mask = np.zeros(10, dtype=np.float32)
        action_mask[4] = 1

        class FakeEnvironment:
            def __init__(self, **kwargs):
                self.action_space = SimpleNamespace(n=10)
                self.loaded_ammo = 2
                self.max_ammo = 2
                self.reserve_ammo = 6
                self.range_limit = 5
                self.royal_guard_active = False
                self.melee_damage = 0.0

            def reset(self):
                return state.copy(), {"reset_ready": True}

            def get_action_mask(self):
                return action_mask.copy()

            def step(self, action):
                return state.copy(), 0.5, True, False, {
                    "action": action,
                    "input_succeeded": True,
                    "transition_ready": True,
                }

        class FakeAgent:
            def __init__(self, **kwargs):
                self.policy_net = None

            def load(self, path):
                return None

            def act(self, state, action_mask=None, epsilon=0.1):
                return 4

            def remember(self, *args):
                return None

            def replay(self, batch_size=32):
                return 0.0

            def update_target_network(self):
                return None

            def save(self, path):
                return None

        with TemporaryDirectory() as directory, patch.object(
            train, "ShotgunKingEnv", FakeEnvironment
        ), patch.object(train, "DQNAgent", FakeAgent), patch.object(
            train, "SummaryWriter", None
        ):
            train.train_dqn(episodes=1, trace_directory=directory, verbose=False)
            paths = list(Path(directory).glob("*.jsonl"))
            payload = json.loads(paths[0].read_text(encoding="utf-8"))

        self.assertEqual(len(paths), 1)
        self.assertEqual(payload["mode"], "autonomous")
        self.assertEqual(payload["decision"]["selected_action"], 4)
        self.assertEqual(payload["decision"]["selection_source"], "epsilon_greedy")
        self.assertTrue(payload["result"]["accepted_for_replay"])

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
            mode="autonomous",
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
