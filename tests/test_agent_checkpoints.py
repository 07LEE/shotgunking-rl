"""Small CPU-only checkpoint checks; no game or replay learning."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from agent import DQNAgent
from shotgun_king_rl.state_schema import CHECKPOINT_FORMAT, OBSERVATION_SCHEMA


class AgentCheckpointTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]), patch.object(torch.cuda, 'is_available', return_value=False):
            torch.manual_seed(0)
            self.agent = DQNAgent(state_size=np.int64(345), action_size=np.int64(10))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'model.pth'

    def payload(self):
        return {
            'format': CHECKPOINT_FORMAT,
            'observation_schema': OBSERVATION_SCHEMA,
            'state_size': 345,
            'action_size': 10,
            'policy_state_dict': {key: torch.full_like(value, .125) for key, value in self.agent.policy_net.state_dict().items()},
        }

    def assert_rejected_without_changes(self, payload):
        policy = {key: value.clone() for key, value in self.agent.policy_net.state_dict().items()}
        target = {key: value.clone() for key, value in self.agent.target_net.state_dict().items()}
        torch.save(payload, self.path)
        original_file = self.path.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertFalse(self.agent.load(str(self.path)))
        self.assertEqual(self.path.read_bytes(), original_file)
        for network, before in ((self.agent.policy_net, policy), (self.agent.target_net, target)):
            for key, value in network.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)

    def test_save_load_round_trip_records_schema_and_syncs_both_networks(self):
        self.agent.policy_net.load_state_dict(self.payload()['policy_state_dict'])
        with redirect_stdout(io.StringIO()):
            self.agent.save(str(self.path))
        checkpoint = torch.load(self.path, map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['observation_schema'], OBSERVATION_SCHEMA)
        self.assertEqual(checkpoint['format'], CHECKPOINT_FORMAT)
        self.assertEqual((checkpoint['state_size'], checkpoint['action_size']), (345, 10))
        with torch.no_grad():
            for parameter in self.agent.policy_net.parameters():
                parameter.zero_()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.agent.load(str(self.path)))
        for network in (self.agent.policy_net, self.agent.target_net):
            for key, value in network.state_dict().items():
                self.assertTrue(torch.equal(value, checkpoint['policy_state_dict'][key]), key)

    def test_legacy_raw_weights_are_rejected_even_with_identical_shapes(self):
        self.assert_rejected_without_changes(self.payload()['policy_state_dict'])

    def test_wrong_schema_dimensions_and_missing_metadata_are_rejected(self):
        for key, value in (('observation_schema', 'distinct-royals-281-v1'), ('format', 'other'), ('state_size', 281), ('action_size', 18), ('state_size', True)):
            with self.subTest(key=key, value=value):
                self.assert_rejected_without_changes(self.payload() | {key: value})
        payload = self.payload()
        del payload['observation_schema']
        self.assert_rejected_without_changes(payload)

    def test_invalid_weight_shape_is_rejected_before_partial_parameter_load(self):
        payload = self.payload()
        weights = payload['policy_state_dict']
        key = list(weights)[-1]
        weights[key] = torch.zeros(1)
        self.assert_rejected_without_changes(payload)

    def test_forward_accepts_signed_pieces_and_terrain(self):
        observation = torch.zeros((1, 345))
        observation[0, :3] = torch.tensor([1, -5, 9])
        observation[0, 281 + 4 * 8:281 + 5 * 8] = 1
        with torch.no_grad():
            result = self.agent.policy_net(observation)
        self.assertEqual(result.shape, (1, 10))
        self.assertTrue(torch.isfinite(result).all())

    def test_policy_output_changes_when_only_terrain_changes(self):
        observation = torch.zeros((1, 345))
        observation[0, :3] = torch.tensor([1, -5, 9])
        with torch.no_grad():
            without_moat = self.agent.policy_net(observation)
            observation[0, 281 + 4 * 8:281 + 5 * 8] = 1
            with_moat = self.agent.policy_net(observation)
        self.assertFalse(torch.equal(without_moat, with_moat))


if __name__ == '__main__':
    unittest.main()
