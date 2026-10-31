"""Card layouts tolerate coherent shifts without producing board pieces."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shotgun_king_rl.vision.screens import check_card_selection_screen
from shotgun_king_rl.collection.prepare import draft_annotation


class CardScreenProposalTests(unittest.TestCase):
    def image(self, dy=0, dx=0):
        image = np.zeros((720, 1280, 3), np.uint8)
        for y in (220, 328, 420, 528):
            for x in (545, 619, 656, 730):
                image[y + dy, x + dx] = 255
        return image

    def test_shared_shift_is_accepted_but_partial_layout_is_not(self):
        self.assertTrue(check_card_selection_screen(self.image(3, -2)))
        image = self.image(3, -2)
        image[223, 543] = 0
        self.assertFalse(check_card_selection_screen(image))
        self.assertFalse(check_card_selection_screen(np.zeros_like(image)))

    def test_non_gameplay_proposal_skips_piece_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('shotgun_king_rl.collection.prepare.propose', return_value={'screen_state': {'screen_kind': 'card_selection'}}), patch('shotgun_king_rl.collection.prepare.analyzer.get_state_matrix') as read:
                result = draft_annotation(self.image(), 'screen.png', Path(directory), 'session', 'train', crop=[0, 0, 1280, 720])
        read.assert_not_called()
        self.assertTrue(all(cell == 'empty' for row in result['board'] for cell in row))
        self.assertFalse(result['confirmed'])
