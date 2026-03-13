import unittest
import numpy as np
from engine.action_space import ACTIONS, ACTION_TO_INDEX, INDEX_TO_ACTION, legal_action_mask, action_to_engine_move
from engine.game import MorabarabaGame
from engine.constants import Player, Phase

class TestActionSpace(unittest.TestCase):
    def test_action_space_size(self):
        """Verify the total number of actions is 1200."""
        self.assertEqual(len(ACTIONS), 1200)
        self.assertEqual(len(ACTION_TO_INDEX), 1200)
        self.assertEqual(len(INDEX_TO_ACTION), 1200)

    def test_action_mapping_integrity(self):
        """Verify index -> action -> index roundtrip."""
        for i in range(1200):
            action = INDEX_TO_ACTION[i]
            self.assertEqual(ACTION_TO_INDEX[action], i)

    def test_placing_mask(self):
        """Verify mask in initial placing phase."""
        game = MorabarabaGame()
        state = game.get_state()
        mask = legal_action_mask(state)
        
        # In initial state, all 24 positions should be legal for placement
        self.assertEqual(np.sum(mask), 24)
        # Indices 0-23 are placing actions
        self.assertTrue(np.all(mask[0:24] == 1.0))
        # No other actions should be legal
        self.assertTrue(np.all(mask[24:] == 0.0))

    def test_capture_mask(self):
        """Verify mask during pending capture."""
        game = MorabarabaGame()
        # Set up a mill formation for WHITE
        game.apply_move({"type": "place", "to": 0}) # W
        game.apply_move({"type": "place", "to": 8}) # B
        game.apply_move({"type": "place", "to": 1}) # W
        game.apply_move({"type": "place", "to": 9}) # B
        game.apply_move({"type": "place", "to": 2}) # W - Mill!
        
        state = game.get_state()
        self.assertTrue(state["pending_capture"])
        mask = legal_action_mask(state)
        
        # Only capture actions should be legal
        # Black pieces are at 8 and 9
        # Capture indices are 1176 - 1199
        # 1176 + 8 = 1184, 1176 + 9 = 1185
        self.assertEqual(np.sum(mask), 2)
        self.assertEqual(mask[1176 + 8], 1.0)
        self.assertEqual(mask[1176 + 9], 1.0)
        
        # Verify other 22 capture actions are 0
        capture_indices = np.where(mask == 1.0)[0]
        self.assertTrue(all(1176 <= idx < 1200 for idx in capture_indices))

    def test_moving_mask_adjacency(self):
        """Verify adjacency is enforced in moving phase mask."""
        game = MorabarabaGame()
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White piece at 0, neighbors are 1, 7, 8
        game.board.place_piece(0, Player.WHITE)
        # Empty neighbors: 1, 7, 8
        
        state = game.get_state()
        mask = legal_action_mask(state)
        
        # Moving actions start at index 24
        # (MOVING, 0, 1, -1) -> 24 + (0*24 + 1) = 25
        # (MOVING, 0, 7, -1) -> 24 + (0*24 + 7) = 31
        # (MOVING, 0, 8, -1) -> 24 + (0*24 + 8) = 32
        
        self.assertEqual(np.sum(mask), 3)
        self.assertEqual(mask[25], 1.0)
        self.assertEqual(mask[31], 1.0)
        self.assertEqual(mask[32], 1.0)

    def test_action_to_engine_move_conversion(self):
        """Verify index -> move dict conversion."""
        # Index 0: (PLACING, -1, 0, -1)
        move0 = action_to_engine_move(0)
        self.assertEqual(move0, {"type": "place", "to": 0})
        
        # Index 25: (MOVING, 0, 1, -1)
        move25 = action_to_engine_move(25)
        self.assertEqual(move25, {"type": "move", "from": 0, "to": 1})
        
        # Index 1176 + 7 = 1183: (CAPTURE, -1, -1, 7)
        move1183 = action_to_engine_move(1183)
        self.assertEqual(move1183, {"type": "capture", "position_captured": 7})

if __name__ == '__main__':
    unittest.main()
