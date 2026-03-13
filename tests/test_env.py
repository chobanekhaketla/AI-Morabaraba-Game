import unittest
import numpy as np
from engine.env import MorabarabaEnv
from engine.action_space import ACTION_TO_INDEX
from engine.constants import Player, Phase

class TestMorabarabaEnv(unittest.TestCase):
    def setUp(self):
        self.env = MorabarabaEnv()

    def test_reset(self):
        """Verify environment reset returns initial state."""
        state = self.env.reset()
        self.assertEqual(state["phase"], "PLACING")
        self.assertEqual(state["current_player"], 1) # WHITE
        self.assertFalse(state["game_over"])
        self.assertFalse(self.env.done)

    def test_step_placement_and_reward(self):
        """Verify placement move updates state and gives 0 reward (no mill)."""
        # Action: Place at position 0
        action_idx = ACTION_TO_INDEX[("PLACING", -1, 0, -1)]
        next_state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(next_state["board"]["0"], 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["mills_formed"], 0)
        self.assertFalse(info["piece_captured"])

    def test_step_mill_and_reward(self):
        """Verify forming a mill gives +1 reward."""
        # Setup mill for White: 0, 1, 2
        # White at 0
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 0, -1)])
        # Black at 8
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 8, -1)])
        # White at 1
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 1, -1)])
        # Black at 9
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 9, -1)])
        
        # White at 2 -> Mill!
        action_idx = ACTION_TO_INDEX[("PLACING", -1, 2, -1)]
        next_state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, 1.0)
        self.assertTrue(next_state["pending_capture"])
        self.assertEqual(info["mills_formed"], 1)

    def test_step_capture_and_reward(self):
        """Verify capture gives +1 reward."""
        # Setup mill
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 0, -1)]) # W
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 8, -1)]) # B
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 1, -1)]) # W
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 9, -1)]) # B
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 2, -1)]) # W mill
        
        # Capture Black's piece at 8
        action_idx = ACTION_TO_INDEX[("CAPTURE", -1, -1, 8)]
        next_state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, 1.0)
        self.assertFalse(next_state["pending_capture"])
        self.assertTrue(info["piece_captured"])

    def test_illegal_action_penalty(self):
        """Verify illegal actions are penalized and end episode."""
        # Try to move from non-existent piece in PLACING phase
        action_idx = ACTION_TO_INDEX[("MOVING", 0, 1, -1)]
        next_state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)
        self.assertTrue(info["illegal_action"])

    def test_get_encoding(self):
        """Verify board encoding returns correct shape/type."""
        encoding = self.env.get_encoding()
        self.assertIsInstance(encoding, np.ndarray)
        self.assertEqual(encoding.shape, (24,))
        self.assertEqual(encoding.dtype, np.float32)

    def test_win_reward(self):
        """Verify winning rewards +10 (+1 if mill/capture occurs same step)."""
        # We'll just set up the game state directly
        self.env.game.phase = Phase.MOVING
        self.env.game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White mill at 0,1,2. Black has 3 pieces at 10,11,12.
        # Capturing 10 will reduce Black to 2.
        for i in [0, 1, 2]: self.env.game.board.place_piece(i, Player.WHITE)
        for i in [10, 11, 12]: self.env.game.board.place_piece(i, Player.BLACK)
        
        self.env.game.current_player = Player.WHITE
        self.env.game.pending_capture = True
        
        action_idx = ACTION_TO_INDEX[("CAPTURE", -1, -1, 10)]
        next_state, reward, done, info = self.env.step(action_idx)
        
        # Reward: +1.0 (capture) + 10.0 (win) = 11.0
        self.assertEqual(reward, 11.0)
        self.assertTrue(done)
        self.assertEqual(next_state["winner"], 1)

if __name__ == '__main__':
    unittest.main()
