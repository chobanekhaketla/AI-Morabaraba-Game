"""
COMPREHENSIVE TEST SUITE FOR MORABARABA ENVIRONMENT
Tests all edge cases, reward scenarios, and DQN-critical behaviors
"""

import unittest
import numpy as np
from engine.env import MorabarabaEnv
from engine.action_space import ACTION_TO_INDEX, legal_action_mask
from engine.constants import Player, Phase, PIECES_PER_PLAYER
from engine.game import MorabarabaGame


class TestMorabarabaEnvCore(unittest.TestCase):
    """Core environment functionality tests"""
    
    def setUp(self):
        self.env = MorabarabaEnv()

    def test_reset_returns_initial_state(self):
        """Verify environment reset returns clean initial state."""
        state = self.env.reset()
        
        # Check all required keys
        required_keys = ["board", "phase", "current_player", "pending_capture", 
                        "game_over", "winner", "pieces_to_place"]
        for key in required_keys:
            self.assertIn(key, state)
        
        # Check initial values
        self.assertEqual(state["phase"], "PLACING")
        self.assertEqual(state["current_player"], 1)  # WHITE
        self.assertFalse(state["game_over"])
        self.assertFalse(state["pending_capture"])
        self.assertIsNone(state["winner"])
        self.assertFalse(self.env.done)
        
        # Check board is empty
        for pos_str, val in state["board"].items():
            self.assertEqual(val, 0, f"Position {pos_str} should be empty")
    
    def test_reset_multiple_times(self):
        """Verify reset can be called multiple times safely."""
        # Play some moves
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 0, -1)])
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 1, -1)])
        
        # Reset
        state1 = self.env.reset()
        self.assertEqual(state1["phase"], "PLACING")
        self.assertFalse(self.env.done)
        
        # Play more moves
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 5, -1)])
        
        # Reset again
        state2 = self.env.reset()
        self.assertEqual(state2["phase"], "PLACING")
        self.assertFalse(self.env.done)
    
    def test_step_returns_correct_tuple_structure(self):
        """Verify step returns (state, reward, done, info) tuple."""
        action_idx = ACTION_TO_INDEX[("PLACING", -1, 0, -1)]
        result = self.env.step(action_idx)
        
        # Should return 4-tuple
        self.assertEqual(len(result), 4)
        state, reward, done, info = result
        
        # Check types
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


class TestMorabarabaEnvRewards(unittest.TestCase):
    """Comprehensive reward testing"""
    
    def setUp(self):
        self.env = MorabarabaEnv()
    
    def test_normal_placement_zero_reward(self):
        """Verify normal placement with no mill gives 0 reward."""
        action_idx = ACTION_TO_INDEX[("PLACING", -1, 0, -1)]
        state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["mills_formed"], 0)
    
    def test_mill_formation_reward(self):
        """Verify forming a mill gives +1.0 reward."""
        # Setup mill: 0, 1, 2
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 0, -1)])  # W
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 8, -1)])  # B
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 1, -1)])  # W
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 9, -1)])  # B
        
        # Form mill
        action_idx = ACTION_TO_INDEX[("PLACING", -1, 2, -1)]  # W
        state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["mills_formed"], 1)
        self.assertTrue(state["pending_capture"])
    
    def test_capture_reward(self):
        """Verify capture gives +1.0 reward."""
        # Form mill first
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 0, -1)])  # W
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 8, -1)])  # B
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 1, -1)])  # W
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 9, -1)])  # B
        self.env.step(ACTION_TO_INDEX[("PLACING", -1, 2, -1)])  # W - Mill
        
        # Capture
        action_idx = ACTION_TO_INDEX[("CAPTURE", -1, -1, 8)]
        state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["piece_captured"])
    
    def test_win_reward_positive(self):
        """Verify winning gives +10.0 reward (plus any action rewards)."""
        # Setup win scenario
        self.env.game.phase = Phase.MOVING
        self.env.game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White has pieces at 0,1,2; Black has 2 pieces at 10,11
        for i in [0, 1, 2]:
            self.env.game.board.place_piece(i, Player.WHITE)
        for i in [10, 11]:
            self.env.game.board.place_piece(i, Player.BLACK)
        
        self.env.game.current_player = Player.WHITE
        self.env.game.pending_capture = True
        
        # Capture to win
        action_idx = ACTION_TO_INDEX[("CAPTURE", -1, -1, 10)]
        state, reward, done, info = self.env.step(action_idx)
        
        # Reward: +1.0 (capture) + 10.0 (win) = 11.0
        self.assertEqual(reward, 11.0)
        self.assertTrue(done)
        self.assertEqual(state["winner"], 1)  # WHITE

    def test_illegal_action_penalty(self):
        """Verify illegal action gives -1.0 penalty and ends episode."""
        # Try to use MOVING action in PLACING phase
        action_idx = ACTION_TO_INDEX[("MOVING", 0, 1, -1)]
        state, reward, done, info = self.env.step(action_idx)
        
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)
        self.assertTrue(info["illegal_action"])


class TestMorabarabaEnvIllegalActions(unittest.TestCase):
    """Test illegal action handling - critical for DQN stability"""
    
    def setUp(self):
        self.env = MorabarabaEnv()
    
    def test_illegal_action_ends_episode(self):
        """Verify illegal actions terminate the episode."""
        action_idx = ACTION_TO_INDEX[("MOVING", 0, 1, -1)]
        state, reward, done, info = self.env.step(action_idx)
        self.assertTrue(done)
        self.assertTrue(self.env.done)
    
    def test_action_after_done_returns_zero_reward(self):
        """Verify actions after episode end return 0 reward and stay done."""
        self.env.step(ACTION_TO_INDEX[("MOVING", 0, 1, -1)])
        self.assertTrue(self.env.done)
        
        state, reward, done, info = self.env.step(ACTION_TO_INDEX[("PLACING", -1, 5, -1)])
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertIn("error", info)


class TestMorabarabaEnvPhaseTransitions(unittest.TestCase):
    """Test phase transitions during gameplay"""
    
    def setUp(self):
        self.env = MorabarabaEnv()
    
    def test_placing_to_moving_transition(self):
        """Verify transition from PLACING to MOVING after 24 pieces."""
        for pos in range(24):
            if self.env.game.game_over:
                break
                
            state, reward, done, info = self.env.step(
                ACTION_TO_INDEX[("PLACING", -1, pos, -1)]
            )
            
            # If a mill was formed, we MUST capture an opponent piece
            # to allow the next placement to proceed.
            if state["pending_capture"] and not done:
                mask = legal_action_mask(state)
                # Find any legal capture action
                capture_indices = np.where(mask == 1.0)[0]
                # Filter to capture block (1176-1199)
                capture_indices = [idx for idx in capture_indices if 1176 <= idx < 1200]
                if capture_indices:
                    self.env.step(capture_indices[0])

        final_state = self.env.game.get_state()
        self.assertIn(final_state["phase"], ["MOVING", "FLYING"])
        self.assertEqual(sum(self.env.game.pieces_placed.values()), 24)

    def test_moving_to_flying_transition(self):
        """Verify transition to FLYING when player reaches 3 pieces."""
        self.env.game.phase = Phase.MOVING
        self.env.game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        for i in [0, 1, 2, 3]: self.env.game.board.place_piece(i, Player.WHITE)
        for i in [8, 9, 10]: self.env.game.board.place_piece(i, Player.BLACK)
        
        self.env.game.current_player = Player.BLACK
        # Force phase update
        self.env.game._update_phase()
        self.assertEqual(self.env.game.phase, Phase.FLYING)


class TestMorabarabaEnvEncoding(unittest.TestCase):
    """Test state encoding for neural network input"""
    
    def setUp(self):
        self.env = MorabarabaEnv()
    
    def test_encoding_shape_and_type(self):
        encoding = self.env.get_encoding()
        self.assertEqual(encoding.shape, (24,))
        self.assertEqual(encoding.dtype, np.float32)

    def test_encoding_multiple_pieces(self):
        """Verify encoding with multiple pieces on board, no mills."""
        white_positions = [0, 3, 6]
        black_positions = [8, 11, 14]
        
        for i in range(3):
            # White move
            self.env.step(ACTION_TO_INDEX[("PLACING", -1, white_positions[i], -1)])
            # Black move
            self.env.step(ACTION_TO_INDEX[("PLACING", -1, black_positions[i], -1)])
        
        # Current player is now WHITE (after 6 absolute steps)
        # Wait, step 1 (W), 2 (B), 3 (W), 4 (B), 5 (W), 6 (B). 
        # After step 6, current player is WHITE.
        encoding = self.env.get_encoding(Player.WHITE)
        
        for pos in white_positions:
            self.assertEqual(encoding[pos], 1.0)
        for pos in black_positions:
            self.assertEqual(encoding[pos], -1.0)


class TestMorabarabaEnvDeterminism(unittest.TestCase):
    """Test deterministic behavior"""
    
    def setUp(self):
        self.env = MorabarabaEnv()
    
    def test_same_actions_same_results(self):
        actions = [
            ACTION_TO_INDEX[("PLACING", -1, 0, -1)],
            ACTION_TO_INDEX[("PLACING", -1, 8, -1)],
        ]
        
        env1 = MorabarabaEnv()
        s1_moves = [env1.step(a)[0] for a in actions]
        
        env2 = MorabarabaEnv()
        s2_moves = [env2.step(a)[0] for a in actions]
        
        for s1, s2 in zip(s1_moves, s2_moves):
            self.assertEqual(s1["board"], s2["board"])

    def test_reset_determinism(self):
        state1 = self.env.reset()
        state2 = self.env.reset()
        self.assertEqual(state1["board"], state2["board"])
        self.assertEqual(state1["phase"], state2["phase"])



class TestMorabarabaEnvLimits(unittest.TestCase):
    """Test environment limits and boundary conditions"""
    
    def test_move_limit_termination(self):
        """Verify episode terminates with draw after max_moves."""
        # Initialize with a very small move limit
        limit = 10
        env = MorabarabaEnv(max_moves=limit)
        env.reset()
        
        # Play 9 legal moves
        for i in range(limit - 1):
            state, reward, done, info = env.step(ACTION_TO_INDEX[("PLACING", -1, i, -1)])
            self.assertFalse(done, f"Should not be done at move {i+1}")
            self.assertEqual(reward, 0.0)
            self.assertEqual(env.move_count, i + 1)
            # Keys should now exist by default
            self.assertFalse(info["draw"])
            self.assertIsNone(info["termination_reason"])
            
        # Play the 10th move (reaches limit)
        state, reward, done, info = env.step(ACTION_TO_INDEX[("PLACING", -1, limit - 1, -1)])
        
        self.assertTrue(done, "Should be done at move 10 (limit reached)")
        self.assertEqual(reward, 0.0, "Should have neutral reward for move limit draw")
        self.assertTrue(info["draw"], "Info should indicate a draw")
        self.assertEqual(info["termination_reason"], "move_limit")
        self.assertTrue(info["game_over"])
        self.assertEqual(env.move_count, limit)

    def test_move_limit_reset_behavior(self):
        """Verify move_count is reset on env.reset()"""
        env = MorabarabaEnv(max_moves=10)
        env.reset()
        env.step(ACTION_TO_INDEX[("PLACING", -1, 0, -1)])
        self.assertEqual(env.move_count, 1)
        
        env.reset()
        self.assertEqual(env.move_count, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
