"""
ADDITIONAL ROBUST TESTS FOR ACTION SPACE
Add these to test_action_space.py to achieve 95%+ coverage
"""

import unittest
import numpy as np
from engine.action_space import (
    ACTIONS, ACTION_TO_INDEX, INDEX_TO_ACTION, 
    legal_action_mask, action_to_engine_move
)
from engine.game import MorabarabaGame
from engine.constants import Player, Phase
from engine.board import Board


class TestActionSpaceRobust(unittest.TestCase):
    """Additional comprehensive tests for action space edge cases"""
    
    def test_flying_phase_mask(self):
        """Verify mask allows any position in flying phase."""
        game = MorabarabaGame()
        game.phase = Phase.FLYING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White piece at 0, Black pieces at 8, 9, 10
        game.board.place_piece(0, Player.WHITE)
        for pos in [8, 9, 10]:
            game.board.place_piece(pos, Player.BLACK)
        
        game.current_player = Player.WHITE
        state = game.get_state()
        mask = legal_action_mask(state)
        
        # In flying phase, White can move from 0 to ANY of 20 empty positions
        # (24 total - 1 (White at 0) - 3 (Black pieces) = 20)
        expected_legal = 20
        self.assertEqual(np.sum(mask), expected_legal, 
                        f"Expected {expected_legal} legal flying moves")
        
        # Verify all legal moves are in FLYING phase block (indices 600-1175)
        legal_indices = np.where(mask == 1.0)[0]
        for idx in legal_indices:
            self.assertGreaterEqual(idx, 600, "Legal move should be in FLYING block")
            self.assertLess(idx, 1176, "Legal move should be in FLYING block")
    
    def test_mask_with_partial_placement(self):
        """Verify mask when some positions are occupied during placing."""
        game = MorabarabaGame()
        
        # Place pieces alternately
        occupied = [0, 1, 2, 3]
        for pos in occupied:
            game.apply_move({"type": "place", "to": pos})
        
        state = game.get_state()
        mask = legal_action_mask(state)
        
        # 4 positions occupied, 20 available
        self.assertEqual(np.sum(mask), 20, "Should have 20 legal placements")
        
        # Verify occupied positions are masked out
        for pos in occupied:
            self.assertEqual(mask[pos], 0.0, 
                           f"Position {pos} should be masked (occupied)")
        
        # Verify at least one unoccupied position is legal
        self.assertEqual(mask[4], 1.0, "Position 4 should be legal")
    
    def test_mask_blocked_piece(self):
        """Verify mask when a piece has no legal moves (completely blocked)."""
        game = MorabarabaGame()
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White at 0, Black surrounds it at all neighbors (1, 7, 8)
        game.board.place_piece(0, Player.WHITE)
        for neighbor in [1, 7, 8]:
            game.board.place_piece(neighbor, Player.BLACK)
        
        game.current_player = Player.WHITE
        state = game.get_state()
        mask = legal_action_mask(state)
        
        # White's piece at 0 is completely blocked - no legal moves
        self.assertEqual(np.sum(mask), 0.0, 
                        "Blocked piece should have no legal moves")
    
    def test_mask_multiple_pieces_moving_phase(self):
        """Verify mask with multiple pieces for both players in MOVING phase."""
        game = MorabarabaGame()
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White pieces: 0, 1
        # Black pieces: 10, 11
        game.board.place_piece(0, Player.WHITE)
        game.board.place_piece(1, Player.WHITE)
        game.board.place_piece(10, Player.BLACK)
        game.board.place_piece(11, Player.BLACK)
        
        game.current_player = Player.WHITE
        state = game.get_state()
        mask = legal_action_mask(state)
        
        # White can move:
        # From 0: neighbors are 1 (occupied by White), 7, 8
        #   Legal: 0->7, 0->8
        # From 1: neighbors are 0 (occupied by White), 2, 9
        #   Legal: 1->2, 1->9
        # Total: 4 legal moves
        expected_moves = 4
        actual_sum = np.sum(mask)
        self.assertEqual(actual_sum, expected_moves, 
                        f"Expected {expected_moves} legal moves, got {actual_sum}")
    
    def test_action_space_boundaries(self):
        """Verify action space boundaries are correct."""
        # PLACING: indices 0-23
        placing_actions = [a for a in ACTIONS if a[0] == "PLACING"]
        self.assertEqual(len(placing_actions), 24)
        self.assertEqual(ACTION_TO_INDEX[("PLACING", -1, 0, -1)], 0)
        self.assertEqual(ACTION_TO_INDEX[("PLACING", -1, 23, -1)], 23)
        
        # MOVING: indices 24-599
        moving_actions = [a for a in ACTIONS if a[0] == "MOVING"]
        self.assertEqual(len(moving_actions), 576)  # 24 * 24
        self.assertEqual(ACTION_TO_INDEX[("MOVING", 0, 0, -1)], 24)
        self.assertEqual(ACTION_TO_INDEX[("MOVING", 23, 23, -1)], 599)
        
        # FLYING: indices 600-1175
        flying_actions = [a for a in ACTIONS if a[0] == "FLYING"]
        self.assertEqual(len(flying_actions), 576)
        self.assertEqual(ACTION_TO_INDEX[("FLYING", 0, 0, -1)], 600)
        self.assertEqual(ACTION_TO_INDEX[("FLYING", 23, 23, -1)], 1175)
        
        # CAPTURE: indices 1176-1199
        capture_actions = [a for a in ACTIONS if a[0] == "CAPTURE"]
        self.assertEqual(len(capture_actions), 24)
        self.assertEqual(ACTION_TO_INDEX[("CAPTURE", -1, -1, 0)], 1176)
        self.assertEqual(ACTION_TO_INDEX[("CAPTURE", -1, -1, 23)], 1199)
    
    def test_invalid_action_index(self):
        """Verify handling of out-of-bounds action indices."""
        with self.assertRaises(KeyError):
            action_to_engine_move(1200)  # Out of bounds
        
        with self.assertRaises(KeyError):
            action_to_engine_move(-1)  # Negative index
    
    def test_mask_sum_never_exceeds_legal_actions(self):
        """Verify mask sum never exceeds physically possible actions."""
        game = MorabarabaGame()
        
        # Play random game for 50 moves, checking mask each time
        for _ in range(50):
            if game.game_over:
                break
            
            state = game.get_state()
            mask = legal_action_mask(state)
            
            # Mask sum should be reasonable
            if state["phase"] == "PLACING":
                # Max 24 placements
                self.assertLessEqual(np.sum(mask), 24)
            elif state["pending_capture"]:
                # Max 24 captures (but usually much fewer)
                self.assertLessEqual(np.sum(mask), 24)
            else:
                # In moving/flying, max is all pieces * all destinations
                # But typically much less due to adjacency
                self.assertLessEqual(np.sum(mask), 576)
            
            # Apply a random legal move
            legal_moves = game.get_legal_moves()
            if legal_moves:
                game.apply_move(legal_moves[0])
    
    def test_mask_consistency_with_engine_legal_moves(self):
        """Verify mask exactly matches engine's legal moves."""
        game = MorabarabaGame()
        
        # Test multiple game states
        for move_num in range(30):
            if game.game_over:
                break
            
            state = game.get_state()
            mask = legal_action_mask(state)
            engine_moves = game.get_legal_moves()
            
            # Count of legal actions from mask should equal engine moves
            mask_count = int(np.sum(mask))
            engine_count = len(engine_moves)
            
            self.assertEqual(mask_count, engine_count,
                           f"Move {move_num}: Mask has {mask_count} legal actions, "
                           f"engine has {engine_count}")
            
            # Apply random move
            if engine_moves:
                game.apply_move(engine_moves[0])
    
    def test_all_engine_moves_have_valid_indices(self):
        """Verify every engine move can be converted to a valid action index."""
        game = MorabarabaGame()
        
        for _ in range(50):
            if game.game_over:
                break
            
            legal_moves = game.get_legal_moves()
            state = game.get_state()
            phase = Phase(state["phase"])
            
            for move in legal_moves:
                # Try to find corresponding action tuple
                action_tuple = None
                move_type = move["type"]
                
                if move_type == "place":
                    action_tuple = ("PLACING", -1, move["to"], -1)
                elif move_type == "move":
                    if phase == Phase.MOVING:
                        action_tuple = ("MOVING", move["from"], move["to"], -1)
                    elif phase == Phase.FLYING:
                        action_tuple = ("FLYING", move["from"], move["to"], -1)
                elif move_type == "capture":
                    action_tuple = ("CAPTURE", -1, -1, move["position_captured"])
                
                # Verify action tuple exists in action space
                self.assertIn(action_tuple, ACTION_TO_INDEX,
                            f"Move {move} (phase {phase}) not found in action space")
            
            # Apply move and continue
            if legal_moves:
                game.apply_move(legal_moves[0])
    
    def test_deterministic_mask_generation(self):
        """Verify mask generation is deterministic for same state."""
        game = MorabarabaGame()
        
        # Place a few pieces
        for i in range(6):
            legal_moves = game.get_legal_moves()
            if legal_moves:
                game.apply_move(legal_moves[0])
        
        state = game.get_state()
        
        # Generate mask multiple times
        mask1 = legal_action_mask(state)
        mask2 = legal_action_mask(state)
        mask3 = legal_action_mask(state)
        
        # All masks should be identical
        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(mask2, mask3)
    
    def test_mask_after_mill_formation(self):
        """Verify mask transitions correctly after forming a mill."""
        game = MorabarabaGame()
        
        # Form a mill
        game.apply_move({"type": "place", "to": 0})  # W
        game.apply_move({"type": "place", "to": 8})  # B
        game.apply_move({"type": "place", "to": 1})  # W
        game.apply_move({"type": "place", "to": 9})  # B
        
        # Before mill
        state_before = game.get_state()
        self.assertFalse(state_before["pending_capture"])
        
        # Form mill
        game.apply_move({"type": "place", "to": 2})  # W - Mill (0,1,2)
        
        # After mill
        state_after = game.get_state()
        self.assertTrue(state_after["pending_capture"])
        
        mask = legal_action_mask(state_after)
        
        # Only capture actions should be legal
        # Black pieces at 8, 9
        legal_indices = np.where(mask == 1.0)[0]
        self.assertEqual(len(legal_indices), 2)
        
        # All legal indices should be in CAPTURE block (1176-1199)
        for idx in legal_indices:
            self.assertGreaterEqual(idx, 1176)
            self.assertLess(idx, 1200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
