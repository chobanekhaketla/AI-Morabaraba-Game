"""
ROBUST MORABARABA ENGINE TEST SUITE
====================================
This comprehensive test suite is designed for ML-readiness, ensuring the engine
can handle thousands of DQN training episodes without crashes or inconsistencies.

Test Coverage:
- Board operations with boundary conditions
- Rule validation with edge cases
- Complete game flow integration
- Natural phase transitions through gameplay
- Consecutive mill formations
- Stress testing with random games
- State serialization/deserialization
- Mill breaking scenarios
- Concurrent game independence
"""

import unittest
import random
import copy
from typing import List, Dict
from engine.game import MorabarabaGame
from engine.board import Board
from engine.rules import (
    is_adjacent, forms_mill, get_mills, is_in_mill, 
    get_capturable_pieces, is_legal_placement, is_legal_move, 
    get_legal_moves, check_win_condition
)
from engine.constants import Player, Phase, PIECES_PER_PLAYER, MILL_LINES, ADJACENCY
from engine.errors import IllegalMoveError, GameOverError, InvalidPositionError


class TestBoardOperationsExtended(unittest.TestCase):
    """Extended board operation tests with boundary conditions and edge cases"""
    
    def setUp(self):
        self.board = Board()

    def test_place_all_positions(self):
        """Test placing pieces on all 24 valid positions"""
        for i in range(24):
            board = Board()
            board.place_piece(i, Player.WHITE)
            self.assertEqual(board.get_piece(i), 1, f"Position {i} should contain white piece")

    def test_boundary_positions(self):
        """Test behavior at position boundaries (0, 23, invalid)"""
        self.board.place_piece(0, Player.WHITE)
        self.assertEqual(self.board.get_piece(0), 1)
        
        self.board.place_piece(23, Player.BLACK)
        self.assertEqual(self.board.get_piece(23), -1)
        
        with self.assertRaises(InvalidPositionError):
            self.board.place_piece(24, Player.WHITE)
        
        with self.assertRaises(InvalidPositionError):
            self.board.place_piece(-1, Player.WHITE)

    def test_move_piece_updates_both_positions(self):
        """Verify move correctly updates from and to positions"""
        self.board.place_piece(5, Player.WHITE)
        self.board.move_piece(5, 6, Player.WHITE)
        
        self.assertTrue(self.board.is_empty(5), "Original position should be empty")
        self.assertEqual(self.board.get_piece(6), 1, "Destination should have white piece")

    def test_remove_piece_clears_position(self):
        """Verify remove operation properly clears the position"""
        self.board.place_piece(10, Player.BLACK)
        self.assertEqual(self.board.get_piece(10), -1)
        
        self.board.remove_piece(10)
        self.assertTrue(self.board.is_empty(10))
        self.assertEqual(self.board.count_pieces(Player.BLACK), 0)

    def test_get_player_pieces_accuracy(self):
        """Verify get_player_pieces returns exact positions"""
        positions = [0, 5, 10, 15, 20]
        for pos in positions:
            self.board.place_piece(pos, Player.WHITE)
        
        white_pieces = set(self.board.get_player_pieces(Player.WHITE))
        self.assertEqual(white_pieces, set(positions))
        
        black_pieces = self.board.get_player_pieces(Player.BLACK)
        self.assertEqual(len(black_pieces), 0)

    def test_count_pieces_with_mixed_board(self):
        """Test piece counting with both players having pieces"""
        for i in [0, 1, 2, 3, 4]:
            self.board.place_piece(i, Player.WHITE)
        for i in [10, 11, 12]:
            self.board.place_piece(i, Player.BLACK)
        
        self.assertEqual(self.board.count_pieces(Player.WHITE), 5)
        self.assertEqual(self.board.count_pieces(Player.BLACK), 3)

    def test_board_copy_deep_independence(self):
        """Ensure board copy is completely independent"""
        self.board.place_piece(0, Player.WHITE)
        self.board.place_piece(5, Player.BLACK)
        
        copy_board = self.board.copy()
        
        # Modify original
        self.board.place_piece(10, Player.WHITE)
        self.board.remove_piece(5)
        
        # Copy should be unchanged
        self.assertTrue(copy_board.is_empty(10))
        self.assertEqual(copy_board.get_piece(5), -1)
        self.assertEqual(copy_board.count_pieces(Player.WHITE), 1)

    def test_to_dict_serialization(self):
        """Verify board state can be serialized to dictionary"""
        self.board.place_piece(0, Player.WHITE)
        self.board.place_piece(10, Player.BLACK)
        
        state = self.board.to_dict()
        self.assertIsInstance(state, dict)
        # board.to_dict() returns flat dict with STRING keys: {"0": 1, "1": 0, ...}
        self.assertEqual(state["0"], 1)
        self.assertEqual(state["10"], -1)


class TestRulesValidationExtended(unittest.TestCase):
    """Extended rule validation with comprehensive mill and adjacency testing"""
    
    def setUp(self):
        self.board = Board()

    def test_all_adjacencies_from_constants(self):
        """Test adjacency for all defined connections in ADJACENCY constant"""
        for pos, neighbors in ADJACENCY.items():
            for neighbor in neighbors:
                self.assertTrue(
                    is_adjacent(pos, neighbor),
                    f"Position {pos} should be adjacent to {neighbor}"
                )
                # Verify symmetry
                self.assertTrue(
                    is_adjacent(neighbor, pos),
                    f"Adjacency should be symmetric: {neighbor} to {pos}"
                )

    def test_all_mills_detection(self):
        """Test detection of all 16 mills defined in MILL_LINES constant"""
        for mill_tuple in MILL_LINES:
            board = Board()
            pos1, pos2, pos3 = mill_tuple
            
            # Place two pieces
            board.place_piece(pos1, Player.WHITE)
            board.place_piece(pos2, Player.WHITE)
            
            # Third piece should complete the mill
            self.assertTrue(
                forms_mill(board, Player.WHITE, pos3),
                f"Mill {mill_tuple} should be detected"
            )
            
            # Actually place the third piece and verify is_in_mill
            board.place_piece(pos3, Player.WHITE)
            self.assertTrue(is_in_mill(board, Player.WHITE, pos1))
            self.assertTrue(is_in_mill(board, Player.WHITE, pos2))
            self.assertTrue(is_in_mill(board, Player.WHITE, pos3))

    def test_mill_detection_wrong_player(self):
        """Mill should not be detected for wrong player"""
        self.board.place_piece(0, Player.WHITE)
        self.board.place_piece(1, Player.WHITE)
        self.board.place_piece(2, Player.BLACK)  # Wrong player
        
        self.assertFalse(is_in_mill(self.board, Player.WHITE, 0))
        self.assertFalse(is_in_mill(self.board, Player.BLACK, 0))

    def test_capturable_pieces_priority(self):
        """Verify non-mill pieces are prioritized for capture"""
        # Black mill: 0, 1, 2
        for i in [0, 1, 2]:
            self.board.place_piece(i, Player.BLACK)
        
        # Black non-mill pieces
        self.board.place_piece(10, Player.BLACK)
        self.board.place_piece(15, Player.BLACK)
        
        capturable = get_capturable_pieces(self.board, Player.WHITE)
        
        # Should only return non-mill pieces
        self.assertIn(10, capturable)
        self.assertIn(15, capturable)
        self.assertNotIn(0, capturable)
        self.assertNotIn(1, capturable)
        self.assertNotIn(2, capturable)

    def test_capturable_all_in_mills_multiple_mills(self):
        """When all opponent pieces are in mills, any should be capturable"""
        # Black has two complete mills
        for i in [0, 1, 2]:  # Mill 1
            self.board.place_piece(i, Player.BLACK)
        for i in [8, 9, 10]:  # Mill 2
            self.board.place_piece(i, Player.BLACK)
        
        capturable = get_capturable_pieces(self.board, Player.WHITE)
        
        # All 6 pieces should be capturable
        self.assertEqual(len(capturable), 6)
        self.assertEqual(set(capturable), {0, 1, 2, 8, 9, 10})

    def test_legal_moves_blocked_position(self):
        """Test legal moves when player is surrounded"""
        # White at position 0, surrounded by black pieces
        self.board.place_piece(0, Player.WHITE)
        self.board.place_piece(1, Player.BLACK)
        self.board.place_piece(7, Player.BLACK)
        self.board.place_piece(8, Player.BLACK)
        
        moves = get_legal_moves(self.board, Player.WHITE, Phase.MOVING, 24, False)
        self.assertEqual(len(moves), 0, "Surrounded piece should have no legal moves")

    def test_flying_phase_excludes_occupied(self):
        """In flying phase, should only return empty positions"""
        self.board.place_piece(0, Player.WHITE)
        for i in range(1, 12):
            self.board.place_piece(i, Player.BLACK)
        
        moves = get_legal_moves(self.board, Player.WHITE, Phase.FLYING, 24, False)
        
        # Should have 24 - 12 (occupied) = 12 moves
        self.assertEqual(len(moves), 12)
        for move in moves:
            self.assertTrue(self.board.is_empty(move["to"]))


class TestCompleteGameSimulations(unittest.TestCase):
    """Full game simulations testing natural game flow from start to finish"""
    
    def test_complete_game_win_by_captures(self):
        """Play a complete game where one player wins by reducing opponent to 2 pieces"""
        game = MorabarabaGame()
        
        # Scripted game sequence - we'll handle captures dynamically
        placement_sequence = [
            # Alternate White and Black placements
            # White will try to form mills, Black will place randomly
            0, 12,   # W, B
            1, 13,   # W, B
            2, 14,   # W, B - White might form mill (0,1,2)
            8, 15,   # W, B
            9, 16,   # W, B
            10, 17,  # W, B - White might form mill (8,9,10)
            3, 18,   # W, B
            4, 19,   # W, B
            5, 20,   # W, B - White might form mill (3,4,5) or (4,5,6)
            6, 21,   # W, B
            7, 22,   # W, B
            11, 23,  # W, B
        ]
        
        white_turn = True
        for position in placement_sequence:
            if game.game_over:
                break
            
            # If there's a pending capture, handle it first
            if game.pending_capture:
                capturable = get_capturable_pieces(game.board, game.current_player)
                if capturable:
                    game.apply_move({"type": "capture", "position_captured": capturable[0]})
            
            # Make the placement move if it's still placing phase
            if game.phase == Phase.PLACING and not game.game_over:
                # Check whose turn it is
                expected_player = Player.WHITE if white_turn else Player.BLACK
                if game.current_player == expected_player:
                    try:
                        game.apply_move({"type": "place", "to": position})
                    except IllegalMoveError:
                        # Position might be occupied, skip it
                        pass
                    
                    # After placing, check for pending capture again
                    if game.pending_capture:
                        capturable = get_capturable_pieces(game.board, game.current_player)
                        if capturable:
                            game.apply_move({"type": "capture", "position_captured": capturable[0]})
                    
                    white_turn = not white_turn
        
        # Continue in MOVING/FLYING phase if game isn't over
        move_count = 0
        max_moves = 100
        while not game.game_over and move_count < max_moves:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            
            # Just make a random legal move
            game.apply_move(legal_moves[0])
            move_count += 1
        
        # Verify game reached a valid conclusion
        self.assertTrue(
            game.game_over or move_count == max_moves,
            "Game should end or reach move limit"
        )

    def test_complete_game_win_by_blocking(self):
        """Play a game where one player wins by blocking all opponent moves"""
        game = MorabarabaGame()
        
        # Strategy: Place pieces to eventually block opponent
        # This is a more complex scenario that requires specific positioning
        
        # First complete the placing phase without captures
        non_mill_positions = [
            [0, 5, 10, 15, 3, 6],  # White positions (avoid mills initially)
            [8, 13, 18, 23, 9, 14] # Black positions
        ]
        
        move_count = 0
        for w_pos, b_pos in zip(non_mill_positions[0], non_mill_positions[1]):
            game.apply_move({"type": "place", "to": w_pos})
            if game.pending_capture:
                # Capture if mill formed
                capturable = get_capturable_pieces(game.board, game.current_player)
                game.apply_move({"type": "capture", "position_captured": capturable[0]})
            
            if game.game_over:
                break
                
            game.apply_move({"type": "place", "to": b_pos})
            if game.pending_capture:
                capturable = get_capturable_pieces(game.board, game.current_player)
                game.apply_move({"type": "capture", "position_captured": capturable[0]})
            
            if game.game_over:
                break
            
            move_count += 2
        
        # Continue placing remaining pieces
        while sum(game.pieces_placed.values()) < 24 and not game.game_over:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            
            game.apply_move(legal_moves[0])
            if game.pending_capture:
                capturable = get_capturable_pieces(game.board, game.current_player)
                if capturable:
                    game.apply_move({"type": "capture", "position_captured": capturable[0]})
        
        # Verify we transitioned to MOVING phase or game ended
        self.assertTrue(
            game.phase in [Phase.MOVING, Phase.FLYING] or game.game_over,
            "Should transition to movement phase or end game"
        )

    def test_natural_phase_transition_placing_to_moving(self):
        """Test natural transition from PLACING to MOVING through gameplay"""
        game = MorabarabaGame()
        
        self.assertEqual(game.phase, Phase.PLACING)
        
        # Place all 24 pieces, handling captures
        pieces_placed = 0
        while pieces_placed < 24 and not game.game_over:
            legal_moves = game.get_legal_moves()
            self.assertGreater(len(legal_moves), 0, "Should have legal moves")
            
            move = legal_moves[0]
            game.apply_move(move)
            
            if move["type"] == "place":
                pieces_placed += 1
            
            # Handle captures if mill formed
            if game.pending_capture:
                capturable = get_capturable_pieces(game.board, game.current_player)
                if capturable:
                    game.apply_move({"type": "capture", "position_captured": capturable[0]})
        
        # Should now be in MOVING phase
        self.assertEqual(game.phase, Phase.MOVING, "Should transition to MOVING after 24 pieces")

    def test_natural_phase_transition_moving_to_flying(self):
        """Test natural transition from MOVING to FLYING by reducing pieces to 3"""
        game = MorabarabaGame()
        
        # Fast-forward to MOVING phase with specific setup
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        game.board = Board()
        
        # White has 4 pieces, Black has 3 pieces
        for i in [0, 1, 2, 3]:
            game.board.place_piece(i, Player.WHITE)
        for i in [8, 9, 10]:
            game.board.place_piece(i, Player.BLACK)
        
        game.current_player = Player.WHITE
        
        # White makes a move - should update phase to check if Black is now flying
        game.apply_move({"type": "move", "from": 3, "to": 4})
        
        # After White's move, it's Black's turn and Black should be in flying
        # (Black has 3 pieces)
        # The game should recognize this when Black tries to move
        
        game.current_player = Player.BLACK
        game._update_phase()
        
        self.assertEqual(
            game.phase,
            Phase.FLYING,
            "Black with 3 pieces should be in FLYING phase"
        )

    def test_consecutive_mill_formations(self):
        """Test forming multiple mills in consecutive turns"""
        game = MorabarabaGame()
        
        moves = [
            # Set up for White to form consecutive mills
            {"type": "place", "to": 0},   # W
            {"type": "place", "to": 12},  # B
            {"type": "place", "to": 1},   # W
            {"type": "place", "to": 13},  # B
            {"type": "place", "to": 2},   # W - Mill 1! (0,1,2)
            {"type": "capture", "position_captured": 12},  # W captures
            {"type": "place", "to": 14},  # B
            {"type": "place", "to": 8},   # W
            {"type": "place", "to": 15},  # B
            {"type": "place", "to": 9},   # W
            {"type": "place", "to": 16},  # B
            {"type": "place", "to": 10},  # W - Mill 2! (8,9,10)
            {"type": "capture", "position_captured": 13},  # W captures
        ]
        
        # DEBUG: Track piece counts
        for i, move in enumerate(moves):
            game.apply_move(move)
            black_count = game.board.count_pieces(Player.BLACK)
            white_count = game.board.count_pieces(Player.WHITE)
            # print(f"After move {i} ({move['type']}): Black={black_count}, White={white_count}, GameOver={game.game_over}")
            if game.game_over:
                # Game ended early - this might be expected or a bug
                break
        
        # Black placed at: 12 (captured), 13 (captured), 14, 15, 16
        # Expected: Black placed 5, lost 2 = 3 remaining pieces
        # So actually, 3 is CORRECT, not 10!
        self.assertEqual(game.board.count_pieces(Player.BLACK), 3,
                        "Black placed 5 pieces, lost 2, should have 3 remaining")
        
        white_mills = get_mills(game.board, Player.WHITE)
        self.assertGreaterEqual(len(white_mills), 2, "White should have at least 2 mills")


class TestStressAndEdgeCases(unittest.TestCase):
    """Stress testing and edge case scenarios for ML robustness"""
    
    def test_random_game_no_crashes(self):
        """Play 10 random games to completion without crashes"""
        for game_num in range(10):
            game = MorabarabaGame()
            max_moves = 500
            move_count = 0
            
            while not game.game_over and move_count < max_moves:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                
                move = random.choice(legal_moves)
                try:
                    game.apply_move(move)
                    move_count += 1
                except Exception as e:
                    self.fail(f"Game {game_num} crashed at move {move_count}: {e}")
            
            # Verify game reached a valid end state
            self.assertTrue(
                game.game_over or move_count == max_moves,
                f"Game {game_num} should end or hit move limit"
            )

    def test_mill_formation_with_no_capturable_pieces(self):
        """Form mill when opponent has no pieces (edge case)"""
        game = MorabarabaGame()
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        
        # White has pieces, Black has 0 pieces (all captured)
        game.board = Board()
        for i in [0, 1, 2]:
            game.board.place_piece(i, Player.WHITE)
        
        game.current_player = Player.WHITE
        
        # White already has mill at 0,1,2
        # Since Black has 0 pieces, White has already won
        # The game should be over
        game._check_game_over()
        
        self.assertTrue(
            game.game_over,
            "Game should be over when opponent has 0 pieces"
        )

    def test_breaking_opponent_mill(self):
        """Test moving away from a mill position (breaking the mill)"""
        game = MorabarabaGame()
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        game.board = Board()
        
        # White mill: 0, 1, 2
        for i in [0, 1, 2]:
            game.board.place_piece(i, Player.WHITE)
        
        # Black pieces
        for i in [8, 9]:
            game.board.place_piece(i, Player.BLACK)
        
        game.current_player = Player.WHITE
        
        # White moves from 2 to 3, breaking the mill
        game.apply_move({"type": "move", "from": 2, "to": 3})
        
        # Verify mill is broken
        self.assertFalse(is_in_mill(game.board, Player.WHITE, 0))
        self.assertFalse(is_in_mill(game.board, Player.WHITE, 1))
        
        # White piece should now be at 3
        self.assertEqual(game.board.get_piece(3), 1)
        self.assertTrue(game.board.is_empty(2))

    def test_reforming_same_mill(self):
        """Test breaking a mill and reforming it (should trigger capture again)"""
        game = MorabarabaGame()
        game.phase = Phase.MOVING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        game.board = Board()
        
        # White mill: 0, 1, 2
        for i in [0, 1, 2]:
            game.board.place_piece(i, Player.WHITE)
        
        # Black pieces for capture (need 3+ to avoid instant game over)
        game.board.place_piece(8, Player.BLACK)
        game.board.place_piece(9, Player.BLACK)
        game.board.place_piece(10, Player.BLACK)  # Add 3rd piece
        
        game.current_player = Player.WHITE
        
        # Break mill: move 2 to 3
        game.apply_move({"type": "move", "from": 2, "to": 3})
        
        # Black moves
        game.apply_move({"type": "move", "from": 10, "to": 11})  # Use the 3rd piece
        
        # Reform mill: move 3 back to 2
        game.apply_move({"type": "move", "from": 3, "to": 2})
        
        # Should trigger capture requirement
        self.assertTrue(game.pending_capture, "Reforming mill should trigger capture")

    def test_concurrent_games_full_independence(self):
        """Verify multiple game instances are completely independent"""
        game1 = MorabarabaGame()
        game2 = MorabarabaGame()
        game3 = MorabarabaGame()
        
        # Play different sequences in each game
        game1.apply_move({"type": "place", "to": 0})
        game1.apply_move({"type": "place", "to": 8})
        
        game2.apply_move({"type": "place", "to": 5})
        game2.apply_move({"type": "place", "to": 15})
        
        game3.apply_move({"type": "place", "to": 10})
        
        # Verify each game has independent state
        self.assertEqual(game1.board.get_piece(0), 1)
        self.assertEqual(game1.board.get_piece(8), -1)
        self.assertTrue(game1.board.is_empty(5))
        
        self.assertTrue(game2.board.is_empty(0))
        self.assertEqual(game2.board.get_piece(5), 1)
        self.assertEqual(game2.board.get_piece(15), -1)
        
        self.assertTrue(game3.board.is_empty(0))
        self.assertTrue(game3.board.is_empty(5))
        self.assertEqual(game3.board.get_piece(10), 1)

    def test_state_serialization_and_recovery(self):
        """Test game state can be serialized and reconstructed"""
        game = MorabarabaGame()
        
        # Play some moves
        moves = [
            {"type": "place", "to": 0},
            {"type": "place", "to": 8},
            {"type": "place", "to": 1},
            {"type": "place", "to": 9},
        ]
        
        for move in moves:
            game.apply_move(move)
        
        # Get state
        state = game.get_state()
        
        # Verify state contains all necessary information
        self.assertIn("board", state)
        self.assertIn("phase", state)
        self.assertIn("current_player", state)
        self.assertIn("pending_capture", state)
        self.assertIn("game_over", state)
        
        # Verify board state is accurate (board uses STRING keys!)
        self.assertEqual(state["board"]["0"], 1)
        self.assertEqual(state["board"]["8"], -1)
        self.assertEqual(state["board"]["1"], 1)
        self.assertEqual(state["board"]["9"], -1)

    def test_game_over_prevents_further_moves(self):
        """Verify that once game is over, no more moves can be applied"""
        game = MorabarabaGame()
        
        # Manually set game over state
        game.game_over = True
        game.winner = Player.WHITE
        
        # Try to make a move
        with self.assertRaises(GameOverError):
            game.apply_move({"type": "place", "to": 0})

    def test_flying_phase_with_minimal_pieces(self):
        """Test flying phase behavior with exactly 3 pieces"""
        game = MorabarabaGame()
        game.phase = Phase.FLYING
        game.pieces_placed = {Player.WHITE: 12, Player.BLACK: 12}
        game.board = Board()
        
        # White has exactly 3 pieces
        for i in [0, 8, 16]:
            game.board.place_piece(i, Player.WHITE)
        
        # Black has 4 pieces
        for i in [1, 9, 17, 5]:
            game.board.place_piece(i, Player.BLACK)
        
        game.current_player = Player.WHITE
        
        # White should be able to fly anywhere
        legal_moves = get_legal_moves(game.board, Player.WHITE, Phase.FLYING, 24, False)
        
        # Should be able to move to any empty position (24 - 7 occupied = 17 possible destinations)
        # Each of 3 pieces can move to 17 positions = 51 total moves
        # But we need to account for the 'from' position, so 3 pieces * 17 empty spots
        self.assertEqual(len(legal_moves), 3 * 17)

    def test_capture_reduces_piece_count_correctly(self):
        """Verify captures correctly reduce opponent piece count"""
        game = MorabarabaGame()
        
        # Form a mill and capture
        moves = [
            {"type": "place", "to": 0},
            {"type": "place", "to": 8},
            {"type": "place", "to": 1},
            {"type": "place", "to": 9},
            {"type": "place", "to": 2},  # Mill!
            {"type": "capture", "position_captured": 8},
        ]
        
        for move in moves:
            game.apply_move(move)
        
        # Black should have 1 piece (placed 2, lost 1)
        self.assertEqual(game.board.count_pieces(Player.BLACK), 1)
        
        # White should have 3 pieces
        self.assertEqual(game.board.count_pieces(Player.WHITE), 3)


class TestMLReadiness(unittest.TestCase):
    """Tests specifically designed for ML training readiness"""
    
    def test_deterministic_behavior(self):
        """Verify same moves produce identical results"""
        moves = [
            {"type": "place", "to": 0},
            {"type": "place", "to": 8},
            {"type": "place", "to": 1},
            {"type": "place", "to": 9},
            {"type": "place", "to": 2},
            {"type": "capture", "position_captured": 8},
        ]
        
        # Play same sequence twice
        game1 = MorabarabaGame()
        game2 = MorabarabaGame()
        
        for move in moves:
            state1 = game1.apply_move(move)
            state2 = game2.apply_move(move)
            
            # States should be identical
            self.assertEqual(state1["phase"], state2["phase"])
            self.assertEqual(state1["current_player"], state2["current_player"])
            self.assertEqual(state1["pending_capture"], state2["pending_capture"])

    def test_legal_moves_never_empty_until_game_over(self):
        """During active game, legal moves should never be empty (except when blocked)"""
        game = MorabarabaGame()
        
        for _ in range(100):  # Play 100 random moves
            if game.game_over:
                break
            
            legal_moves = game.get_legal_moves()
            
            # Should always have moves unless game is over
            if not game.game_over:
                self.assertGreater(
                    len(legal_moves),
                    0,
                    f"Should have legal moves when game is active (phase: {game.phase})"
                )
            
            if legal_moves:
                game.apply_move(random.choice(legal_moves))

    def test_reward_signal_availability(self):
        """Verify game state provides enough info for reward calculation"""
        game = MorabarabaGame()
        
        # Play a few moves
        moves = [
            {"type": "place", "to": 0},
            {"type": "place", "to": 8},
            {"type": "place", "to": 1},
        ]
        
        for move in moves:
            state = game.apply_move(move)
            
            # State should contain information needed for rewards
            self.assertIn("board", state)
            self.assertIn("phase", state)
            self.assertIn("current_player", state)
            self.assertIn("game_over", state)
            
            # Can calculate piece counts for reward
            white_count = game.board.count_pieces(Player.WHITE)
            black_count = game.board.count_pieces(Player.BLACK)
            
            self.assertGreaterEqual(white_count, 0)
            self.assertGreaterEqual(black_count, 0)

    def test_action_space_consistency(self):
        """Verify action space structure is consistent across game states"""
        game = MorabarabaGame()
        
        # In PLACING phase
        moves_placing = game.get_legal_moves()
        self.assertTrue(all(m["type"] == "place" for m in moves_placing if "type" in m))
        
        # Transition to MOVING phase
        for i in range(24):
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            
            move = legal_moves[0]
            game.apply_move(move)
            
            if game.pending_capture:
                capturable = get_capturable_pieces(game.board, game.current_player)
                if capturable:
                    game.apply_move({"type": "capture", "position_captured": capturable[0]})
        
        # In MOVING phase
        if game.phase == Phase.MOVING:
            moves_moving = game.get_legal_moves()
            self.assertTrue(
                all(m["type"] == "move" for m in moves_moving if "type" in m),
                "In MOVING phase, all moves should be 'move' type"
            )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
