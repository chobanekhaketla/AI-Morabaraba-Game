from typing import Optional, Dict, List, Any
from .constants import Player, Phase, PIECES_PER_PLAYER
from .board import Board
from .errors import IllegalMoveError, GameOverError, MorabarabaError
from .import rules

class MorabarabaGame:
    """
    Orchestrates Morabaraba game flow, enforces rules, and manages state.
    """
    def __init__(self):
        self.board = Board()
        self.current_player = Player.WHITE
        # Stored phase is replaced by a derived property with a legacy backdoor for tests.
        self._phase_override: Optional[Phase] = None
        self.pieces_placed = {Player.WHITE: 0, Player.BLACK: 0}
        self.pending_capture = False
        self.winner: Optional[Player] = None
        self.game_over = False
        self.termination_reason: Optional[str] = None

    @property
    def phase(self) -> Phase:
        """Derived on-demand, player-specific phase. Supports manual override for tests."""
        if self._phase_override is not None:
            return self._phase_override
        return self._get_player_phase(self.current_player)

    @phase.setter
    def phase(self, value: Phase):
        """Legacy setter to support existing tests that fast-forward phase."""
        self._phase_override = value

    def _get_player_phase(self, player: Player) -> Phase:
        """Error 1 Fix: Pure derivation function for a player's phase."""
        total_placed = sum(self.pieces_placed.values())
        if total_placed < 2 * PIECES_PER_PLAYER:
            return Phase.PLACING
        
        piece_count = self.board.count_pieces(player)
        if piece_count == 3:
            return Phase.FLYING
        return Phase.MOVING

    def apply_move(self, move: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and applies a move.
        Move formats:
        - Placement: {"type": "place", "to": 5}
        - Movement: {"type": "move", "from": 3, "to": 4}
        - Capture: {"type": "capture", "position_captured": 7}
        """
        if self.game_over:
            raise GameOverError("The game is already over.")

        move_type = move.get("type")
        
        # 1. Capture handling
        if self.pending_capture:
            if move_type != "capture":
                raise IllegalMoveError("Must perform a capture move.")
            self._handle_capture(move.get("position_captured"))
            self._check_game_over()
            if not self.game_over:
                self._switch_turn()
            return self.get_state()

        # 2. General move handling
        if move_type == "place":
            if self.phase != Phase.PLACING:
                raise IllegalMoveError("Cannot place pieces outside of PLACING phase.")
            self._handle_placement(move.get("to"))
        elif move_type == "move":
            if self.phase == Phase.PLACING:
                raise IllegalMoveError("Cannot move pieces during PLACING phase.")
            self._handle_movement(move.get("from"), move.get("to"))
        else:
            raise IllegalMoveError(f"Invalid move type: {move_type}")

        # 3. Post-move logic (Mill check, Phase transition, Turn switching)
        if not self.pending_capture:
            self._check_game_over()
            if not self.game_over:
                self._switch_turn()

        return self.get_state()

    def _handle_placement(self, to_pos: int):
        if to_pos is None:
            raise IllegalMoveError("Placement move must specify 'to' position.")
        
        if not rules.is_legal_placement(self.board, to_pos):
            raise IllegalMoveError(f"Position {to_pos} is occupied.")

        self.board.place_piece(to_pos, self.current_player)
        self.pieces_placed[self.current_player] += 1
        
        if rules.forms_mill(self.board, self.current_player, to_pos):
            self.pending_capture = True

    def _handle_movement(self, from_pos: int, to_pos: int):
        if from_pos is None or to_pos is None:
            raise IllegalMoveError("Movement move must specify 'from' and 'to' positions.")

        if not rules.is_legal_move(self.board, self.current_player, from_pos, to_pos, self.phase):
            raise IllegalMoveError(f"Illegal move from {from_pos} to {to_pos}.")

        self.board.move_piece(from_pos, to_pos, self.current_player)
        
        if rules.forms_mill(self.board, self.current_player, to_pos):
            self.pending_capture = True

    def _handle_capture(self, position_captured: int):
        if position_captured is None:
            raise IllegalMoveError("Capture move must specify 'position_captured'.")

        capturable = rules.get_capturable_pieces(self.board, self.current_player)
        if position_captured not in capturable:
            raise IllegalMoveError(f"Piece at {position_captured} cannot be captured.")

        self.board.remove_piece(position_captured)
        self.pending_capture = False

    def _switch_turn(self):
        """Error 2 Fix: Immediate blocked-player detection after turn switch."""
        self.current_player = self.current_player.opponent
        # Check if new player is blocked immediately
        if not self.get_legal_moves():
            self.game_over = True
            self.winner = self.current_player.opponent
            self.termination_reason = "blocked_player"

    def _update_phase(self):
        """Deprecated: Clears manual phase override to restore dynamic derivation."""
        self._phase_override = None

    def _check_game_over(self):
        """Standard win detection."""
        total_placed = sum(self.pieces_placed.values())
        opponent_phase = self._get_player_phase(self.current_player.opponent)
        if rules.check_win_condition(self.board, self.current_player, opponent_phase, total_placed):
            self.game_over = True
            self.winner = self.current_player
            self.termination_reason = "win_loss"

    def get_state(self) -> Dict[str, Any]:
        # Count pieces captured = placed - on_board
        white_on_board = sum(1 for v in self.board.state if v == Player.WHITE.value)
        black_on_board = sum(1 for v in self.board.state if v == Player.BLACK.value)
        
        return {
            "board": self.board.to_dict(),
            "phase": self.phase.value,
            "current_player": self.current_player.value,
            "pending_capture": self.pending_capture,
            "game_over": self.game_over,
            "winner": self.winner.value if self.winner else None,
            "pieces_to_place": {
                "WHITE": PIECES_PER_PLAYER - self.pieces_placed[Player.WHITE],
                "BLACK": PIECES_PER_PLAYER - self.pieces_placed[Player.BLACK]
            },
            "pieces_captured": {
                "WHITE": self.pieces_placed[Player.WHITE] - white_on_board,
                "BLACK": self.pieces_placed[Player.BLACK] - black_on_board
            }
        }

    def get_legal_moves(self) -> List[Dict[str, Any]]:
        """Error 3 Fix: Safety vet to prevent empty action mask leakage."""
        total_placed = sum(self.pieces_placed.values())
        moves = rules.get_legal_moves(
            self.board, 
            self.current_player, 
            self.phase, 
            total_placed, 
            self.pending_capture
        )
        
        if not moves and not self.game_over:
            self.game_over = True
            self.winner = self.current_player.opponent
            self.termination_reason = "no_legal_moves"
            
        return moves
