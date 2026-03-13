from typing import List, Dict, Any
from .constants import Player, Phase, ADJACENCY, MILL_LINES, PIECES_PER_PLAYER
from .board import Board

def is_adjacent(pos1: int, pos2: int) -> bool:
    """Returns True if pos1 and pos2 are connected on the board."""
    return pos2 in ADJACENCY.get(pos1, [])

def forms_mill(board: Board, player: Player, position: int) -> bool:
    """Check if placing/moving to position completes a mill for the player."""
    val = player.value
    for mill in MILL_LINES:
        if position in mill:
            # Wait, board.get_piece(p) == player is only true if piece is already there.
            # If we are checking BEFORE applying, we need to check other 2 positions.
            other_pos = [p for p in mill if p != position]
            if all(board.get_piece(p) == val for p in other_pos):
                return True
    return False

def get_mills(board: Board, player: Player) -> List[tuple]:
    """Returns a list of all current mills for the player."""
    val = player.value
    current_mills = []
    for mill in MILL_LINES:
        if all(board.get_piece(p) == val for p in mill):
            current_mills.append(mill)
    return current_mills

def is_in_mill(board: Board, player: Player, position: int) -> bool:
    """Returns True if the piece at position is part of any mill."""
    val = player.value
    for mill in MILL_LINES:
        if position in mill:
            if all(board.get_piece(p) == val for p in mill):
                return True
    return False

def get_capturable_pieces(board: Board, player: Player) -> List[int]:
    """
    Returns list of opponent pieces that can be captured.
    Rule: Prefer pieces NOT in mills; if all opponent pieces are in mills, any can be captured.
    """
    opponent = player.opponent
    opponent_pieces = board.get_player_pieces(opponent)
    
    not_in_mill = [p for p in opponent_pieces if not is_in_mill(board, opponent, p)]
    
    if not_in_mill:
        return not_in_mill
    return opponent_pieces

def is_legal_placement(board: Board, position: int) -> bool:
    """Placement is legal if the position is empty."""
    return board.is_empty(position)

def is_legal_move(board: Board, player: Player, from_pos: int, to_pos: int, phase: Phase) -> bool:
    """
    Validation for moving/flying phase.
    MOVING phase: must be adjacent.
    FLYING phase: can move anywhere.
    """
    if not board.is_empty(to_pos):
        return False
    if board.get_piece(from_pos) != player.value:
        return False
        
    if phase == Phase.MOVING:
        return is_adjacent(from_pos, to_pos)
    elif phase == Phase.FLYING:
        return True
    return False

def get_legal_moves(board: Board, player: Player, phase: Phase, pieces_placed: int, pending_capture: bool) -> List[Dict[str, Any]]:
    """Returns all possible legal moves for the current state."""
    moves = []
    
    if pending_capture:
        for pos in get_capturable_pieces(board, player):
            moves.append({"type": "capture", "position_captured": pos})
        return moves

    if phase == Phase.PLACING:
        for pos in range(24):
            if is_legal_placement(board, pos):
                moves.append({"type": "place", "to": pos})
    
    else: # MOVING or FLYING
        player_pieces = board.get_player_pieces(player)
        for from_pos in player_pieces:
            for to_pos in range(24):
                if is_legal_move(board, player, from_pos, to_pos, phase):
                    moves.append({"type": "move", "from": from_pos, "to": to_pos})
                    
    return moves

def has_legal_moves(board: Board, player: Player, phase: Phase, pieces_placed: int, pending_capture: bool) -> bool:
    """Returns True if the player has at least one legal move."""
    return len(get_legal_moves(board, player, phase, pieces_placed, pending_capture)) > 0

def check_win_condition(board: Board, player: Player, phase: Phase, pieces_placed: int) -> bool:
    """
    Opponent loses if: has < 3 pieces on board (after placing phase) 
    OR has no legal moves.
    Returns True if 'player' has won.
    """
    opponent = player.opponent
    
    # After placing phase, check piece count
    if pieces_placed >= 2 * PIECES_PER_PLAYER:
        if board.count_pieces(opponent) < 3:
            return True
            
    # Check if opponent is blocked
    # This should be checked at the start of opponent's turn, 
    # but check_win_condition usually checks if the current player's MOVE just won them the game.
    # If the move made the opponent have no moves, player wins.
    if not has_legal_moves(board, opponent, phase, pieces_placed, False):
        return True
        
    return False
