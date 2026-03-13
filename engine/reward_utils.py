from typing import List, Dict, Any, Optional
from .constants import Player, Phase, MILL_LINES
from .board import Board
from .rules import get_mills, get_legal_moves, forms_mill

def compute_reward_metrics(
    board_before: Board,
    board_after: Board,
    player: Player,
    phase: Phase,
    move: Dict[str, Any],
    pending_capture_before: bool
) -> Dict[str, Any]:
    """
    Computes deterministic metrics for a move.
    No ML, no rewards, just instrumentation.
    """
    val = player.value
    opponent = player.opponent
    opp_val = opponent.value
    move_type = move.get("type")

    # 1. Mill Metrics
    mills_before = get_mills(board_before, player)
    mills_after = get_mills(board_after, player)
    
    mills_formed = len(mills_after) - len(mills_before)
    
    opp_mills_before = get_mills(board_before, opponent)
    opp_mills_after = get_mills(board_after, opponent)
    opp_mills_broken = len(opp_mills_before) - len(opp_mills_after)

    # 2. Capture Context
    capture_performed = (move_type == "capture")
    capture_from_mill = False
    if capture_performed:
        pos = move.get("position_captured")
        if pos is not None:
             for mill in MILL_LINES:
                 if pos in mill:
                     if all(board_before.get_piece(p) == opp_val for p in mill):
                         capture_from_mill = True
                         break

    return {
        "mills_formed": max(0, mills_formed),
        "mills_broken": max(0, -mills_formed),
        "opponent_mills_broken": max(0, opp_mills_broken),
        "capture_performed": capture_performed,
        "capture_from_mill": capture_from_mill
    }

def encode_board(board: Board, current_player: Player) -> List[int]:
    """
    Encodes board to 24-position integer list.
    1  = current player
    -1 = opponent
    0  = empty
    """
    encoding = []
    val = current_player.value
    opp_val = current_player.opponent.value
    for i in range(24):
        piece = board.get_piece(i)
        if piece == val:
            encoding.append(1)
        elif piece == opp_val:
            encoding.append(-1)
        else:
            encoding.append(0)
    return encoding
