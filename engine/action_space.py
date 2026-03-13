from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from .constants import Phase, Player
from . import rules

# Define static phases for action representation
ACTION_PHASES = ["PLACING", "MOVING", "FLYING", "CAPTURE"]

# Action tuple format: (phase, from_pos, to_pos, capture_pos)
Action = Tuple[str, int, int, int]

def _generate_actions() -> List[Action]:
    actions = []
    
    # 1. PLACING Phase Actions (24)
    # (PLACING, -1, to, -1)
    for to_pos in range(24):
        actions.append(("PLACING", -1, to_pos, -1))
        
    # 2. MOVING Phase Actions (576)
    # (MOVING, from, to, -1)
    for from_pos in range(24):
        for to_pos in range(24):
            actions.append(("MOVING", from_pos, to_pos, -1))
            
    # 3. FLYING Phase Actions (576)
    # (FLYING, from, to, -1)
    for from_pos in range(24):
        for to_pos in range(24):
            actions.append(("FLYING", from_pos, to_pos, -1))
            
    # 4. CAPTURE Actions (24)
    # (CAPTURE, -1, -1, capture)
    for cap_pos in range(24):
        actions.append(("CAPTURE", -1, -1, cap_pos))
        
    return actions

ACTIONS: List[Action] = _generate_actions()
ACTION_TO_INDEX: Dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}
INDEX_TO_ACTION: Dict[int, Action] = {i: a for i, a in enumerate(ACTIONS)}

def legal_action_mask(game_state: Dict[str, Any]) -> np.ndarray:
    """
    Produces a binary mask (float32) of length 1200.
    1.0 for legal actions, 0.0 for illegal actions.
    """
    mask = np.zeros(len(ACTIONS), dtype=np.float32)
    
    # We need a Board object to use rules functions
    from .board import Board
    board = Board()
    # board_state is a dict with string keys from game_state["board"]
    for pos_str, val in game_state["board"].items():
        if val != 0:
            board.place_piece(int(pos_str), Player(val))
    
    current_player = Player(game_state["current_player"])
    phase = Phase(game_state["phase"])
    pending_capture = game_state["pending_capture"]
    
    # Get physical legal moves from engine
    # pieces_placed is needed for rules.get_legal_moves but it's used for phase transitions
    # Since phase is already in state, we can pass a dummy pieces_placed if it's over 24
    # Actually, rules.get_legal_moves uses pieces_placed to check if it's PLACING
    # We can infer it from the pieces_to_place in state
    pieces_to_place = game_state.get("pieces_to_place", {})
    w_left = pieces_to_place.get("WHITE", 0)
    b_left = pieces_to_place.get("BLACK", 0)
    # Total placed = 24 - (w_left + b_left)
    total_placed = 24 - (w_left + b_left)

    legal_moves = rules.get_legal_moves(board, current_player, phase, total_placed, pending_capture)
    
    for move in legal_moves:
        action_tuple = None
        move_type = move["type"]
        
        if move_type == "place":
            action_tuple = ("PLACING", -1, move["to"], -1)
        elif move_type == "move":
            # Determine if this moving action is physically represented in MOVING or FLYING block
            # In the unified action space, the agent should choose the action corresponding to the current phase
            if phase == Phase.MOVING:
                action_tuple = ("MOVING", move["from"], move["to"], -1)
            elif phase == Phase.FLYING:
                action_tuple = ("FLYING", move["from"], move["to"], -1)
        elif move_type == "capture":
            action_tuple = ("CAPTURE", -1, -1, move["position_captured"])
            
        if action_tuple and action_tuple in ACTION_TO_INDEX:
            index = ACTION_TO_INDEX[action_tuple]
            mask[index] = 1.0
            
    return mask

def action_to_engine_move(action_index: int) -> Dict[str, Any]:
    """Converts an action index back to an engine-compatible move dict."""
    action = INDEX_TO_ACTION[action_index]
    phase_str, from_pos, to_pos, cap_pos = action
    
    if phase_str == "PLACING":
        return {"type": "place", "to": to_pos}
    elif phase_str in ["MOVING", "FLYING"]:
        return {"type": "move", "from": from_pos, "to": to_pos}
    elif phase_str == "CAPTURE":
        return {"type": "capture", "position_captured": cap_pos}
    
    raise ValueError(f"Unknown action phase: {phase_str}")
