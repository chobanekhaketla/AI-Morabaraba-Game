from typing import Tuple, Dict, Any, Optional
import numpy as np
from .game import MorabarabaGame
from .action_space import ACTION_TO_INDEX, INDEX_TO_ACTION, action_to_engine_move, legal_action_mask
from .constants import Player, Phase
from .reward_utils import compute_reward_metrics, encode_board

try:
    # This check is just to ensure the file exists and is importable
    import engine.reward_utils
except ImportError:
    raise ImportError(
        "reward_utils module required. Ensure reward_utils.py exists in engine/"
    )

class MorabarabaEnv:
    """
    Deterministic Environment Layer for Morabaraba DQN.
    Provides a standard step(action_index) interface.
    """
    def __init__(self, max_moves: int = 500):
        self.game = MorabarabaGame()
        self.done = False
        self.move_count = 0
        self.max_moves = max_moves

    def reset(self) -> Dict[str, Any]:
        """Resets the environment to the initial state."""
        self.game = MorabarabaGame()
        self.done = False
        self.move_count = 0
        return self.game.get_state()

    def step(self, action_index: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Advances the environment by one action.

        Reward Calculation Notes:
        - Rewards are calculated from the perspective of the player who JUST MOVED (player_before)
        - If a move causes self-loss (e.g., blocks self), they get -10
        - This is intentional to discourage self-blocking moves
        
        Args:
            action_index: The integer index of the action to take.
            
        Returns:
            next_state: The new state of the game.
            reward: Scalar reward for the action.
            done: Whether the episode has ended.
            info: Diagnostic information.
        """
        if self.done:
            return self.game.get_state(), 0.0, True, {"error": "Game already over"}

        # 1. Action Decoding
        try:
            action_tuple = INDEX_TO_ACTION[action_index]
        except KeyError:
            # Invalid index
            self.done = True
            return self.game.get_state(), -10.0, True, {"error": "Invalid action index", "illegal_action": True}

        # 2. Legality Masking
        state_before_move = self.game.get_state()
        mask = legal_action_mask(state_before_move)
        
        # Validate mask
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"legal_action_mask returned {type(mask)}, expected np.ndarray")
        if mask.shape[0] != 1200:
            raise ValueError(f"Mask has {mask.shape[0]} actions, expected 1200")
        
        if mask[action_index] == 0:
            # Illegal move
            self.done = True
            return state_before_move, -1.0, True, {"illegal_action": True, "action_index": action_index}

        # 3. State Before (for reward/metrics)
        board_before = self.game.board.copy()
        player_before = self.game.current_player
        phase_before = self.game.phase
        pending_capture_before = self.game.pending_capture

        # 4. Apply Move
        move_dict = action_to_engine_move(action_index)
        try:
            self.game.apply_move(move_dict)
        except Exception as e:
            # This should not be reachable if masking is perfect
            self.done = True
            return state_before_move, -1.0, True, {"error": str(e), "illegal_action": True}

        # 5. State After
        board_after = self.game.board.copy()
        new_state = self.game.get_state()
        self.done = new_state["game_over"]

        # Increment move count for legal move
        self.move_count += 1

        # Check for move limit termination (draw)
        move_limit_reached = False
        if not self.done and self.move_count >= self.max_moves:
            self.done = True
            move_limit_reached = True

        # 6. Reward Calculation
        metrics = compute_reward_metrics(
            board_before,
            board_after,
            player_before,
            phase_before,
            move_dict,
            pending_capture_before
        )
        
        reward = 0.0
        termination_reason = None
        is_draw = False

        if move_limit_reached:
            is_draw = True
            termination_reason = "move_limit"
            reward = 0.0
        else:
            # Positive rewards for progress
            if metrics.get("mills_formed", 0) > 0:
                reward += 1.0
            if metrics.get("capture_performed"):
                reward += 1.0
                
            # Win/Loss rewards
            if self.done:
                if self.game.winner == player_before:
                    reward += 10.0
                    termination_reason = "win_loss"
                elif self.game.winner is not None:
                    # Opponent won, current player lost
                    reward -= 10.0
                    termination_reason = "win_loss"
                else:
                    # Draw/Stalemate (rare)
                    is_draw = True
                    termination_reason = "stalemate"

        # 7. Info Dictionary
        info = {
            "mills_formed": metrics.get("mills_formed", 0),
            "mills_broken": metrics.get("mills_broken", 0),
            "piece_captured": metrics.get("capture_performed", False),
            "illegal_action": False,
            "current_phase": self.game.phase.value,
            "game_over": self.done,
            "winner": self.game.winner.value if self.game.winner else None,
            "draw": is_draw,
            "termination_reason": termination_reason
        }

        return new_state, reward, self.done, info

    def get_encoding(self, player: Optional[Player] = None) -> np.ndarray:
        """Returns the board encoded as a 24-position numpy array."""
        p = player or self.game.current_player
        encoding = np.array(encode_board(self.game.board, p), dtype=np.float32)
        
        # Validate encoding
        assert encoding.shape == (24,), f"Encoding shape mismatch: {encoding.shape}"
        assert encoding.dtype == np.float32, f"Encoding dtype mismatch: {encoding.dtype}"
        
        return encoding
