import numpy as np
import random

class RandomLegalAgent:
    """
    An agent that selects actions uniformly at random from the pool of legal moves.
    Used for environment stress testing and filling the initial replay buffer.
    """
    def select_action(self, state_encoding: np.ndarray, legal_mask: np.ndarray) -> int:
        """
        Samples a legal action index based on the mask.
        
        Args:
            state_encoding: (unused) The encoded board state.
            legal_mask: Array of shape (1200,) where 1.0 indicates a legal action.
            
        Returns:
            The integer index of the selected action.
        """
        legal_indices = np.where(legal_mask == 1.0)[0]
        if len(legal_indices) == 0:
            # This should only happen if the game logic has a bug where 
            # no moves are legal but the game is not marked as over.
            raise ValueError("No legal actions available according to the mask.")
            
        return int(random.choice(legal_indices))
