from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import os
import torch
import numpy as np
from datetime import datetime

from app.schemas import AIMoveRequest, AIMoveResponse
from agent.dqn_agent import DQNAgent
from engine.game import MorabarabaGame
from engine.constants import Player, Phase, PIECES_PER_PLAYER
from engine.action_space import legal_action_mask, action_to_engine_move
from engine.env import MorabarabaEnv

router = APIRouter(prefix="/ai")

class BrainLoader:
    _instance = None
    _agent: Optional[DQNAgent] = None
    _brain_id: str = "v0.0.0"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BrainLoader()
        return cls._instance

    def load_default_brain(self):
        # Path to the final model from Phase 8
        model_path = "models/dqn_final.pth"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. AI will use random weights.")
            self._brain_id = "random-init"
        else:
            self._brain_id = "dqn-final"
            
        # Initialize agent
        # Note: Params must match training! (Ideally these are in metadata.json)
        self._agent = DQNAgent(eps=0.0) # Greedy inference
        
        if os.path.exists(model_path):
            try:
                self._agent.load(model_path)
                print(f"Successfully loaded brain: {self._brain_id}")
            except Exception as e:
                print(f"Failed to load model weights: {e}")
                self._brain_id = "error-fallback"

    def get_agent(self) -> DQNAgent:
        if self._agent is None:
            self.load_default_brain()
        return self._agent
    
    def get_brain_id(self) -> str:
        return self._brain_id

# Dependency
def get_brain_loader():
    return BrainLoader.get_instance()

@router.post("/move", response_model=AIMoveResponse)
async def get_ai_move(request: AIMoveRequest, loader: BrainLoader = Depends(get_brain_loader)):
    agent = loader.get_agent()
    
    # 1. Reconstruct Game State
    try:
        game = MorabarabaGame()
        
        # Hydrate Board
        # Request board keys are strings "0", "1"... values are 1, -1, 0
        import engine.board # Need to ensure Board is available 
        # Actually Board is imported from game usually, but here we access game.board
        # game.board is an instance of Board. 
        # To clear it provided we don't have clear(), just replace it.
        from engine.board import Board
        game.board = Board() # Reset empty
        for pos_str, val in request.board.items():
            if val != 0:
                game.board.place_piece(int(pos_str), Player(val))
                
        # Hydrate State
        game.current_player = Player(request.current_player) if isinstance(request.current_player, int) else \
                              (Player.WHITE if request.current_player == "WHITE" else Player.BLACK)
        
        # Phase handling (string to enum)
        phase_map = {"PLACING": Phase.PLACING, "MOVING": Phase.MOVING, "FLYING": Phase.FLYING}
        if request.phase in phase_map:
            # We must use the override because the engine derives phase from piece counts,
            # but the UI might report something slightly different or we want to trust the UI.
            # Actually, using the derived phase is safer, but let's trust the request for this stateless protocol.
            game.phase = phase_map[request.phase] # Uses the setter we added in Phase 7!
            
        game.pending_capture = request.pending_capture
        
        # Hydrate Pieces Placed (derived from request)
        # We need this for rules that check if placing phase is over
        # pieces_to_place = 12 - placed
        w_to_place = request.pieces_to_place.get("WHITE", 0)
        b_to_place = request.pieces_to_place.get("BLACK", 0)
        game.pieces_placed[Player.WHITE] = PIECES_PER_PLAYER - w_to_place
        game.pieces_placed[Player.BLACK] = PIECES_PER_PLAYER - b_to_place
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reconstruct game state: {str(e)}")

    # 2. Generate Encoding and Mask
    # We can use the Env's helper method if we attach the game to an env
    # Or just replicate the logic. Reusing Env is cleaner but Env __init__ makes a new game.
    # Let's just use a dummy env.
    env = MorabarabaEnv()
    env.game = game # Inject reconstructed game
    
    state_encoding = env.get_encoding()
    
    # Generate Mask
    # We need to construct the state_dict format expected by legal_action_mask
    state_dict = game.get_state() 
    # Ensure board uses string keys for the mask function (it might expect it)
    # legal_action_mask in action_space.py iterates: for pos_str, val in game_state["board"].items()
    # game.get_state() returns string keys in board dict. Perfect.
    
    mask = legal_action_mask(state_dict)
    
    # 3. Select Action
    try:
        action_idx = agent.select_action(state_encoding, mask)
        move_dict = action_to_engine_move(action_idx)
    except ValueError as e:
        # This usually happens if mask is all zeros (no legal moves)
        raise HTTPException(status_code=400, detail=f"AI cannot move: {str(e)}")
    
    # 4. Construct Response
    return AIMoveResponse(
        move=move_dict,
        confidence=0.0, # DQNAgent select_action doesn't return q-values currently, only index
        brain_id=loader.get_brain_id(),
        timestamp=datetime.utcnow().isoformat()
    )
