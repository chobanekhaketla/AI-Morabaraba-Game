from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from .schemas import MoveRequest, GameStateResponse, NewGameResponse, MoveHistoryItem, GameStats, NewGameRequest
from .services import game_service
from engine.errors import MorabarabaError

router = APIRouter(prefix="/games")

@router.post("/new", response_model=NewGameResponse)
async def create_new_game(request: Optional[NewGameRequest] = None):
    # Default to human vs human if no body provided
    is_ai = False
    ai_side = "BLACK"
    if request:
        is_ai = request.play_vs_ai
        ai_side = request.ai_player
        
    game_id = game_service.create_game(is_ai_opponent=is_ai, ai_player=ai_side)
    state = game_service.get_game_state(game_id)
    return {"game_id": game_id, "state": state}

@router.get("/{game_id}", response_model=GameStateResponse)
async def get_game_status(game_id: str):
    try:
        return game_service.get_game_state(game_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{game_id}/move", response_model=GameStateResponse)
async def make_move(game_id: str, move: MoveRequest):
    try:
        # Convert Pydantic model to dict, aliasing 'from_pos' back to 'from' if needed
        # But our engine expects 'from'
        move_dict = move.dict(by_alias=True, exclude_none=True)
        return game_service.apply_move(game_id, move_dict)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MorabarabaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/{game_id}/history", response_model=List[MoveHistoryItem])
async def get_game_history(game_id: str):
    try:
        return game_service.get_history(game_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/completed/all", response_model=List[Dict[str, Any]])
async def get_completed_games():
    return game_service.get_completed_games()

@router.get("/{game_id}/stats", response_model=GameStats)
async def get_game_stats(game_id: str):
    try:
        return game_service.get_game_stats(game_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
