from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List, Union

class MoveRequest(BaseModel):
    type: str = Field(..., description="Type of move: 'place', 'move', or 'capture'")
    to: Optional[int] = Field(None, description="Target position for 'place' and 'move' moves (0-23)")
    from_pos: Optional[int] = Field(None, alias="from", description="Source position for 'move' moves (0-23)")
    position_captured: Optional[int] = Field(None, description="Position for 'capture' moves (0-23)")

    class Config:
        allow_population_by_field_name = True
        validate_by_name = True

class RewardMetrics(BaseModel):
    mills_formed: int
    mills_broken: int
    opponent_mills_broken: int
    capture_performed: bool
    capture_from_mill: bool

class GameStateResponse(BaseModel):
    board: Dict[str, int]
    phase: str
    current_player: Union[str, int]
    pending_capture: bool
    game_over: bool
    winner: Optional[Union[str, int]]
    pieces_to_place: Dict[str, int]
    pieces_captured: Optional[Dict[str, int]] = None
    last_move_metrics: Optional[RewardMetrics] = None
    is_ai_opponent: bool = False
    ai_player: Optional[Union[str, int]] = None

class NewGameResponse(BaseModel):
    game_id: str
    state: GameStateResponse

class MoveHistoryItem(BaseModel):
    player: Union[str, int]
    move_type: str
    from_pos: Optional[int] = None
    to_pos: Optional[int] = None
    position_captured: Optional[int] = None
    phase: str
    timestamp: str
    reward_metrics: RewardMetrics
    board_state: List[int]

class GameStats(BaseModel):
    game_id: str
    status: str
    winner: Optional[Union[str, int]]
    total_moves: int
    mills_per_player: Dict[str, int]
    captures_per_player: Dict[str, int]
    is_ai_opponent: bool = False

class NewGameRequest(BaseModel):
    play_vs_ai: bool = False
    ai_player: str = "BLACK" # WHITE or BLACK

class AIMoveRequest(BaseModel):
    board: Dict[str, int]
    phase: str
    current_player: Union[str, int]
    pending_capture: bool
    pieces_to_place: Dict[str, int]
    brain_id: Optional[str] = None
    return_analysis: bool = False

class AIMoveResponse(BaseModel):
    move: Dict[str, Any]
    confidence: Optional[float] = None
    brain_id: Optional[str] = None
    timestamp: str
    analysis: Optional[Dict[str, Any]] = None
