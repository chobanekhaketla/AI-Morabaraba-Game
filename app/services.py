# SymPy mpmath fix - MUST be at the very top before any torch imports
import sys
try:
    import mpmath
    sys.modules['sympy.mpmath'] = mpmath
except ImportError:
    pass

import uuid
import json
from typing import Dict, Optional, List, Any
from engine.game import MorabarabaGame
from engine.errors import MorabarabaError
from engine.reward_utils import compute_reward_metrics, encode_board
from engine.constants import Player, Phase
from .database import SessionLocal, GameModel, MoveModel, GameStatus, init_db
from .ai_routes import get_brain_loader, BrainLoader
from .schemas import AIMoveRequest

class GameService:
    def __init__(self):
        # In-memory store for active games
        self.games: Dict[str, MorabarabaGame] = {}
        # Store last move metrics in memory for response
        self.last_metrics: Dict[str, Any] = {}
        init_db()

    def create_game(self, is_ai_opponent: bool = False, ai_player: str = "BLACK") -> str:
        game_id = str(uuid.uuid4())
        game = MorabarabaGame()
        self.games[game_id] = game
        
        # Persist new game
        db = SessionLocal()
        state = game.get_state()
        db_game = GameModel(
            game_id=game_id,
            current_player=state["current_player"],
            phase=state["phase"],
            status=GameStatus.ONGOING,
            is_ai_opponent=1 if is_ai_opponent else 0,
            ai_player=ai_player if is_ai_opponent else None
        )
        db.add(db_game)
        db.commit()
        db.close()
        
        return game_id

    def get_game(self, game_id: str) -> Optional[MorabarabaGame]:
        return self.games.get(game_id)

    def apply_move(self, game_id: str, move_dict: dict) -> dict:
        game = self.get_game(game_id)
        if not game:
            raise KeyError(f"Game session {game_id} not found.")
        
        # 1. State before
        board_before = game.board.copy()
        player_before = game.current_player
        phase_before = game.phase
        pending_capture_before = game.pending_capture

        # 2. Apply move
        state = game.apply_move(move_dict)

        # 3. State after
        board_after = game.board.copy()

        # 4. Compute metrics
        metrics = compute_reward_metrics(
            board_before,
            board_after,
            player_before,
            phase_before,
            move_dict,
            pending_capture_before
        )
        self.last_metrics[game_id] = metrics

        # 5. Persist move and update game
        db = SessionLocal()
        
        # Update Game record
        db_game = db.query(GameModel).filter(GameModel.game_id == game_id).first()
        if db_game:
            db_game.current_player = state["current_player"]
            db_game.phase = state["phase"]
            if state["game_over"]:
                db_game.status = GameStatus.FINISHED
                db_game.winner = state["winner"]
        
        # Create Move record
        db_move = MoveModel(
            game_id=game_id,
            player=player_before.value,
            from_pos=move_dict.get("from"),
            to_pos=move_dict.get("to"),
            position_captured=move_dict.get("position_captured"),
            move_type=move_dict.get("type"),
            phase=phase_before.value,
            reward_metrics=metrics,
            board_state=encode_board(board_after, player_before)
        )
        db.add(db_move)
        db.commit()
        db.close()

        # Add metrics to return state
        state["last_move_metrics"] = metrics

        # Frontend will call /ai/move separately (stateless AI approach)
        return state

    def _is_ai_turn(self, game_id: str, current_player) -> bool:
        db = SessionLocal()
        db_game = db.query(GameModel).filter(GameModel.game_id == game_id).first()
        is_ai = db_game.is_ai_opponent == 1
        ai_side = db_game.ai_player
        db.close()
        
        if not is_ai:
            return False

        # Normalize current_player to string
        if isinstance(current_player, Player):
            p_str = "WHITE" if current_player == Player.WHITE else "BLACK"
        elif isinstance(current_player, int):
            p_str = "WHITE" if current_player == 1 else "BLACK"
        else:
            p_str = str(current_player).upper()

        return p_str == ai_side

    def _trigger_ai_move(self, game_id: str) -> dict:
        """Internal helper to calculate and apply an AI move."""
        game = self.games[game_id]
        state = game.get_state()
        
        # 1. Prepare Request for Brain
        # Need to reconstruct board dict from Board object
        board_dict = game.board.to_dict() # { "0": 1, "1": 0 ... } but values are Ints
        
        # Pieces to place
        w_placed = game.pieces_placed[Player.WHITE]
        b_placed = game.pieces_placed[Player.BLACK]
        pieces_to_place = {
             "WHITE": 12 - w_placed,
             "BLACK": 12 - b_placed
        }

        # Current player
        curr_p = "WHITE" if game.current_player == Player.WHITE else "BLACK"
        
        # Phase - derived from game
        phase_str = game.phase.name # PLACING, MOVING, FLYING

        req = AIMoveRequest(
            board=board_dict,
            phase=phase_str,
            current_player=curr_p,
            pending_capture=game.pending_capture,
            pieces_to_place=pieces_to_place,
            brain_id="latest"
        )
        
        # 2. Get Move from Agent (BrainLoader)
        loader = get_brain_loader()
        agent = loader.get_agent()
        
        # We REUSE the logic from ai_routes logic but we are internal here.
        # Ideally we refactor logic out of ai_routes.
        # But we can just use the agent directly since we have the game object!
        # We don't need to reconstruct the game like ai_routes does.
        
        from engine.env import MorabarabaEnv
        from engine.action_space import legal_action_mask, action_to_engine_move
        
        # Temp Env wrapper
        env = MorabarabaEnv()
        env.game = game # Inject CURRENT game instance directly
        
        encoding = env.get_encoding()
        mask = legal_action_mask(game.get_state())
        
        action_idx = agent.select_action(encoding, mask)
        move_dict = action_to_engine_move(action_idx)
        
        # 3. Apply AI Move - but DON'T trigger further AI moves
        # We directly apply the move without going through apply_move's AI check
        game.apply_move(move_dict)
        state = game.get_state()
        
        # Save AI move to DB
        db = SessionLocal()
        db_move = MoveModel(
            game_id=game_id,
            player="BLACK",  # AI is always black in our setup
            from_pos=move_dict.get("from"),
            to_pos=move_dict.get("to"),
            position_captured=move_dict.get("position_captured"),
            move_type=move_dict.get("type"),
            phase=game.phase.name,
            reward_metrics={},
            board_state=game.board.to_dict()
        )
        db.add(db_move)
        db.commit()
        db.close()
        
        return state

    def get_game_state(self, game_id: str) -> dict:
        game = self.get_game(game_id)
        if not game:
            raise KeyError(f"Game session {game_id} not found.")
        if not game:
            raise KeyError(f"Game session {game_id} not found.")
        
        state = game.get_state()
        state["last_move_metrics"] = self.last_metrics.get(game_id)
        
        # Enrich with AI session info
        db = SessionLocal()
        db_game = db.query(GameModel).filter(GameModel.game_id == game_id).first()
        if db_game:
            state["is_ai_opponent"] = (db_game.is_ai_opponent == 1)
            state["ai_player"] = db_game.ai_player
        db.close()
        
        return state

    def get_history(self, game_id: str) -> List[dict]:
        db = SessionLocal()
        moves = db.query(MoveModel).filter(MoveModel.game_id == game_id).order_by(MoveModel.timestamp).all()
        db.close()
        
        history = []
        for m in moves:
            history.append({
                "player": m.player,
                "move_type": m.move_type,
                "from_pos": m.from_pos,
                "to_pos": m.to_pos,
                "position_captured": m.position_captured,
                "phase": m.phase,
                "timestamp": m.timestamp.isoformat(),
                "reward_metrics": m.reward_metrics,
                "board_state": m.board_state
            })
        return history

    def get_completed_games(self) -> List[dict]:
        db = SessionLocal()
        games = db.query(GameModel).filter(GameModel.status == GameStatus.FINISHED).all()
        db.close()
        return [{"game_id": g.game_id, "winner": g.winner, "created_at": g.created_at.isoformat()} for g in games]

    def get_game_stats(self, game_id: str) -> dict:
        db = SessionLocal()
        game = db.query(GameModel).filter(GameModel.game_id == game_id).first()
        if not game:
            raise KeyError(f"Game {game_id} not found.")
        
        moves = db.query(MoveModel).filter(MoveModel.game_id == game_id).all()
        db.close()

        total_moves = len(moves)
        mills = {"WHITE": 0, "BLACK": 0}
        captures = {"WHITE": 0, "BLACK": 0}

        player_map = {"1": "WHITE", "-1": "BLACK"}

        for m in moves:
            metrics = m.reward_metrics
            p_name = player_map.get(str(m.player), str(m.player))
            if metrics.get("mills_formed", 0) > 0:
                mills[p_name] += metrics["mills_formed"]
            if metrics.get("capture_performed"):
                captures[p_name] += 1

        return {
            "game_id": game_id,
            "status": game.status.value,
            "winner": game.winner,
            "total_moves": total_moves,
            "mills_per_player": mills,
            "captures_per_player": captures
        }

# Global game service instance
game_service = GameService()
