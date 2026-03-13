from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import enum

SQLALCHEMY_DATABASE_URL = "sqlite:///./storage/morabaraba.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class GameStatus(str, enum.Enum):
    ONGOING = "ONGOING"
    FINISHED = "FINISHED"

class GameModel(Base):
    __tablename__ = "games"

    game_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    current_player = Column(String)
    phase = Column(String)
    status = Column(SQLEnum(GameStatus), default=GameStatus.ONGOING)
    winner = Column(String, nullable=True)
    is_ai_opponent = Column(Integer, default=0) # 0=False, 1=True
    ai_player = Column(String, nullable=True) # WHITE or BLACK

class MoveModel(Base):
    __tablename__ = "moves"

    move_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    game_id = Column(String, ForeignKey("games.game_id"))
    player = Column(String)
    from_pos = Column(Integer, nullable=True)
    to_pos = Column(Integer, nullable=True)
    position_captured = Column(Integer, nullable=True)
    move_type = Column(String)
    phase = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    reward_metrics = Column(JSON)
    board_state = Column(JSON)  # 24-position integer encoding

def init_db():
    Base.metadata.create_all(bind=engine)
