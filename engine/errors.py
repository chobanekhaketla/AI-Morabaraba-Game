class MorabarabaError(Exception):
    """Base exception for Morabaraba game engine."""
    pass

class IllegalMoveError(MorabarabaError):
    """Raised when an illegal move is attempted."""
    pass

class GameOverError(MorabarabaError):
    """Raised when a move is attempted on a finished game."""
    pass

class InvalidPositionError(MorabarabaError):
    """Raised when an invalid board position is referenced."""
    pass
