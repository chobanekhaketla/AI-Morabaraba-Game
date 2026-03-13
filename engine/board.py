from typing import List, Optional, Union
from .constants import Player, MILL_LINES, ADJACENCY
from .errors import IllegalMoveError, InvalidPositionError

class Board:
    """
    Refined Morabaraba board.
    Dumb state container with structure and basic piece manipulation.
    Indices: 0-23
    Values: 0 (empty), 1 (White), -1 (Black)
    """
    def __init__(self, state: Optional[List[int]] = None):
        if state:
            if len(state) != 24:
                raise ValueError("Board state must be exactly 24 elements.")
            self.state = list(state)
        else:
            self.state = [0] * 24

    def get_piece(self, pos: int) -> int:
        self._validate_pos(pos)
        return self.state[pos]

    def place_piece(self, pos: int, player: Union[int, Player]):
        self._validate_pos(pos)
        if self.state[pos] != 0:
            raise IllegalMoveError(f"Position {pos} is already occupied.")
        val = player.value if isinstance(player, Player) else player
        self.state[pos] = val

    def move_piece(self, from_pos: int, to_pos: int, player: Union[int, Player]):
        self._validate_pos(from_pos)
        self._validate_pos(to_pos)
        if self.state[from_pos] == 0:
            raise IllegalMoveError(f"No piece to move at position {from_pos}.")
        if self.state[to_pos] != 0:
            raise IllegalMoveError(f"Destination position {to_pos} is occupied.")
        val = player.value if isinstance(player, Player) else player
        self.state[from_pos] = 0
        self.state[to_pos] = val

    def remove_piece(self, pos: int):
        self._validate_pos(pos)
        if self.state[pos] == 0:
            raise IllegalMoveError(f"No piece to remove at position {pos}.")
        self.state[pos] = 0

    def is_empty(self, pos: int) -> bool:
        return self.get_piece(pos) == 0

    def is_mill(self, pos: int, player: Union[int, Player]) -> bool:
        """Checks if piece at pos forms a mill for player."""
        val = player.value if isinstance(player, Player) else player
        if self.get_piece(pos) != val:
            return False
            
        for line in MILL_LINES:
            if pos in line:
                if all(self.state[i] == val for i in line):
                    return True
        return False

    def get_player_pieces(self, player: Union[int, Player]) -> List[int]:
        val = player.value if isinstance(player, Player) else player
        return [i for i, v in enumerate(self.state) if v == val]

    def count_pieces(self, player: Union[int, Player]) -> int:
        return len(self.get_player_pieces(player))

    def get_legal_moves(self, player: Union[int, Player], phase_flying: bool = False) -> List[int]:
        """Returns all legal destination positions for current player."""
        # This is a simplified version; full logic belongs in rules.py
        # But if we need it here as per requirement:
        return [i for i, v in enumerate(self.state) if v == 0]

    def print_board(self):
        """Prints a text-based representation of the board."""
        def p(i):
            v = self.state[i]
            if v == 1: return "W"
            if v == -1: return "B"
            return str(i).zfill(2)

        print(f"{p(0)}---------{p(1)}---------{p(2)}")
        print(f"|         |         |")
        print(f"|  {p(8)}------{p(9)}------{p(10)}  |")
        print(f"|  |      |      |  |")
        print(f"|  |  {p(16)}---{p(17)}---{p(18)}  |  |")
        print(f"|  |  |       |  |  |")
        print(f"{p(7)}--{p(15)}--{p(23)}      {p(19)}--{p(11)}--{p(3)}")
        print(f"|  |  |       |  |  |")
        print(f"|  |  {p(22)}---{p(21)}---{p(20)}  |  |")
        print(f"|  |      |      |  |")
        print(f"|  {p(14)}------{p(13)}------{p(12)}  |")
        print(f"|         |         |")
        print(f"{p(6)}---------{p(5)}---------{p(4)}")

    def _validate_pos(self, pos: int):
        if not (0 <= pos <= 23):
            raise InvalidPositionError(f"Board index {pos} out of range (0-23).")

    def to_dict(self) -> dict:
        return {str(i): v for i, v in enumerate(self.state)}

    def copy(self) -> 'Board':
        return Board(state=self.state)
