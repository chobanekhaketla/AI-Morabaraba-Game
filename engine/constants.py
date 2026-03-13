from enum import Enum

class Phase(Enum):
    PLACING = "PLACING"
    MOVING = "MOVING"
    FLYING = "FLYING"

class Player(Enum):
    WHITE = 1
    BLACK = -1

    @property
    def opponent(self):
        return Player.BLACK if self == Player.WHITE else Player.WHITE

# Board layout: 24 positions
# Outer ring: 0-7, Middle ring: 8-15, Inner ring: 16-23

# Adjacency graph representing the Morabaraba board topology
ADJACENCY = {
    # Outer Ring
    0: [1, 7, 8],
    1: [0, 2, 9],
    2: [1, 3, 10],
    3: [2, 4, 11],
    4: [3, 5, 12],
    5: [4, 6, 13],
    6: [5, 7, 14],
    7: [6, 0, 15],
    
    # Middle Ring
    8: [0, 9, 15, 16],
    9: [1, 8, 10, 17],
    10: [2, 9, 11, 18],
    11: [3, 10, 12, 19],
    12: [4, 11, 13, 20],
    13: [5, 12, 14, 21],
    14: [6, 13, 15, 22],
    15: [7, 14, 8, 23],
    
    # Inner Ring
    16: [8, 17, 23],
    17: [9, 16, 18],
    18: [10, 17, 19],
    19: [11, 18, 20],
    20: [12, 19, 21],
    21: [13, 20, 22],
    22: [14, 21, 23],
    23: [15, 22, 16],
}

# All possible 3-in-a-row combinations (Mills)
MILL_LINES = [
    # Outer Ring Mills
    (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0),
    # Middle Ring Mills
    (8, 9, 10), (10, 11, 12), (12, 13, 14), (14, 15, 8),
    # Inner Ring Mills
    (16, 17, 18), (18, 19, 20), (20, 21, 22), (22, 23, 16),
    # Cross-Ring Midpoint Mills
    (1, 9, 17), (3, 11, 19), (5, 13, 21), (7, 15, 23),
]

PIECES_PER_PLAYER = 12
MIN_PIECES_FOR_MOVING = 3  # Threshold for flying phase (when exactly 3 pieces left)
