from engine.board import Board
from engine.constants import Player

def show():
    board = Board()
    print("--- EMPTY BOARD ---")
    board.print_board()
    
    print("\n--- BOARD WITH SOME PIECES ---")
    # Cross-ring midpoint mill: 1, 9, 17
    board.place_piece(1, Player.WHITE)
    board.place_piece(9, Player.WHITE)
    board.place_piece(17, Player.WHITE)
    
    # Diagonal movement (non-mill): 0, 8, 16
    board.place_piece(0, Player.BLACK)
    board.place_piece(8, Player.BLACK)
    board.place_piece(16, Player.BLACK)
    
    board.print_board()
    
    print("\nMill check for White at 17 (1, 9, 17):", board.is_mill(17, Player.WHITE))
    print("Mill check for Black at 16 (0, 8, 16) [Should be False]:", board.is_mill(16, Player.BLACK))

if __name__ == "__main__":
    show()
