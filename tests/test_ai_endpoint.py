from fastapi.testclient import TestClient
import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

class TestAIEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_ai_move_endpoint(self):
        """Test that /ai/move returns a valid move."""
        # Construct a simple initial board state (empty)
        # Note: API expects string keys for board positions
        board = {str(i): 0 for i in range(24)}
        
        # Place a few pieces to make it realistic
        board["0"] = 1  # Player 1 (White)
        board["1"] = -1 # Player 2 (Black)
        
        payload = {
            "board": board,
            "phase": "PLACING",
            "current_player": "WHITE", # or 1
            "pending_capture": False,
            "pieces_to_place": {"WHITE": 11, "BLACK": 11},
            "brain_id": "latest",
            "return_analysis": False
        }
        
        response = self.client.post("/ai/move", json=payload)
        
        # Check status code
        self.assertEqual(response.status_code, 200, f"Request failed: {response.text}")
        
        data = response.json()
        
        # Check structure
        self.assertIn("move", data)
        self.assertIn("brain_id", data)
        self.assertIn("timestamp", data)
        
        move = data["move"]
        self.assertIn("type", move)
        self.assertEqual(move["type"], "place")
        self.assertIn("to", move)
        # Ensure 'to' is a valid integer position
        self.assertTrue(0 <= move["to"] < 24)
        
        print(f"\nAI Response: {data}")

if __name__ == "__main__":
    unittest.main()
