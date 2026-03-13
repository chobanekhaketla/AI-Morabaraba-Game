import pytest
import json
import os
from playwright.sync_api import Page, expect, sync_playwright

# ---------------------------------------------------------------------
# System Chrome Playwright Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,   # set False if you want to see the browser
            slow_mo=30
        )
        yield browser
        browser.close()


@pytest.fixture
def page(browser: Page):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()

# ---------------------------------------------------------------------
# Static HTML Path
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = f"file:///{os.path.join(BASE_DIR, 'static', 'index.html').replace(os.sep, '/')}"

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def mock_game_state():
    return {
        "board": {str(i): 0 for i in range(24)},
        "phase": "PLACING",
        "current_player": "WHITE",
        "pending_capture": False,
        "game_over": False,
        "winner": None,
        "pieces_to_place": {"WHITE": 12, "BLACK": 12},
        "pieces_captured": {"WHITE": 0, "BLACK": 0},
        "is_ai_opponent": True,
        "ai_player": "BLACK"
    }

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def post_data(route):
    return route.request.post_data_json()

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_initial_load(page: Page):
    page.goto(HTML_PATH)

    expect(page.locator("h1")).to_contain_text("Morabaraba")
    expect(page.locator("#btn-new-game")).to_be_visible()

    expect(page.locator("#p1-pieces")).to_contain_text("12")
    expect(page.locator("#p2-pieces")).to_contain_text("12")


def test_new_game_vs_ai(page: Page, mock_game_state):

    def handle_new_game(route):
        route.fulfill(
            json={
                "game_id": "test-game-123",
                "state": mock_game_state
            }
        )

    page.route("**/games/new", handle_new_game)

    page.goto(HTML_PATH)
    page.click("#btn-new-game")

    expect(page.locator("#toast")).to_contain_text("New Game vs AI Started")
    expect(page.locator("#status-display")).to_contain_text("Your Turn")
    expect(page.locator("#phase-indicator")).to_contain_text("PLACING")


def test_game_flow_with_ai_response(page: Page, mock_game_state):

    # -------------------------------------------------------------
    # Route handlers
    # -------------------------------------------------------------

    def handle_new_game(route):
        route.fulfill(
            json={
                "game_id": "game-1",
                "state": mock_game_state
            }
        )

    def handle_user_move(route):
        data = post_data(route)

        if data["to"] == 0:
            mock_game_state["board"]["0"] = 1  # White
            mock_game_state["pieces_to_place"]["WHITE"] = 11
            mock_game_state["current_player"] = "BLACK"

            route.fulfill(json=mock_game_state)

    def handle_ai_decision(route):
        route.fulfill(
            json={
                "move": {"type": "place", "to": 1},
                "timestamp": "2024-01-01T00:00:00"
            }
        )

    def handle_apply_ai_move(route):
        data = post_data(route)

        if data["to"] == 1:
            mock_game_state["board"]["1"] = -1  # Black
            mock_game_state["pieces_to_place"]["BLACK"] = 11
            mock_game_state["current_player"] = "WHITE"

            route.fulfill(json=mock_game_state)

    def handle_move(route):
        data = post_data(route)
        if data["to"] == 1:
            handle_apply_ai_move(route)
        else:
            handle_user_move(route)

    # -------------------------------------------------------------
    # Register routes
    # -------------------------------------------------------------

    page.route("**/games/new", handle_new_game)
    page.route("**/games/*/move", handle_move)
    page.route("**/ai/move", handle_ai_decision)

    # -------------------------------------------------------------
    # Test flow
    # -------------------------------------------------------------

    page.goto(HTML_PATH)
    page.click("#btn-new-game")

    # User places at position 0
    node_0 = page.locator(".node-hitbox[data-pos='0']")
    node_0.click()

    # Verify user piece
    expect(page.locator(".piece.white")).to_have_count(1)

    # Verify AI response
    expect(page.locator(".piece.black")).to_have_count(1)

    # Verify updated stats
    expect(page.locator("#p1-pieces")).to_contain_text("11")
    expect(page.locator("#p2-pieces")).to_contain_text("11")
    expect(page.locator("#status-display")).to_contain_text("Your Turn")
