
const API_BASE = "";
let currentGameId = null;
let gameState = null;
let selectedPos = null;

// Board Layout (x, y coordinates for 0-23)
// 0(50,50) ---- 1(400,50) ---- 2(750,50)
const NODES = {
    0: { x: 50, y: 50 }, 1: { x: 400, y: 50 }, 2: { x: 750, y: 50 },
    3: { x: 750, y: 400 }, 4: { x: 750, y: 750 }, 5: { x: 400, y: 750 },
    6: { x: 50, y: 750 }, 7: { x: 50, y: 400 },
    8: { x: 150, y: 150 }, 9: { x: 400, y: 150 }, 10: { x: 650, y: 150 },
    11: { x: 650, y: 400 }, 12: { x: 650, y: 650 }, 13: { x: 400, y: 650 },
    14: { x: 150, y: 650 }, 15: { x: 150, y: 400 },
    16: { x: 250, y: 250 }, 17: { x: 400, y: 250 }, 18: { x: 550, y: 250 },
    19: { x: 550, y: 400 }, 20: { x: 550, y: 550 }, 21: { x: 400, y: 550 },
    22: { x: 250, y: 550 }, 23: { x: 250, y: 400 }
};

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    initBoard();
    document.getElementById("btn-new-game").addEventListener("click", startNewGame);
});

function initBoard() {
    const nodesLayer = document.getElementById("nodes-layer");
    const piecesLayer = document.getElementById("pieces-layer");

    // Draw click targets for all nodes
    for (let i = 0; i < 24; i++) {
        const coords = NODES[i];

        // Node visual
        const visual = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        visual.setAttribute("cx", coords.x);
        visual.setAttribute("cy", coords.y);
        visual.setAttribute("r", 8);
        visual.classList.add("node-visual");
        nodesLayer.appendChild(visual);

        // Larger Hitbox
        const hitbox = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        hitbox.setAttribute("cx", coords.x);
        hitbox.setAttribute("cy", coords.y);
        hitbox.setAttribute("r", 25);
        hitbox.classList.add("node-hitbox");
        hitbox.dataset.pos = i;
        hitbox.addEventListener("click", () => handleNodeClick(i));
        nodesLayer.appendChild(hitbox);
    }
}

async function startNewGame() {
    try {
        const res = await fetch(`${API_BASE}/games/new`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ play_vs_ai: true, ai_player: "BLACK" })
        });
        const data = await res.json();
        currentGameId = data.game_id;
        gameState = data.state;

        renderState();
        showToast("New Game vs AI Started!");
        addToLog("Game started", "System");
    } catch (e) {
        showToast("Error starting game", true);
        console.error(e);
    }
}

async function handleNodeClick(pos) {
    if (!gameState || gameState.game_over) return;

    // If it's not user's turn (assuming User is WHITE vs AI BLACK)
    // Actually API supports playing both sides if we want, but UI assumes P1 (White) is User.
    if (gameState.is_ai_opponent && gameState.current_player === gameState.ai_player) {
        showToast("Wait for AI...", true);
        return;
    }

    const phase = gameState.phase;
    let move = null;

    if (gameState.pending_capture) {
        // Capture logic
        move = { type: "capture", position_captured: pos };
    } else {
        if (phase === "PLACING") {
            move = { type: "place", to: pos };
        } else {
            // Moving/Flying
            if (selectedPos === null) {
                // Select source
                if (gameState.board[pos.toString()] === 1) { // User is 1 (White)
                    selectedPos = pos;
                    renderPieces(); // Update selection visual
                }
                return;
            } else {
                // Select dest
                if (selectedPos === pos) {
                    selectedPos = null; // Deselect
                } else {
                    move = { type: "move", from: selectedPos, to: pos };
                }
            }
        }
    }

    if (move) {
        await sendMove(move);
        selectedPos = null;
    }
}

async function sendMove(move) {
    try {
        const res = await fetch(`${API_BASE}/games/${currentGameId}/move`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(move)
        });

        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || "Invalid Move", true);
            return;
        }

        const newState = await res.json();

        // Log user's move
        logMove(move, "You");

        gameState = newState;
        renderState();

        if (gameState.game_over) {
            showToast(`Game Over! Winner: ${gameState.winner}`);
            return;
        }

        // Check if it's AI's turn (current_player is -1 for BLACK)
        if (gameState.current_player === -1 || gameState.current_player === "BLACK") {
            await callAI();
        }

    } catch (e) {
        showToast("Network Error", true);
        console.error(e);
    }
}

async function callAI() {
    try {
        showToast("AI is thinking...");

        // Build AI move request from current game state
        // IMPORTANT: We use the LATEST gameState which might have pending_capture=True
        const aiReq = {
            board: gameState.board,
            phase: gameState.phase,
            current_player: "BLACK",
            pending_capture: gameState.pending_capture,
            pieces_to_place: gameState.pieces_to_place,
            brain_id: "latest"
        };

        const aiRes = await fetch(`${API_BASE}/ai/move`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(aiReq)
        });

        if (!aiRes.ok) {
            const err = await aiRes.json();
            showToast(err.detail || "AI Error", true);
            return;
        }

        const aiMoveData = await aiRes.json();
        const aiMove = aiMoveData.move;

        // Apply AI's move to the game
        const applyRes = await fetch(`${API_BASE}/games/${currentGameId}/move`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(aiMove)
        });

        if (!applyRes.ok) {
            const err = await applyRes.json();
            showToast(err.detail || "AI Move Failed", true);
            return;
        }

        const updatedState = await applyRes.json();

        // Log AI's move
        logMove(aiMove, "AI");

        gameState = updatedState;
        renderState();

        if (gameState.game_over) {
            showToast(`Game Over! Winner: ${gameState.winner}`);
            return;
        }

        // If AI formed a mill, it needs to capture immediately.
        // The turn is STILL "BLACK" (or -1) and pending_capture is True.
        if ((gameState.current_player === -1 || gameState.current_player === "BLACK") && gameState.pending_capture) {
            showToast("AI is capturing...");
            // Recursive call to handle the capture move
            await new Promise(r => setTimeout(r, 500)); // Small delay for visual clarity
            await callAI();
        } else {
            showToast("Your Turn");
        }

    } catch (e) {
        showToast("AI Network Error", true);
        console.error(e);
    }
}

function renderState() {
    renderPieces();
    updateUI();
}

function renderPieces() {
    const layer = document.getElementById("pieces-layer");
    layer.innerHTML = ""; // Clear

    Object.entries(gameState.board).forEach(([posStr, val]) => {
        const pos = parseInt(posStr);
        if (val === 0) return;

        const coords = NODES[pos];
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", coords.x);
        circle.setAttribute("cy", coords.y);
        circle.setAttribute("r", 15);
        circle.classList.add("piece");
        circle.classList.add(val === 1 ? "white" : "black");

        if (selectedPos === pos) {
            circle.classList.add("selected");
        }

        layer.appendChild(circle);
    });
}

function updateUI() {
    // Stats
    document.getElementById("p1-pieces").innerText = gameState.pieces_to_place.WHITE > 0 ?
        `To Place: ${gameState.pieces_to_place.WHITE}` : `On Board: ${countPieces(1)}`;

    document.getElementById("p2-pieces").innerText = gameState.pieces_to_place.BLACK > 0 ?
        `To Place: ${gameState.pieces_to_place.BLACK}` : `On Board: ${countPieces(-1)}`;

    // Captured counts
    if (gameState.pieces_captured) {
        document.getElementById("p1-captured").innerText = gameState.pieces_captured.WHITE || 0;
        document.getElementById("p2-captured").innerText = gameState.pieces_captured.BLACK || 0;
    }

    document.getElementById("phase-indicator").innerText = gameState.phase;

    // Status Text
    const statusDiv = document.getElementById("status-display");
    if (gameState.game_over) {
        statusDiv.innerText = `Winner: ${gameState.winner}`;
        statusDiv.style.color = "#22c55e";
    } else {
        const isUserTurn = (gameState.current_player === "WHITE" || gameState.current_player === 1);
        statusDiv.innerText = isUserTurn ? (gameState.pending_capture ? "Select opponent piece to CAPTURE" : "Your Turn") : "AI Thinking...";
        statusDiv.style.color = isUserTurn ? "#fff" : "#94a3b8";
    }

    // Active Player Card
    document.getElementById("p1-card").classList.toggle("active", gameState.current_player === "WHITE" || gameState.current_player === 1);
    document.getElementById("p2-card").classList.toggle("active", gameState.current_player === "BLACK" || gameState.current_player === -1);
}

function countPieces(playerVal) {
    return Object.values(gameState.board).filter(v => v === playerVal).length;
}

function logMove(move, player) {
    const list = document.getElementById("move-list");
    const li = document.createElement("li");
    let text = "";
    if (move.type === "place") text = `Placed at ${move.to}`;
    if (move.type === "move") text = `Moved ${move.from} → ${move.to}`;
    if (move.type === "capture") text = `Captured ${move.position_captured}`;

    li.innerHTML = `<span class="ply-num">${player}:</span> ${text}`;
    list.prepend(li);
}

function addToLog(msg, sender) {
    const list = document.getElementById("move-list");
    const li = document.createElement("li");
    li.innerHTML = `<span class="ply-num">${sender}:</span> ${msg}`;
    list.prepend(li);
}

function showToast(msg, isError = false) {
    const t = document.getElementById("toast");
    t.innerText = msg;
    t.style.border = isError ? "1px solid #ef4444" : "1px solid #22c55e";
    t.classList.remove("hidden");
    setTimeout(() => t.classList.add("hidden"), 3000);
}
