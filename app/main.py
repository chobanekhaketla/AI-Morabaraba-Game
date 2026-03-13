import sys
try:
    import mpmath
    sys.modules['sympy.mpmath'] = mpmath
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import router as game_router
from app.ai_routes import router as ai_router

app = FastAPI(title="Morabaraba API", description="Backend for multiplayer Morabaraba board game")

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Include routers
app.include_router(game_router)
app.include_router(ai_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Morabaraba! Go to /static/index.html to play."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
