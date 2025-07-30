# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the router from our chat endpoint file (path fixed for Render)
from api.v1 import chat

# Create an instance of the FastAPI class
app = FastAPI(
    title="Prompt Alchemist API",
    description="The backend for a conversational prompt engineering assistant.",
    version="0.1.0",
)

# --- Middleware ---
# We will add the live frontend URL here later
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "https://chipper-bombolone-83adcd.netlify.app",   # Netlify preview
    "https://whattoprompt.com",                       # your custom domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---
# Include the router from our v1 chat API.
app.include_router(chat.router, prefix="/api/v1", tags=["v1"])


# --- API Endpoints ---
@app.get("/")
def read_root():
    """
    A simple welcome endpoint for our API.
    """
    return {"message": "Welcome to the Prompt Alchemist Backend! Let's create some magic. ✨"}
