# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the router from our chat endpoint file
from backend.api.v1 import chat

# Create an instance of the FastAPI class
app = FastAPI(
    title="Prompt Alchemist API",
    description="The backend for a conversational prompt engineering assistant.",
    version="0.1.0",
)

# --- Middleware ---
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
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
# All routes defined in chat.router will now be part of the main app.
# The 'prefix' adds '/api/v1' to the start of all routes from that router.
# So, the '/chat' endpoint becomes '/api/v1/chat'.
# The 'tags' are used to group the endpoints in the automatic API docs.
app.include_router(chat.router, prefix="/api/v1", tags=["v1"])


# --- API Endpoints ---
@app.get("/")
def read_root():
    """
    A simple welcome endpoint for our API.
    """
    return {"message": "Welcome to the Prompt Alchemist Backend! Let's create some magic. ✨"}