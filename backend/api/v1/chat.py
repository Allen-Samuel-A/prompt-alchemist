# backend/api/v1/chat.py

from fastapi import APIRouter
# Import paths fixed for Render
from models.chat_models import ChatRequest
from pydantic import BaseModel
from core.alchemy_engine import process_chat_request

router = APIRouter()

# Define our response structure
class AlchemyResponse(BaseModel):
    expert_prompt: str
    explanation: str

@router.post("/chat", response_model=AlchemyResponse)
async def handle_chat_request(request: ChatRequest):
    """
    Handles the chat request and returns a structured response.
    """
    response_data = await process_chat_request(
        messages=request.messages, 
        model=request.target_model,
        mode=request.mode
    )
    return response_data
