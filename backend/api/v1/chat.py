# backend/api/v1/chat.py

from fastapi import APIRouter
from backend.models.chat_models import ChatRequest
from pydantic import BaseModel
from backend.core.alchemy_engine import process_chat_request

router = APIRouter()

# Define our response structure to match the new engine output
class AlchemyResponse(BaseModel):
    expert_prompt: str
    explanation: str

# Update the response_model to use our new AlchemyResponse
@router.post("/chat", response_model=AlchemyResponse)
async def handle_chat_request(request: ChatRequest):
    """
    Handles the chat request and returns a structured response
    with an expert-crafted prompt and an explanation.
    """
    # We now pass the 'mode' from the request to our engine
    response_data = await process_chat_request(
        messages=request.messages, 
        model=request.target_model,
        mode=request.mode
    )
    return response_data