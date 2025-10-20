# backend/api/v1/chat.py

from fastapi import APIRouter, HTTPException
# REMOVED: from pydantic import BaseModel
# REMOVED: from typing import List, Optional, Dict, Any

from core.alchemy_engine import process_chat_request
# UPDATED: Import all necessary models from the central models file
from models.chat_models import ChatMessage, ChatRequest, ChatResponse 

router = APIRouter()

# REMOVED: 
# class ChatRequest(BaseModel):
#     messages: List[Dict[str, str]]
#     target_model: str = "google/gemini-flash-1.5-8b"
#     mode: str = "guided"  # "guided" or "visual"
#
# class ChatResponse(BaseModel):
#     expert_prompt: str
#     explanation: str = ""
#     quality_score: Optional[Dict[str, Any]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Enhanced chat endpoint that returns prompts with quality scores
    """
    try:
        # Convert dict messages to ChatMessage objects
        # NOTE: This conversion is needed because the 'messages' field 
        # in the imported ChatRequest expects a List[Dict] from the API call, 
        # but the core logic expects a List of Pydantic ChatMessage objects.
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in request.messages
        ]
        
        # Process the request with new enhanced engine
        result = await process_chat_request(
            messages=chat_messages,
            model=request.target_model,
            mode=request.mode
        )
        
        # Return the final response using the imported ChatResponse model
        return ChatResponse(
            expert_prompt=result["expert_prompt"],
            explanation=result.get("explanation", ""),
            quality_score=result.get("quality_score")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )