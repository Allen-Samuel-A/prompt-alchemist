# backend/api/v1/chat.py

from fastapi import APIRouter, HTTPException
from core.alchemy_engine import process_chat_request
# UPDATED: Import all necessary models from the central models file
from models.chat_models import ChatMessage, ChatRequest, ChatResponse 

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Enhanced chat endpoint that receives user intent and returns a quality-scored expert prompt.
    """
    try:
        # Convert dict messages from the API request into Pydantic ChatMessage objects
        # This standardizes the data for the core processing engine.
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in request.messages
        ]
        
        # Process the request using the multi-layered Prompt Alchemist engine.
        # It's CRITICAL that request.target_model is passed here for the Audit (Layer 2)
        # to calculate cost, check model size thresholds, and apply model-specific research.
        result = await process_chat_request(
            messages=chat_messages,
            model=request.target_model,
            mode=request.mode
        )
        
        # Return the final structured response. FastAPI automatically validates this
        # against the ChatResponse schema imported from models/chat_models.py.
        return ChatResponse(
            expert_prompt=result["expert_prompt"],
            explanation=result.get("explanation", ""),
            quality_score=result.get("quality_score")
        )
        
    except Exception as e:
        # Catch any unexpected errors during processing or LLM calls
        # and return a standard HTTP 500 server error.
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
