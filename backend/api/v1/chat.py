# backend/api/v1/chat.py

from fastapi import APIRouter, HTTPException
from core.alchemy_engine import process_chat_request, refine_prompt_with_framework, audit_generated_prompt
# UPDATED: Import all necessary models from the central models file
from models.chat_models import ChatMessage, ChatRequest, ChatResponse, RefineRequest

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


@router.post("/refine", response_model=ChatResponse)
async def refine_endpoint(request: RefineRequest):
    """
    Refines an existing prompt by injecting advanced optimization frameworks.
    Called when user clicks the 'Refine' button in the UI.
    
    Expected flow:
    1. User generates a prompt via /chat
    2. System shows quality score with optimization suggestion
    3. User clicks "Refine using CoT" (or ToT, Self-Consistency, etc.)
    4. Frontend sends original prompt + framework suggestion to /refine
    5. This endpoint injects the framework and returns enhanced prompt
    """
    try:
        # Call the refine function from alchemy_engine
        refined_prompt, explanation = await refine_prompt_with_framework(
            original_prompt=request.original_prompt,
            framework_suggestion=request.framework_suggestion,
            model=request.target_model
        )
        
        # Re-audit the refined prompt to show updated quality metrics
        quality_score = await audit_generated_prompt(
            expert_prompt=refined_prompt,
            target_model=request.target_model,
            task_category=request.task_category
        )
        
        return ChatResponse(
            expert_prompt=refined_prompt,
            explanation=f"✨ Refined with {request.framework_suggestion.split('(')[0].strip()}. {explanation}",
            quality_score=quality_score.model_dump() if quality_score else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Refinement failed: {str(e)}"
        )