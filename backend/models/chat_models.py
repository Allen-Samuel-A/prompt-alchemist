# backend/models/chat_models.py

from pydantic import BaseModel
# UPDATED: Added Optional for ChatResponse, and Dict for ChatRequest compatibility
from typing import List, Literal, Union, Dict, Any, Optional 

# Pydantic models define the structure of your data.

class ChatMessage(BaseModel):
    """
    Represents a single message in the conversation.
    """
    role: Literal["user", "assistant"]
    
    # The content can now be a string (for user messages) OR
    # a dictionary/object (for assistant messages from history).
    content: Union[str, Dict, Any] 

class ChatRequest(BaseModel):
    """
    Represents the request body that the frontend will send to our chat endpoint.
    (NOTE: Kept as List[Dict[str, str]] for stable compatibility with frontend JSON payload)
    """
    messages: List[Dict[str, str]] # <--- Kept stable, expecting raw JSON dicts
    
    target_model: str
    
    mode: str = "visual"

class ChatResponse(BaseModel):
    """
    Represents the structured response body returned by the API.
    """
    expert_prompt: str
    explanation: str = ""
    quality_score: Optional[Dict[str, Any]] = None
