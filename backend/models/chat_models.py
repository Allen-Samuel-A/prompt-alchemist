# backend/models/chat_models.py (CORRECTED CODE)

from pydantic import BaseModel
# UPDATED: We need Optional for the ChatResponse model
from typing import List, Literal, Union, Dict, Any, Optional 

# Pydantic models define the structure of your data.

class ChatMessage(BaseModel):
    """
    Represents a single message in the conversation.
    (Used internally by the alchemy_engine)
    """
    role: Literal["user", "assistant"]
    content: Union[str, Dict, Any] 

class ChatRequest(BaseModel):
    """
    Represents the request body that the frontend will send to our chat endpoint.
    """
    # CRITICAL FIX: The frontend sends raw JSON dicts, so we must tell Pydantic 
    # to expect a list of Dictionaries, not a list of ChatMessage objects.
    messages: List[Dict[str, str]] # <--- CHANGED BACK TO LIST[DICT[STR, STR]]
    
    target_model: str
    
    mode: str = "visual"

class ChatResponse(BaseModel):
    """
    Represents the structured response body returned by the API.
    """
    expert_prompt: str
    explanation: str = ""
    quality_score: Optional[Dict[str, Any]] = None