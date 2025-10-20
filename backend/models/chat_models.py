# backend/models/chat_models.py

from pydantic import BaseModel
# UPDATED: Added Optional to imports, as it's needed for quality_score
from typing import List, Literal, Union, Dict, Any, Optional 

# Pydantic models define the structure of your data.
# They provide data validation, serialization (Python -> JSON), 
# and deserialization (JSON -> Python).

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
    """
    # Note: Even though the frontend sends List[Dict], Pydantic automatically 
    # converts this into List[ChatMessage] objects during validation.
    messages: List[ChatMessage]
    
    target_model: str
    
    mode: str = "visual"

class ChatResponse(BaseModel):
    """
    Represents the structured response body returned by the API.
    (ADDED from the old chat.py)
    """
    expert_prompt: str
    explanation: str = ""
    quality_score: Optional[Dict[str, Any]] = None