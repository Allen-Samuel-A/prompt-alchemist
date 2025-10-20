# backend/models/chat_models.py

from pydantic import BaseModel
from typing import List, Literal, Union, Dict, Any 

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
    messages: List[ChatMessage]
    
    target_model: str
    
    mode: str = "visual"
