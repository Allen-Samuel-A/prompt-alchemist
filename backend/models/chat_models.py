# backend/models/chat_models.py

from pydantic import BaseModel, Field
from typing import List, Literal, Union, Dict, Any, Optional 

# Pydantic models define the structure of your data.

class AuditDimension(BaseModel):
    """Schema for individual audit criteria."""
    score: int = Field(..., ge=0, le=100)
    feedback: str

class AuditResult(BaseModel):
    """Detailed structure for the objective quality audit."""
    overall_score: int = Field(..., ge=0, le=100)
    grade: str
    estimated_success_rate: str
    dimensions: Dict[str, AuditDimension]
    strengths: List[str]
    suggestions: List[str]

class ChatMessage(BaseModel):
    """
    Represents a single message in the conversation.
    """
    role: Literal["user", "assistant"]
    content: Union[str, Dict, Any] 

class ChatRequest(BaseModel):
    """
    Represents the request body that the frontend will send to our chat endpoint.
    """
    messages: List[Dict[str, str]]
    target_model: str
    mode: str = "visual"

class ChatResponse(BaseModel):
    """
    Represents the structured response body returned by the API.
    quality_score now explicitly uses the detailed AuditResult schema.
    """
    expert_prompt: str
    explanation: str = ""
    quality_score: Optional[AuditResult] = None # Updated to use AuditResult
