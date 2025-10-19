# backend/core/alchemy_engine.py

from typing import List, Dict, Optional, Tuple
from models.chat_models import ChatMessage
from services.openrouter_client import get_ai_response
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
class ModelConfig:
    """Centralized model configuration with fallback strategy"""
    PRIMARY_MODEL = "openai/gpt-4o-mini"
    FALLBACK_MODELS = [
        "anthropic/claude-3-haiku",
        "openai/gpt-3.5-turbo"
    ]
    MAX_RETRIES = 2
    
    # Research data organized by model family
    RESEARCH_DATA = {
        ("gpt", "openai"): """
- Be explicit and place key instructions at the beginning
- Use clear separators (###, ---) to organize information hierarchically
- Request step-by-step reasoning for complex tasks to improve accuracy
- Provide examples in the prompt for better output consistency
""",
        ("claude", "anthropic"): """
- Use XML-like tags for structured sections (<role>, <task>, <context>)
- Frame instructions positively rather than using prohibitions
- Pre-fill the start of the expected output format for better adherence
- Assign specific roles or personas for more targeted responses
""",
        ("gemini", "google"): """
- Use persona-based prompts with clear role definitions
- Provide comprehensive context for richer, more detailed results
- Break complex tasks into explicit substeps for improved coherence
- Include examples of desired output format
""",
        ("llama", "meta"): """
- Structure prompts using clear markdown layouts with headers
- Provide explicit examples and constraints for better reliability
- Include detailed formatting guidance to reduce output drift
- Use numbered steps for sequential tasks
""",
        "default": """
- Clearly separate role, task, context, and constraints
- Include relevant background context to improve response quality
- Define expected output format explicitly with examples
- Use positive framing for instructions
"""
    }


# ==========================================
# RESEARCH UTILITIES
# ==========================================
def get_research_data(query: str) -> str:
    """
    Retrieves model-specific research data using optimized lookup.
    
    Args:
        query: Search query containing model information
        
    Returns:
        Research guidelines for the specified model family
    """
    logger.info(f"Fetching research data for query: {query}")
    
    query_lower = query.lower()
    
    for keywords, results in ModelConfig.RESEARCH_DATA.items():
        if keywords == "default":
            continue
        if any(keyword in query_lower for keyword in keywords):
            return results
    
    return ModelConfig.RESEARCH_DATA["default"]


# ==========================================
# PROMPT ENGINEERING
# ==========================================
def create_system_prompt(user_idea: str, target_model: str) -> str:
    """
    Generates an optimized system prompt for the Prompt Alchemist.
    
    Args:
        user_idea: User's input describing their prompt requirements
        target_model: Target LLM model for optimization
        
    Returns:
        Complete system prompt with research integration
    """
    research = get_research_data(f"prompting techniques for {target_model}")
    
    return f"""You are 'Prompt Alchemist', an expert AI prompt engineer specializing in creating 
production-ready prompts optimized for {target_model}.

### YOUR TASK
Transform the user's idea into a professionally structured, research-backed prompt that follows 
best practices for {target_model}.

### CRITICAL OUTPUT REQUIREMENTS
You MUST output EXACTLY TWO sections with NOTHING ELSE:

Section 1: The Prompt (starts with "### Prompt")
Section 2: The Explanation (starts with "---EXPLANATION---")

DO NOT include:
- Solution matrices or tables
- Recommended paths or strategies
- Code blocks or JSON
- Multiple prompt versions
- Any text before "### Prompt" or after the explanation

### PROMPT STRUCTURE
Every prompt you create MUST include these four components:

**Role:** [Define the AI's expertise, persona, or domain knowledge]
**Task:** [State the specific objective or goal clearly and actionably]
**Context:** [Provide background, audience, constraints, or environmental details]
**Constraints:** [List rules, format requirements, limitations, or quality standards]

### RESEARCH-BACKED GUIDELINES
Apply these model-specific best practices for {target_model}:
{research}

### USER INPUT
{user_idea}

### STRICT FORMATTING RULES
1. Start IMMEDIATELY with "### Prompt" (no preamble)
2. Include all 4 components: Role, Task, Context, Constraints
3. Add "---EXPLANATION---" after the prompt
4. Write 2-4 sentences explaining how you applied research
5. STOP after explanation - add nothing else

EXAMPLE OUTPUT FORMAT:
### Prompt
**Role:** [role here]
**Task:** [task here]
**Context:** [context here]
**Constraints:** [constraints here]

---EXPLANATION---
[2-4 sentences referencing research application]

Output the final prompt now following this EXACT format."""


# ==========================================
# CONVERSATION FLOW MANAGEMENT
# ==========================================
class GuidedFlowManager:
    """Manages the step-by-step guided conversation flow"""
    
    FLOW_STEPS = [
        {
            "trigger": None,  # Initial state
            "prompt": "Hello! I'm the Prompt Alchemist. Let's craft an expert-level prompt together. What's your main goal or idea?",
            "explanation": "Starting guided prompt creation process."
        },
        {
            "trigger": "goal or idea",
            "prompt": "Got it! What role or expertise should the AI embody? (e.g., 'senior data analyst', 'creative copywriter', 'technical architect')",
            "explanation": "Role definition ensures the AI adopts the right perspective."
        },
        {
            "trigger": "role should the AI",
            "prompt": "Perfect. Now, what's the specific task or objective? Be as precise as possible.",
            "explanation": "Clear task definition is crucial for focused output."
        },
        {
            "trigger": "main task",
            "prompt": "Excellent. Please share any relevant context: audience, environment, background information, or use case details.",
            "explanation": "Context helps the AI understand the bigger picture."
        },
        {
            "trigger": "context",
            "prompt": "Almost there! Are there any constraints, formatting requirements, or quality standards I should include?",
            "explanation": "Constraints shape the boundaries of the AI's creativity."
        }
    ]
    
    @staticmethod
    def get_last_assistant_message(messages: List[ChatMessage]) -> Optional[str]:
        """Extract the last assistant message from conversation history"""
        for msg in reversed(messages[:-1]):
            if msg.role == "assistant":
                if isinstance(msg.content, dict):
                    return msg.content.get("expert_prompt", "")
                return msg.content
        return None
    
    @staticmethod
    def get_current_step(last_message: Optional[str]) -> Optional[Dict[str, str]]:
        """Determine current conversation step based on last message"""
        if last_message is None or "start over" in last_message.lower():
            return GuidedFlowManager.FLOW_STEPS[0]
        
        for step in GuidedFlowManager.FLOW_STEPS[1:]:
            if step["trigger"] and step["trigger"] in last_message.lower():
                idx = GuidedFlowManager.FLOW_STEPS.index(step)
                if idx + 1 < len(GuidedFlowManager.FLOW_STEPS):
                    return GuidedFlowManager.FLOW_STEPS[idx + 1]
        
        return None


# ==========================================
# API INTERACTION WITH RETRY LOGIC
# ==========================================
async def call_ai_with_fallback(
    messages: List[ChatMessage],
    primary_model: str = ModelConfig.PRIMARY_MODEL
) -> Tuple[str, Optional[str]]:
    """
    Calls AI API with fallback strategy for reliability.
    
    Args:
        messages: List of chat messages for the API
        primary_model: Primary model to attempt first
        
    Returns:
        Tuple of (response_text, error_message)
    """
    models_to_try = [primary_model] + ModelConfig.FALLBACK_MODELS
    
    for attempt, model in enumerate(models_to_try):
        try:
            logger.info(f"Attempting API call with model: {model} (attempt {attempt + 1})")
            response = await get_ai_response(messages=messages, model=model)
            return response, None
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Model {model} failed: {error_msg}")
            
            # Don't retry on rate limits - inform user immediately
            if "429" in error_msg:
                return "", "rate_limit"
            
            # Try next model if available
            if attempt < len(models_to_try) - 1:
                continue
            
            # All models failed
            return "", error_msg
    
    return "", "All models exhausted"


def format_response(raw_response: str) -> Tuple[str, str]:
    """
    Ensures consistent response formatting with prompt and explanation.
    
    Args:
        raw_response: Raw AI response text
        
    Returns:
        Tuple of (expert_prompt, explanation)
    """
    # Ensure proper structure
    if not raw_response.strip().startswith("### Prompt"):
        raw_response = "### Prompt\n" + raw_response.strip()
    
    # Split into prompt and explanation
    if "---EXPLANATION---" in raw_response:
        parts = raw_response.split("---EXPLANATION---", 1)
        return parts[0].strip(), parts[1].strip()
    
    # Add missing explanation
    return raw_response.strip(), "Prompt generated based on best practices and research."


# ==========================================
# MAIN PROCESSING LOGIC
# ==========================================
async def process_chat_request(
    messages: List[ChatMessage],
    model: str,
    mode: str
) -> Dict[str, str]:
    """
    Main entry point for processing chat requests in both modes.
    
    Args:
        messages: Conversation history
        model: Target LLM model for optimization
        mode: "guided" or "visual" builder mode
        
    Returns:
        Dictionary with expert_prompt and explanation keys
    """
    if not messages:
        logger.error("Empty messages list received")
        return {
            "expert_prompt": "Error: No messages provided.",
            "explanation": "Unable to process empty conversation."
        }
    
    # ==========================================
    # GUIDED MODE: Step-by-step interaction
    # ==========================================
    if mode == "guided":
        last_msg = GuidedFlowManager.get_last_assistant_message(messages)
        current_step = GuidedFlowManager.get_current_step(last_msg)
        
        # Return next question in the flow
        if current_step:
            return {
                "expert_prompt": current_step["prompt"],
                "explanation": current_step["explanation"]
            }
        
        # All questions answered - generate final prompt
        user_answers = [msg.content for msg in messages if msg.role == "user"]
        
        if len(user_answers) >= 5:
            # Extract the 5 answers: goal, role, task, context, constraints
            assembled_idea = f"""Goal: {user_answers[-5]}
Role: {user_answers[-4]}
Task: {user_answers[-3]}
Context: {user_answers[-2]}
Constraints: {user_answers[-1]}"""
            
            system_prompt = create_system_prompt(assembled_idea, model)
            api_messages = [ChatMessage(role="user", content=system_prompt)]
            
            # Call API with fallback
            raw_response, error = await call_ai_with_fallback(api_messages)
            
            if error == "rate_limit":
                return {
                    "expert_prompt": "⚠️ High traffic detected. The service is temporarily at capacity. Please wait 30-60 seconds and try again.",
                    "explanation": "Rate limit encountered. Consider upgrading to a paid API tier for guaranteed availability."
                }
            
            if error:
                return {
                    "expert_prompt": f"⚠️ Unable to generate prompt: {error}",
                    "explanation": "An unexpected error occurred. Please try again or contact support."
                }
            
            # Format and return
            prompt, explanation = format_response(raw_response)
            return {"expert_prompt": prompt, "explanation": explanation}
        
        # Conversation flow broken - restart
        return {
            "expert_prompt": "I need a bit more information. Let's start over. What's your main goal?",
            "explanation": "Restarting guided flow to ensure quality."
        }
    
    # ==========================================
    # VISUAL BUILDER MODE: Direct generation
    # ==========================================
    else:
        last_user_message = messages[-1].content
        if not isinstance(last_user_message, str):
            last_user_message = str(last_user_message)
        
        system_prompt = create_system_prompt(last_user_message, model)
        api_messages = [ChatMessage(role="user", content=system_prompt)]
        
        # Call API with fallback
        raw_response, error = await call_ai_with_fallback(api_messages)
        
        if error == "rate_limit":
            return {
                "expert_prompt": "⚠️ Service at capacity. Please wait a moment and retry.",
                "explanation": "High traffic volume. Consider using during off-peak hours."
            }
        
        if error:
            return {
                "expert_prompt": f"⚠️ Generation failed: {error}",
                "explanation": "Please check your input and try again."
            }
        
        # Format and return
        prompt, explanation = format_response(raw_response)
        return {"expert_prompt": prompt, "explanation": explanation}