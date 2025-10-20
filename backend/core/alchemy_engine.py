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

### YOUR MISSION
The user has provided a BASIC prompt structure. Your job is to TRANSFORM and ENHANCE it into a 
professional, detailed, research-backed prompt that is SIGNIFICANTLY BETTER than what they provided.

DO NOT simply reformat their input. You must ADD VALUE by:
- Making instructions more specific and actionable
- Adding relevant context and background
- Including best practices from research
- Providing clear success criteria
- Expanding constraints with helpful details

### CRITICAL OUTPUT REQUIREMENTS
You MUST output EXACTLY TWO sections with NOTHING ELSE:

Section 1: The Enhanced Prompt (starts with "### Prompt")
Section 2: The Explanation (starts with "---EXPLANATION---")

DO NOT include:
- Solution matrices or tables
- Recommended paths or strategies
- Code blocks or JSON
- Multiple prompt versions
- Any text before "### Prompt" or after the explanation

### PROMPT STRUCTURE
Every enhanced prompt MUST include these four components with SUBSTANTIAL DETAIL:

**Role:** [Expand on the AI's expertise - add specific skills, knowledge domains, and perspective]
**Task:** [Make the objective crystal clear with specific deliverables and success criteria]
**Context:** [Add relevant background, audience details, use case scenarios, and environmental factors]
**Constraints:** [Expand with quality standards, format requirements, style guidelines, and boundaries]

### RESEARCH-BACKED GUIDELINES FOR {target_model.upper()}
{research}

### USER'S BASIC INPUT (TRANSFORM THIS INTO AN EXPERT-LEVEL PROMPT)
{user_idea}

### STRICT FORMATTING RULES
1. Start IMMEDIATELY with "### Prompt" (no preamble)
2. ENHANCE each of the 4 components with specific, actionable details
3. Make it AT LEAST 2-3x more detailed than the user's input
4. Add "---EXPLANATION---" after the prompt
5. Write 2-4 sentences explaining what research-backed improvements you made
6. STOP after explanation - add nothing else

EXAMPLE OF ENHANCEMENT:
User Input: "Role: Writer, Task: Write blog post"
Your Output: "**Role:** Expert Technology Blogger with 10+ years experience in translating complex technical concepts into engaging narratives for non-technical audiences, specializing in trend analysis and thought leadership"

Now enhance the user's prompt following this approach."""


# ==========================================
# CONVERSATION FLOW MANAGEMENT
# ==========================================
class GuidedFlowManager:
    """Manages the step-by-step guided conversation flow"""
    
    FLOW_STEPS = [
        {
            "trigger": None,  # Initial state
            "prompt": "Hello! I'm the Prompt Alchemist 🪄 Let's craft an expert-level prompt together.\n\nWhat would you like to create? (e.g., 'Write a blog post', 'Generate Python code', 'Create a marketing email')",
            "explanation": "Starting guided prompt creation process."
        },
        {
            "trigger": "goal or idea",
            "prompt": "Great! What expertise or role should the AI have?\n\nExamples:\n• Senior Software Engineer\n• Digital Marketing Specialist\n• Technical Writer\n• Data Analyst",
            "explanation": "Role definition ensures the AI adopts the right perspective."
        },
        {
            "trigger": "role should the AI",
            "prompt": "Perfect! Now describe the specific task in detail.\n\nBe specific about what you want the AI to do. The clearer you are, the better the result!",
            "explanation": "Clear task definition is crucial for focused output."
        },
        {
            "trigger": "main task",
            "prompt": "Excellent! Please provide context:\n\n• Who is the audience?\n• What's the purpose?\n• Any background information?\n\n(This helps the AI understand the bigger picture)",
            "explanation": "Context helps the AI understand the bigger picture."
        },
        {
            "trigger": "context",
            "prompt": "Almost done! Any constraints or requirements?\n\nExamples:\n• Word count limits\n• Specific tone or style\n• Technical requirements\n• Format preferences\n\n(Type 'none' if no constraints)",
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
    Cleans up unwanted content like solution matrices and recommended paths.
    
    Args:
        raw_response: Raw AI response text
        
    Returns:
        Tuple of (expert_prompt, explanation)
    """
    # Clean up the response
    raw_response = raw_response.strip()
    
    # Remove any text before "### Prompt"
    if "### Prompt" in raw_response:
        raw_response = "### Prompt" + raw_response.split("### Prompt", 1)[1]
    elif not raw_response.startswith("### Prompt"):
        raw_response = "### Prompt\n" + raw_response
    
    # Split into prompt and explanation
    if "---EXPLANATION---" in raw_response:
        parts = raw_response.split("---EXPLANATION---", 1)
        prompt_part = parts[0].strip()
        explanation_part = parts[1].strip()
        
        # Remove unwanted sections from prompt (Solution Matrix, Recommended Path, etc.)
        unwanted_markers = [
            "### Solution Matrix",
            "### Recommended Path", 
            "### Alternative",
            "| Solution |",
            "```json",
            "```"
        ]
        
        for marker in unwanted_markers:
            if marker in prompt_part:
                prompt_part = prompt_part.split(marker)[0].strip()
        
        # Clean up explanation - take only first paragraph if too long
        explanation_lines = explanation_part.split('\n\n')
        if len(explanation_lines) > 1:
            explanation_part = explanation_lines[0]
        
        return prompt_part, explanation_part
    
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