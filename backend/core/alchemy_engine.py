# backend/core/alchemy_engine.py

from typing import List, Dict, Optional, Tuple
from models.chat_models import ChatMessage
from services.openrouter_client import get_ai_response
import logging
import re

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
# CONVERSATION INTELLIGENCE
# ==========================================
def analyze_conversation(messages: List[ChatMessage]) -> Dict[str, any]:
    """
    Analyzes the conversation to understand what information has been gathered.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Dictionary with analysis results
    """
    user_messages = [msg.content for msg in messages if msg.role == "user"]
    
    # Combine all user messages to analyze
    full_conversation = " ".join(user_messages).lower()
    
    analysis = {
        "has_task": False,
        "has_role": False,
        "has_context": False,
        "has_constraints": False,
        "task_type": None,
        "completeness_score": 0,
        "message_count": len(user_messages)
    }
    
    # Detect task
    task_indicators = [
        "write", "create", "generate", "make", "build", "design", "develop",
        "draft", "compose", "code", "script", "email", "letter", "blog",
        "post", "article", "function", "program", "campaign"
    ]
    if any(indicator in full_conversation for indicator in task_indicators):
        analysis["has_task"] = True
        analysis["completeness_score"] += 25
        
        # Identify task type
        if any(word in full_conversation for word in ["email", "letter", "message"]):
            analysis["task_type"] = "email"
        elif any(word in full_conversation for word in ["code", "function", "script", "program"]):
            analysis["task_type"] = "code"
        elif any(word in full_conversation for word in ["blog", "post", "article"]):
            analysis["task_type"] = "blog"
        elif any(word in full_conversation for word in ["campaign", "marketing", "ad"]):
            analysis["task_type"] = "marketing"
    
    # Detect role/expertise mentions
    role_indicators = [
        "expert", "developer", "engineer", "writer", "designer", "analyst",
        "specialist", "professional", "senior", "junior", "manager"
    ]
    if any(indicator in full_conversation for indicator in role_indicators):
        analysis["has_role"] = True
        analysis["completeness_score"] += 20
    
    # Detect context
    context_indicators = [
        "for", "audience", "purpose", "background", "about", "regarding",
        "boss", "client", "customer", "user", "team", "company", "project"
    ]
    if any(indicator in full_conversation for indicator in context_indicators):
        analysis["has_context"] = True
        analysis["completeness_score"] += 25
    
    # Detect constraints
    constraint_indicators = [
        "should", "must", "need to", "require", "limit", "words", "tone",
        "style", "format", "professional", "casual", "formal", "length"
    ]
    if any(indicator in full_conversation for indicator in constraint_indicators):
        analysis["has_constraints"] = True
        analysis["completeness_score"] += 20
    
    # Bonus points for detailed messages
    if len(full_conversation) > 50:
        analysis["completeness_score"] += 10
    
    return analysis


def generate_smart_question(analysis: Dict[str, any], messages: List[ChatMessage]) -> Optional[str]:
    """
    Generates an intelligent follow-up question based on what's missing.
    
    Args:
        analysis: Analysis results from analyze_conversation
        messages: Conversation history
        
    Returns:
        Smart question to ask, or None if ready to generate
    """
    # If we have enough information (score >= 70), generate the prompt
    if analysis["completeness_score"] >= 70:
        return None
    
    # If this is the first message and it's too vague, ask for more details
    if analysis["message_count"] == 1 and analysis["completeness_score"] < 25:
        return "I'd love to help you create a great prompt! Could you tell me more about what you'd like to accomplish? For example: 'Write a professional email to resign from my job' or 'Create a Python function to sort data'."
    
    # Ask specific questions based on what's missing
    task_type = analysis.get("task_type")
    
    if not analysis["has_context"]:
        if task_type == "email":
            return "Great! To make this perfect, could you tell me:\n\n• Who is the email for? (boss, colleague, client)\n• What's the main purpose or situation?\n\nThe more context you provide, the better the prompt!"
        elif task_type == "code":
            return "Sounds good! To create the best prompt, I need a bit more info:\n\n• What should the code accomplish?\n• Any specific requirements or use case?\n\nThis helps me craft a precise prompt for you."
        elif task_type == "blog":
            return "Excellent! Let me ask:\n\n• Who is your target audience?\n• What's the main topic or message?\n\nThis will help me create a focused prompt."
        else:
            return "To create the perfect prompt, could you provide more context?\n\n• What's the purpose or goal?\n• Who is the audience?\n• Any specific situation or background?\n\nMore details = better results!"
    
    if not analysis["has_constraints"]:
        if task_type == "email":
            return "Perfect! One more thing - any specific requirements?\n\n• Tone: Professional, casual, or formal?\n• Length preference?\n• Any key points to include or avoid?\n\n(Or type 'none' if no specific constraints)"
        elif task_type == "code":
            return "Almost there! Any technical constraints?\n\n• Programming language preferences?\n• Performance requirements?\n• Code style or standards?\n\n(Type 'none' if you're flexible)"
        else:
            return "Last question - any constraints or requirements?\n\n• Length or format preferences?\n• Tone or style guidelines?\n• Specific things to include or avoid?\n\n(Type 'none' if no constraints)"
    
    # If we have task and context but not much detail, ask for refinement
    if analysis["completeness_score"] < 50:
        return "Thanks! To make this even better, could you add any specific details about:\n\n• The exact outcome you want\n• Any special requirements\n• The level of detail needed\n\nOr if you're happy with what you've provided, just say 'generate' and I'll create your prompt!"
    
    # Default: we have some info, ask if they want to add more or generate
    return "Got it! I have enough to work with. Would you like to:\n\n• Add more details? Just tell me what else you'd like\n• Generate the prompt now? Type 'generate' or 'ready'\n\nWhat works for you?"


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
The user has provided their requirements through a conversation. Your job is to TRANSFORM this into a 
professional, detailed, research-backed prompt that delivers excellent results.

You must ADD VALUE by:
- Making instructions more specific and actionable
- Adding relevant context and background
- Including best practices from research
- Providing clear success criteria
- Expanding with helpful details and examples

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

### USER'S REQUIREMENTS FROM CONVERSATION
{user_idea}

### STRICT FORMATTING RULES
1. Start IMMEDIATELY with "### Prompt" (no preamble)
2. ENHANCE each of the 4 components with specific, actionable details
3. Make it comprehensive and production-ready
4. Add "---EXPLANATION---" after the prompt
5. Write 2-4 sentences explaining what enhancements you made
6. STOP after explanation - add nothing else

EXAMPLE OF GOOD OUTPUT:
### Prompt
**Role:** You are a professional HR communication specialist with expertise in crafting diplomatic resignation letters that maintain positive relationships and professional reputation.

**Task:** Compose a formal resignation letter that clearly communicates the decision to leave, expresses gratitude for opportunities received, offers appropriate transition support, and maintains a respectful, professional tone throughout.

**Context:** This letter will be sent to the employee's direct supervisor and HR department. It needs to be formal yet warm, brief but complete, and should leave the door open for future professional connections. The employee wants to leave on good terms and maintain their professional reputation.

**Constraints:** Keep the letter to 150-200 words, use professional business letter formatting, maintain a positive and grateful tone throughout, avoid any negative comments or reasons for leaving, include a clear last working day (2 weeks from date), offer to help with transition, and end with appreciation for the experience gained.

---EXPLANATION---
This prompt transforms the basic request into a comprehensive guide by specifying professional expertise, defining all deliverables with clear tone requirements, providing crucial context about recipients and goals, and establishing specific constraints to ensure a polished, professional result.

Now create the enhanced prompt following this exact format."""


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
        
        # Remove unwanted sections from prompt
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
    # GUIDED MODE: Intelligent conversation flow
    # ==========================================
    if mode == "guided":
        # Check if this is the first message
        user_messages = [msg for msg in messages if msg.role == "user"]
        
        if len(user_messages) == 1:
            first_message = user_messages[0].content.lower().strip()
            
            # Handle greetings
            if first_message in ['hi', 'hello', 'hey', 'help', 'start']:
                return {
                    "expert_prompt": "Hello! I'm the Prompt Alchemist 🪄\n\nI'll help you create the perfect AI prompt through conversation. Just tell me what you'd like to create!\n\nExamples:\n• 'Write a resignation email to my boss'\n• 'Create a Python sorting function'\n• 'Generate a blog post about AI trends'\n\nWhat can I help you with?",
                    "explanation": "Starting guided conversation to understand your needs."
                }
        
        # Analyze the conversation to understand what we have
        analysis = analyze_conversation(messages)
        
        logger.info(f"Conversation analysis: {analysis}")
        
        # Check if user is saying they're ready to generate
        last_message = messages[-1].content.lower()
        ready_keywords = ['generate', 'ready', 'create it', 'make it', 'go ahead', "that's all", 'done', 'perfect']
        user_wants_to_generate = any(keyword in last_message for keyword in ready_keywords)
        
        # Generate prompt if we have enough info OR user explicitly requests it
        if analysis["completeness_score"] >= 70 or (user_wants_to_generate and analysis["completeness_score"] >= 40):
            logger.info("Generating prompt - sufficient information collected")
            
            # Compile all user messages into context
            user_context = "\n".join([msg.content for msg in messages if msg.role == "user"])
            
            system_prompt = create_system_prompt(user_context, model)
            api_messages = [ChatMessage(role="user", content=system_prompt)]
            
            # Call API with fallback
            raw_response, error = await call_ai_with_fallback(api_messages)
            
            if error == "rate_limit":
                return {
                    "expert_prompt": "⚠️ High traffic detected. The service is temporarily at capacity. Please wait 30-60 seconds and try again.",
                    "explanation": "Rate limit encountered. Try again in a moment."
                }
            
            if error:
                return {
                    "expert_prompt": f"⚠️ Unable to generate prompt: {error}",
                    "explanation": "An unexpected error occurred. Please try again."
                }
            
            # Format and return the final prompt
            prompt, explanation = format_response(raw_response)
            return {"expert_prompt": prompt, "explanation": explanation}
        
        # Not enough info yet - ask a smart follow-up question
        next_question = generate_smart_question(analysis, messages)
        
        if next_question:
            return {
                "expert_prompt": next_question,
                "explanation": f"Gathering more details to create the perfect prompt (completeness: {analysis['completeness_score']}%)."
            }
        
        # Fallback - shouldn't reach here, but just in case
        return {
            "expert_prompt": "I think I have what I need! Would you like me to generate your prompt now, or would you like to add more details?",
            "explanation": "Ready to generate when you are."
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