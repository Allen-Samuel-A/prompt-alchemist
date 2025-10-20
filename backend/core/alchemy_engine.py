# backend/core/alchemy_engine.py

from typing import List, Dict, Optional, Tuple, Any 
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
    # Use only content from user messages that are strings (ignoring any potential assistant dicts)
    user_messages = [msg.content for msg in messages if msg.role == "user" and isinstance(msg.content, str)]
    
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
    
    # Fix 1: Check the last user message to see if they explicitly said "none" or "no"
    last_user_message = messages[-1].content.lower().strip() if messages and isinstance(messages[-1].content, str) else ""
    # Keywords that indicate the user wants to skip the current refinement step
    skip_keywords = ['none', 'no', 'skip', 'nothing', 'not needed', 'n/a']
    user_wants_to_skip = last_user_message in skip_keywords
    
    # If we have enough information (score >= 70), generate the prompt
    if analysis["completeness_score"] >= 70:
        return None
    
    # If this is the first message and it's too vague, ask for more details
    if analysis["message_count"] == 1 and analysis["completeness_score"] < 25:
        # Conversational Greeting
        return "Hello! I'm the Prompt Alchemist 🪄\n\nI'll help you create the perfect AI prompt through conversation. Just tell me what you'd like to create!\n\nExamples:\n• 'Write a resignation email to my boss'\n• 'Create a Python sorting function'\n• 'Generate a blog post about AI trends'\n\nWhat can I help you with?"

    
    # Ask specific questions based on what's missing
    task_type = analysis.get("task_type")
    
    # Question 1: Context
    if not analysis["has_context"] and not user_wants_to_skip:
        if task_type == "email":
            return "That's a great start! To make this email perfect, could you tell me who the email is for (like a boss, colleague, or client) and what the main situation is? More context helps me craft a precise prompt for you."
        elif task_type == "code":
            return "Got it, code generation! I need a little more detail: What exactly should the code accomplish, and what's the specific use case or problem it needs to solve? That helps me zero in on the best prompt."
        elif task_type == "blog":
            return "Awesome, a blog post! Who is your target audience, and what's the main idea or message you want them to take away? Focusing the audience helps a lot!"
        else:
            return "I need a bit more context to craft a really strong prompt. What's the main goal of your prompt, and who is the final audience? Knowing the purpose and the audience makes a huge difference!"
    
    # Question 2: Constraints
    # We only ask for constraints if context and task are present, AND the user didn't just skip the last question.
    if not analysis["has_constraints"] and analysis["has_context"] and analysis["has_task"]:
        # If the user just skipped, we move to the final 'ready' question instead of re-asking constraints.
        if user_wants_to_skip:
            return None # Skip to final 'ready to generate'
        
        # If the user did not skip, ask for constraints.
        if task_type == "email":
            return "We're almost there! Do you have any specific constraints? For example, should the tone be professional or casual, is there a length limit, or any key points that must be included? (Just type 'none' if you're flexible!)"
        elif task_type == "code":
            return "Great! For the code prompt, do you have any technical constraints? Which programming language should be used, are there performance goals, or any specific code styles required? (You can say 'none' if you're flexible!)"
        else:
            return "Last piece of info needed: Do you have any format or style constraints? This could be a length requirement, a specific tone you need (like funny or serious), or things the AI must be sure to avoid. (Type 'none' to move on!)"
    
    # If we reach here and the user has skipped (or we have enough info), we proceed to generation prompt
    if analysis["completeness_score"] >= 40 or user_wants_to_skip:
        return None # Ready to generate!
    
    # If we have task and context but not much detail, ask for refinement
    if analysis["completeness_score"] < 50:
        return "Thanks! Your idea is shaping up well. Would you like to add any more specific details about the exact outcome you want or any special requirements? If not, just say 'generate' and I'll create your prompt!"
    
    # Default: we have some info, ask if they want to add more or generate
    return "Got it! I have enough information to create a detailed prompt. Would you like to add anything else, or should I go ahead and generate the expert prompt now? Type 'generate' when you're ready!"


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
            
            # Handle greetings and start of conversation
            if first_message in ['hi', 'hello', 'hey', 'help', 'start', 'hifj', 'ello', 'hiiii', 'hey there']:
                # Return the conversational greeting from generate_smart_question
                greeting = generate_smart_question({"message_count": 1, "completeness_score": 0}, messages)
                return {
                    "expert_prompt": greeting,
                    "explanation": "Starting guided conversation to understand your needs."
                }
        
        # Analyze the conversation to understand what we have
        analysis = analyze_conversation(messages)
        
        logger.info(f"Conversation analysis: {analysis}")
        
        # Check if user is saying they're ready to generate
        last_message = messages[-1].content.lower()
        ready_keywords = ['generate', 'ready', 'create it', 'make it', 'go ahead', "that's all", 'done', 'perfect']
        user_wants_to_generate = any(keyword in last_message for keyword in ready_keywords)
        
        # Determine if we should generate the prompt now
        next_question = generate_smart_question(analysis, messages)
        
        # Generate prompt if we have enough info OR user explicitly requests it AND there's no follow-up question
        if (analysis["completeness_score"] >= 70 or (user_wants_to_generate and analysis["completeness_score"] >= 40) or next_question is None):
            logger.info("Generating prompt - sufficient information collected or user requested.")
            
            # Compile all user messages into context
            user_context = "\n".join([msg.content for msg in messages if msg.role == "user" and isinstance(msg.content, str)])
            
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
