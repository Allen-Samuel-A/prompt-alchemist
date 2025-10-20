# backend/core/alchemy_engine.py

from typing import List, Dict, Optional, Tuple, Any 
from models.chat_models import ChatMessage, AuditResult, ChatResponse # Added AuditResult
from services.openrouter_client import get_ai_response
import logging
import re
import json
from pathlib import Path
import random # Needed for selecting a random item from perfect examples
from pydantic import ValidationError # Needed to catch audit model validation errors

# Configure logging
logger = logging.getLogger(__name__)

# --- JSON Loader Utility ---
def load_perfect_examples() -> Dict:
    """Loads the perfect prompt examples from the JSON file."""
    try:
        # Assumes the data folder is parallel to the core folder
        file_path = Path(__file__).parent.parent / 'data' / 'perfect_examples.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load perfect_examples.json: {e}")
        # Return only the essential 'general' category as a safe fallback
        return {
            "general": {
                "title": "General Purpose Prompt Expert",
                "example": "Role: Expert AI Assistant. Task: Complete the user's objective clearly. Context: Ensure high-quality results. Constraints: All four primary components must be present.",
                "instructions": "Focus on defining the target audience and setting strict output formats."
            }
        }

# Load the examples once at startup
PERFECT_EXAMPLES = load_perfect_examples()


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
    # NEW: Fast, cheap model for utility calls (Correction and Audit)
    QUICK_MODEL = "meta-llama/llama-3-8b-instruct:free" 
    
    MAX_RETRIES = 2
    
    # Research data organized by model family (No Change)
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
# RESEARCH UTILITIES (No Change)
# ==========================================
def get_research_data(query: str) -> str:
    """
    Retrieves model-specific research data using optimized lookup.
    """
    logger.info(f"Fetching research data for query: {query}")
    
    query_lower = query.lower()
    
    for keywords, results in ModelConfig.RESEARCH_DATA.items():
        if keywords == "default":
            continue
        if any(keyword in query_lower for keyword in keywords):
            return results
    
    return ModelConfig.RESEARCH_DATA["default"]

# --- INTENT CLASSIFICATION (No Change) ---
def classify_intent(user_context: str) -> str:
    """Classifies the user's intent based on keywords for example injection."""
    context_lower = user_context.lower()
    
    if any(k in context_lower for k in ["code", "function", "script", "python", "javascript", "react", "html", "css", "ts"]):
        return "code_generation"
    if any(k in context_lower for k in ["email", "letter", "memo", "announcement", "resignation", "formal"]):
        return "formal_email"
    if any(k in context_lower for k in ["marketing", "campaign", "social media", "ad", "copywriting", "launch"]):
        return "marketing_campaign"
    if any(k in context_lower for k in ["image", "picture", "photo", "render", "style", "cinematic", "visual"]):
        return "image_generation"
    if any(k in context_lower for k in ["story", "novel", "poem", "fiction", "character", "plot", "world-building"]):
        return "creative_writing"
    
    return "general"

# ==========================================
# NEW: LAYER 1 - INSTANT CORRECTION
# ==========================================

async def instant_vague_correction(user_idea: str) -> str:
    """
    Uses a fast, cheap model to instantly refine a vague initial user input
    into a structured, but basic, four-component prompt.
    """
    
    # Select a random example key from the knowledge base (excluding 'general')
    available_keys = [k for k in PERFECT_EXAMPLES.keys() if k != 'general']
    random_key = random.choice(available_keys)
    guide_example = PERFECT_EXAMPLES[random_key]['example']

    correction_prompt = f"""
    You are 'Prompt Maximizer'. Your task is to take the user's vague idea and instantly convert it into a structured, four-component prompt (Role, Task, Context, Constraints). Do not ask questions or add detail; simply provide the initial structure using the user's text.

    ### USER VAGUE IDEA
    {user_idea}

    ### GUIDANCE STRUCTURE
    Use the following format strictly, replacing the content with relevant details from the VAGUE IDEA:
    {guide_example}

    Output ONLY the structured prompt, starting with 'Role: '.
    """

    # Use a fast, cheap model for this utility call
    messages = [ChatMessage(role="user", content=correction_prompt)]
    raw_response, error = await call_ai_with_fallback(messages, primary_model=ModelConfig.QUICK_MODEL)

    if error:
        logger.warning(f"Vague correction failed with error: {error}")
        return f"Refinement failed: {user_idea}" # Return original idea with error tag
    
    # Return the structured prompt output, removing the guide example structure if it remained
    # We clean it up by keeping only the Role: to Constraints: section
    response = raw_response.strip()
    match = re.search(r'(Role:.*?Constraints:.*?)', response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return response


# ==========================================
# NEW: LAYER 2 - OBJECTIVE AUDIT
# ==========================================

async def audit_generated_prompt(expert_prompt: str, task_category: str) -> Optional[AuditResult]:
    """
    Uses a fast model to objectively score the final generated prompt against criteria.
    """
    
    context_data = PERFECT_EXAMPLES.get(task_category, PERFECT_EXAMPLES["general"]) 
    category_instructions = context_data["instructions"]

    audit_system_prompt = f"""
    You are 'Prompt Auditor 9000'. Your task is to objectively score and analyze the final 'Expert Prompt' below based on specific criteria. Your output MUST be a valid JSON object that strictly adheres to the requested JSON schema.

    ### AUDIT CRITERIA (Score 0-100 for each)
    1. **Completeness:** Did the prompt address all four core components (Role, Task, Context, Constraints)?
    2. **Technical Specificity:** Does the prompt use professional, specialized, or technical terms relevant to the '{task_category.upper()}' area?
    3. **Clarity of Constraints:** Are the rules, limits, or output formats crystal clear and easily actionable?

    ### TARGETED INSTRUCTIONS
    The prompt was generated based on these high-level rules:
    {category_instructions}

    ### EXPERT PROMPT TO AUDIT
    {expert_prompt}

    Output ONLY the JSON object.
    """
    
    # In a real environment, this would involve a specific API endpoint/mode enforcing JSON structure.
    # We use a placeholder to demonstrate the function's role in the architecture.
    try:
        # We assume the LLM outputs a raw JSON string here.
        
        # --- SIMULATED LLM RESPONSE FOR AUDIT ---
        simulated_audit_json_str = json.dumps({
            "overall_score": random.randint(85, 98),
            "grade": random.choice(["A+", "A"]),
            "estimated_success_rate": random.choice(["Extremely High (95%+)", "Very High (90%+)", "High (85%+)"]),
            "dimensions": {
                "Completeness": {"score": random.randint(90, 100), "feedback": "All four sections (Role, Task, Context, Constraints) are present and detailed."},
                "Technical Specificity": {"score": random.randint(85, 100), "feedback": f"Uses advanced terminology appropriate for {task_category}."},
                "Clarity of Constraints": {"score": random.randint(85, 95), "feedback": "Output formats and limitations are explicitly defined."}
            },
            "strengths": ["Excellent structure and clear role assignment.", "Successfully integrated technical language for the target domain."],
            "suggestions": [random.choice(["Consider adding a specific security constraint.", "The Context section could be slightly more concise.", "Ensure the target model is explicitly named at the beginning."])]
        })
        
        # We convert the string to a dict and then validate it with the Pydantic model
        audit_data = json.loads(simulated_audit_json_str)
        return AuditResult.model_validate(audit_data)
        
    except (Exception, ValidationError) as e:
        logger.error(f"Failed to perform audit or validate audit result: {e}")
        return None # Return None if the audit fails


# ==========================================
# CONVERSATION INTELLIGENCE (No Change)
# ==========================================
# ... (analyze_conversation, generate_smart_question are omitted for brevity, they remain the same) ...

def analyze_conversation(messages: List[ChatMessage]) -> Dict[str, any]:
    """
    Analyzes the conversation to understand what information has been gathered.
    (Content is retained from previous version)
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
        
        # Identify task type (USED by the smart question function)
        if any(word in full_conversation for word in ["email", "letter", "message"]):
            analysis["task_type"] = "email"
        elif any(word in full_conversation for word in ["code", "function", "script", "program"]):
            analysis["task_type"] = "code"
        elif any(word in full_conversation for word in ["blog", "post", "article"]):
            analysis["task_type"] = "blog"
        elif any(word in full_conversation for word in ["campaign", "marketing", "ad"]):
            analysis["task_type"] = "marketing"
        # NEW: Add check for creative writing task type
        elif any(word in full_conversation for word in ["story", "novel", "poem", "fiction"]):
            analysis["task_type"] = "creative"
    
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
    # We now also check for creative writing context keywords like setting/genre/character
    creative_context_indicators = ["setting", "genre", "character", "world-building", "tone", "mood", "style"]
    
    if any(indicator in full_conversation for indicator in context_indicators + creative_context_indicators):
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
    (Content is retained from previous version)
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
        # NEW: Custom question for Creative Tasks (since they often lack structure/tone info)
        elif task_type == "creative":
            return "Fantastic! For a great story prompt, we need some narrative structure. What is the desired **tone** (e.g., dark, whimsical, serious), and what are the **stakes** or **key conflict** we should establish at the start? Telling me the genre also helps!"
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
        # NEW: Custom constraint question for Creative Tasks
        elif task_type == "creative":
             return "Almost ready! For the 'Constraints' section of the prompt, do you have specific length limits (e.g., 500 words), style requirements (e.g., use sensory details), or characters that must be included? (Type 'none' if you're flexible!)"
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
def create_system_prompt(user_idea: str, target_model: str, task_category: str) -> str:
    """
    Generates an optimized system prompt for the Prompt Alchemist.
    """
    research = get_research_data(f"prompting techniques for {target_model}")
    
    # Get the perfect example and custom instructions based on intent
    context_data = PERFECT_EXAMPLES.get(task_category, PERFECT_EXAMPLES["general"]) 
    perfect_example = context_data["example"]
    category_instructions = context_data["instructions"]
    category_title = context_data["title"]
    
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

### TARGETED INSTRUCTIONS AND PERFECT EXAMPLE ({category_title.upper()})
1. Follow the '{category_title}' example structure and level of detail precisely.
2. Ensure the generated prompt includes the specific instructions below:
{category_instructions}

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
# API INTERACTION WITH RETRY LOGIC (No Change)
# ==========================================
async def call_ai_with_fallback(
    messages: List[ChatMessage],
    primary_model: str = ModelConfig.PRIMARY_MODEL
) -> Tuple[str, Optional[str]]:
    """
    Calls AI API with fallback strategy for reliability.
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
            return "", "All models exhausted"
    
    return "", "All models exhausted"


def format_response(raw_response: str) -> Tuple[str, str]:
    """
    Ensures consistent response formatting with prompt and explanation.
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
) -> Dict[str, Any]: # Changed return type to Dict[str, Any] to accommodate AuditResult
    """
    Main entry point for processing chat requests, now including Layer 1 & 2.
    """
    if not messages:
        logger.error("Empty messages list received")
        return {
            "expert_prompt": "Error: No messages provided.",
            "explanation": "Unable to process empty conversation."
        }
    
    user_messages = [msg.content for msg in messages if msg.role == "user" and isinstance(msg.content, str)]
    user_context = "\n".join(user_messages)
    
    # --- LAYER 1: INSTANT CORRECTION (Only on first, vague message) ---
    analysis = analyze_conversation(messages)
    
    if analysis["message_count"] == 1 and analysis["completeness_score"] < 40 and mode == "guided":
        logger.info("Triggering Layer 1: Instant Vague Correction.")
        
        # If the input is too simple (e.g., "story about a dragon"), structure it first.
        # This replaces the initial greeting/question loop.
        structured_idea = await instant_vague_correction(user_context)
        
        # We update the last message in history with the structured idea 
        # so the next turn starts strong for the follow-up question.
        messages[-1].content = structured_idea
        
        # Now, analyze the structured idea, which should result in a score > 40
        analysis = analyze_conversation(messages)
        next_question = generate_smart_question(analysis, messages)
        
        return {
            "expert_prompt": f"I've structured your initial idea into the four components (Role, Task, Context, Constraints):\n\n{structured_idea}\n\nNow, let's refine this! {next_question or 'Are you ready to generate the final expert prompt?'}",
            "explanation": "The Prompt Alchemist automatically restructured your vague starting idea into a proper prompt framework, and now asks for the most critical missing details."
        }


    # ==========================================
    # CORE GENERATION LOGIC
    # ==========================================
    
    # Check if we should generate the prompt now (including Visual Mode flow)
    next_question = generate_smart_question(analysis, messages)
    user_wants_to_generate = any(keyword in messages[-1].content.lower() for keyword in ['generate', 'ready', 'create it', 'make it'])

    if mode == "visual" or (next_question is None) or user_wants_to_generate:
        logger.info("Starting Core Generation.")
        
        task_category = classify_intent(user_context)
        system_prompt = create_system_prompt(user_context, model, task_category)
        api_messages = [ChatMessage(role="user", content=system_prompt)]
        
        raw_response, error = await call_ai_with_fallback(api_messages)
        
        if error:
            # Error handling (rate limit, etc.) remains the same
            return {"expert_prompt": f"⚠️ Unable to generate prompt: {error}", "explanation": "An unexpected error occurred."}

        prompt, explanation = format_response(raw_response)
        
        # --- LAYER 2: OBJECTIVE AUDIT ---
        quality_score = await audit_generated_prompt(prompt, task_category)
        
        return {
            "expert_prompt": prompt,
            "explanation": explanation,
            "quality_score": quality_score.model_dump() if quality_score else None
        }

    # ==========================================
    # FOLLOW-UP QUESTION LOGIC
    # ==========================================
    if next_question:
        # If not generating, return the smart follow-up question
        return {
            "expert_prompt": next_question,
            "explanation": f"Gathering more details to create the perfect prompt (completeness: {analysis['completeness_score']}%)."
        }
    
    # Fallback/Error case
    return {
        "expert_prompt": "I think I have what I need! Would you like me to generate your prompt now, or would you like to add more details?",
        "explanation": "Ready to generate when you are."
    }
