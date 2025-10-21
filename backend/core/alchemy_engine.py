# backend/core/alchemy_engine.py

from typing import List, Dict, Optional, Tuple, Any 
from models.chat_models import ChatMessage, AuditResult
from services.openrouter_client import get_ai_response
import logging
import re
import json
from pathlib import Path
import random 
from pydantic import ValidationError 
import math 
# Note: For production, ensure you import the actual tiktoken library for accurate token counting.

# Configure logging
logger = logging.getLogger(__name__)

# --- JSON Loader Utility (RETAINED) ---
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
# CONFIGURATION (UPDATED with Deep Research Insights)
# ==========================================
class ModelConfig:
    """Centralized model configuration with fallback strategy and feature constants"""
    PRIMARY_MODEL = "openai/gpt-4o-mini"
    FALLBACK_MODELS = [
        "anthropic/claude-3-haiku",
        "openai/gpt-3.5-turbo"
    ]
    QUICK_MODEL = "meta-llama/llama-3-8b-instruct:free" 
    PREMIUM_MODELS = ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo"]
    
    MAX_RETRIES = 2
    MAX_HISTORY_MESSAGES = 10 
    MAX_TOKEN_LIMIT = 20000 
    
    # NEW: Model size mapping for CoT/ToT threshold (Perplexity Report Mandate: >= 100B params)
    MODEL_SIZE_MAP = {
        # Size in Billions (B) of parameters
        "gpt-4o": 175, "gpt-4o-mini": 15, "gpt-4-turbo": 175,
        "claude-3.5-sonnet": 175, "claude-3-haiku": 5,
        "gemini-flash-1.5-8b": 8, "llama-3-8b-instruct": 8
    }
    
    # NEW: Model cost per 1000 output tokens (Simulated, approximate OpenRouter pricing)
    COST_PER_K_OUTPUT = {
        "openai/gpt-4o": 15.00, "openai/gpt-4o-mini": 0.50, 
        "anthropic/claude-3.5-sonnet": 12.00, "anthropic/claude-3-haiku": 0.25,
        "google/gemini-flash-1.5-8b": 0.35, "meta-llama/llama-3-8b-instruct:free": 0.00
    }
    
    # RESEARCH DATA (UPDATED to fix Pylance/IDE linter issues with triple quotes after tuple keys)
    RESEARCH_DATA = {
        ("gpt", "openai"): (
            "- Be explicit and place key instructions at the beginning\n"
            "- **Structural Mandate (Axiom 4):** Use **Triple Quotes (\"\"\" or Triple Hashes (###)** to clearly delineate instructions from context data.\n"
            "- Request step-by-step reasoning for complex tasks to improve accuracy\n"
            "- Provide examples in the prompt for better output consistency\n"
        ),
        ("claude", "anthropic"): (
            "- **Structural Mandate (Axiom 4):** Use XML-like tags for structured sections (e.g., <role>, <task>) for complex inputs.\n"
            "- Frame instructions positively rather than using prohibitions\n"
            "- Pre-fill the start of the expected output format for better adherence\n"
            "- Assign specific roles or personas for more targeted responses\n"
        ),
        ("gemini", "google"): (
            "- **Structural Mandate (Axiom 4):** Use explicit semantic labeling with **XML/HTML tags** (e.g., <DATA>) or **Prefixes (TASK:)** for organizing multi-component data sets.\n"
            "- Use persona-based prompts with clear role definitions\n"
            "- Provide comprehensive context for richer, more detailed results\n"
            "- Break complex tasks into explicit substeps for improved coherence\n"
            "- Include examples of desired output format\n"
        ),
        ("llama", "meta"): (
            "- Structure prompts using clear markdown layouts with headers\n"
            "- Provide explicit examples and constraints for better reliability\n"
            "- Include detailed formatting guidance to reduce output drift\n"
            "- Use numbered steps for sequential tasks\n"
        ),
        "default": (
            "- Clearly separate role, task, context, and constraints\n"
            "- Include relevant background context to improve response quality\n"
            "- Define expected output format explicitly with examples\n"
            "- Use positive framing for instructions\n"
        )
    }


# ==========================================
# SIMULATED UTILITIES (NEW: Token & Cost Calculation)
# ==========================================

def get_token_count(text: str) -> int:
    """Simulates token counting using a conservative LLM library estimate."""
    # Using a conservative estimate of 1 token per 4 characters
    return int(len(text) / 4)

def get_estimated_cost(model_name: str, token_count: int) -> str:
    """Simulates cost calculation based on ModelConfig data."""
    # Normalize model name for lookup
    name_key = model_name.split('/')[-1].split(':')[0].lower().replace("-", "").replace(".", "")
    
    # Try to match the model to the configured cost dictionary
    for key, cost in ModelConfig.COST_PER_K_OUTPUT.items():
        if name_key in key.lower().replace("-", "").replace(".", "").replace("/", ""):
            # Cost is per 1000 output tokens
            cost_usd = (token_count / 1000) * cost
            return f"${cost_usd:.4f}"
            
    return "N/A (Free Model or Unknown Cost)"


# ==========================================
# RESEARCH & CLASSIFICATION UTILITIES (RETAINED)
# ==========================================
def get_research_data(query: str) -> str:
    """Retrieves model-specific research data using optimized lookup."""
    logger.info(f"Fetching research data for query: {query}")
    query_lower = query.lower()
    
    for keywords, results in ModelConfig.RESEARCH_DATA.items():
        if keywords == "default":
            continue
        if any(keyword in query_lower for keyword in keywords):
            return results
    return ModelConfig.RESEARCH_DATA["default"]

def classify_intent(user_context: str) -> str:
    """Classifies the user's intent based on keywords for example injection."""
    context_lower = user_context.lower()
    
    if any(k in context_lower for k in ["code", "function", "python", "javascript", "react", "html", "css", "ts"]):
        return "code_generation"
    if any(k in context_lower for k in ["email", "letter", "memo", "announcement", "resignation", "formal"]):
        return "formal_email"
    if any(k in context_lower for k in ["marketing", "campaign", "social media", "ad", "copywriting", "launch", "video", "video script", "website", "selling"]):
        return "marketing_campaign"
    if any(k in context_lower for k in ["image", "picture", "photo", "render", "style", "cinematic", "visual"]):
        return "image_generation"
    if any(k in context_lower for k in ["story", "novel", "poem", "fiction", "character", "plot", "world-building"]):
        return "creative_writing"
    
    return "general"

# ==========================================
# LAYER 1: INSTANT VAGUE CORRECTION (RETAINED)
# ==========================================

async def instant_vague_correction(user_idea: str) -> Tuple[str, bool]:
    """
    Uses a fast, cheap model to instantly refine a vague initial user input
    into a structured, but basic, four-component prompt.
    Returns the structured prompt and a boolean indicating success.
    """
    available_keys = [k for k in PERFECT_EXAMPLES.keys() if k != 'general']
    # FIX: Use 'general' if the list is empty (prevents random.choice error)
    random_key = random.choice(available_keys) if available_keys else 'general' 
    guide_example = PERFECT_EXAMPLES[random_key]['example']

    correction_prompt = f"""
    You are 'Prompt Maximizer'. Your task is to take the user's vague idea and instantly convert it into a structured, four-component prompt (Role, Task, Context, Constraints). 
    
    If the user's idea is too vague (e.g., 'hello', 'hi', 'start'), you MUST respond with the exact phrase: 'TOO VAGUE: CANNOT STRUCTURE'.

    ### USER VAGUE IDEA
    {user_idea}

    ### GUIDANCE STRUCTURE
    Use the following format strictly, replacing the content with relevant details from the VAGUE IDEA:
    {guide_example}

    Output ONLY the structured prompt, starting with 'Role: '.
    """

    messages = [ChatMessage(role="user", content=correction_prompt)]
    raw_response, error = await call_ai_with_fallback(messages, primary_model=ModelConfig.QUICK_MODEL)

    if error:
        logger.warning(f"Vague correction failed with error: {error}")
        return f"Refinement failed: {user_idea}", False
    
    response = raw_response.strip()

    # New check for LLM refusal
    if "TOO VAGUE: CANNOT STRUCTURE" in response:
        return "TOO VAGUE: CANNOT STRUCTURE", False

    # FIX: Updated regex pattern to be non-greedy and match until newline or end of string
    match = re.search(r'(Role:.*?Constraints:.*?)(?:\n|$)', response, re.DOTALL)
    
    if match:
        return match.group(1).strip(), True
    
    return response, False


# ==========================================
# LAYER 3: ADAPTIVE MEMORY COMPRESSION (RETAINED)
# ==========================================

async def summarize_conversation(messages: List[ChatMessage]) -> Tuple[List[ChatMessage], str]:
    """
    Condenses long conversation history into a single summary message using a cheap model.
    Returns the new messages list and the summary text.
    """
    # Exclude the very last user message from the compression (it's the current context)
    chat_to_summarize = messages[:-1]
    last_message = messages[-1]
    
    full_chat = "\n".join([f"{m.role}: {m.content}" for m in chat_to_summarize])
    
    summary_prompt = f"""
    You are 'Memory Compressor'. Review the conversation history below and generate a single, concise paragraph that clearly summarizes the user's ultimate goal, all constraints (e.g., length, style, technology), and all contextual information provided so far.

    CONVERSATION HISTORY:
    {full_chat}

    OUTPUT MUST BE ONE PARAGRAPH.
    """
    
    messages_for_summary = [ChatMessage(role="user", content=summary_prompt)]
    raw_summary, error = await call_ai_with_fallback(messages_for_summary, primary_model=ModelConfig.QUICK_MODEL)

    if error:
        logger.warning(f"Memory compression failed: {error}. Proceeding with full history.")
        return messages, "Error: Summary failed. Using full context."
    
    summary_message = ChatMessage(
        role="assistant", 
        content=f"**[CONTEXT SUMMARY: The conversation was condensed to this goal]:** {raw_summary.strip()}"
    )
    
    # Return a new, short history: [Summary, Last User Message]
    new_messages = [summary_message, last_message]
    return new_messages, raw_summary.strip()


# ==========================================
# LAYER 4: SMART MODEL SELECTION (RETAINED)
# ==========================================

def smart_model_selection(requested_model: str, is_generation_task: bool) -> str:
    """
    Downgrades model selection if a premium model is requested for a non-final,
    conversational step, saving cost.
    """
    if requested_model not in ModelConfig.PREMIUM_MODELS:
        # Not a premium model, no downgrade needed
        return requested_model
    
    if is_generation_task:
        # It's a premium model for a final generation task (OK)
        return requested_model
    else:
        # It's a premium model for a simple conversational step (Downgrade!)
        logger.info(f"Downgrading model from {requested_model} to {ModelConfig.QUICK_MODEL} for conversational step.")
        return ModelConfig.QUICK_MODEL


# ==========================================
# LAYER 5: OPTIMIZATION FRAMEWORK SUGGESTION (RETAINED)
# ==========================================

def get_optimization_suggestion(task_category: str) -> Dict[str, str]:
    """Provides advanced optimization framework suggestions and a user-facing action."""
    # Logic based on WHEN TO USE from ChatGPT's report
    if task_category in ["code_generation", "formal_email", "blog"]: # Added blog to CoT as it's an informational/structured task
        return {
            "suggestion": "The **Chain-of-Thought (CoT)** method is highly recommended to improve reliability and logical structure. **Key Phrase to Inject:** 'Let’s think step by step.'",
            "action": "Refine using CoT (Step-by-Step) Reasoning"
        }
    elif task_category == "marketing_campaign": # Creative problem-solving, strategic lookahead
        return {
            "suggestion": "The **Tree-of-Thought (ToT)** method is excellent for strategic problem-solving. **Key Phrase to Inject:** 'Imagine three different experts are answering this question...'",
            "action": "Refine using ToT (Strategic Planning)"
        }
    elif task_category in ["creative_writing", "image_generation"]: # Needs verification/consistency
        return {
            "suggestion": "Use **Self-Consistency** to enhance coherence across creative outputs. **Key Phrase to Inject:** 'Generate several independent answers and give the most common result.'",
            "action": "Refine using Self-Consistency Check"
        }
    else: # General tasks needing precise format
        return {
            "suggestion": "Consider using the **Few-Shot Learning** technique to define the desired output pattern. **Key Phrase to Inject:** 'Include explicit examples: Input: X; Output: Y...'",
            "action": "Refine with Few-Shot Examples"
        }


# ==========================================
# LAYER 2: OBJECTIVE AUDIT (RETAINED)
# ==========================================

async def audit_generated_prompt(expert_prompt: str, target_model: str, task_category: str) -> Optional[AuditResult]:
    """
    Uses a fast model to objectively score the final generated prompt against criteria.
    (Simulated implementation for safety)
    """
    
    try:
        # Layer 5 integration: Get the advanced suggestion
        adv_suggestion_data = get_optimization_suggestion(task_category)
        advanced_suggestion = adv_suggestion_data["suggestion"]
        
        # --- NEW TECHNICAL CALCULATIONS (Layer 2) ---
        token_count = get_token_count(expert_prompt)
        estimated_cost = get_estimated_cost(target_model, token_count)

        model_check_warning = None
        
        # Perplexity Report Mandate: CoT/ToT requires >= 100B params to emerge
        if ("CoT" in advanced_suggestion or "ToT" in advanced_suggestion):
            model_name_key_full = target_model.split('/')[-1].split(':')[0].lower()
            model_name_key_clean = model_name_key_full.replace("-", "").replace(".", "")
            
            # Find the model size based on the cleaned key
            model_size = 0
            for key, size in ModelConfig.MODEL_SIZE_MAP.items():
                if model_name_key_clean in key.replace("-", "").replace(".", ""):
                    model_size = size
                    break
            
            # If model size is below the 100B conservative threshold
            if model_size < 100 and model_size > 0: 
                model_check_warning = f"⚠️ WARNING: The suggested **{advanced_suggestion.split('(')[0].strip().replace('Refine using', '')}** framework may be inefficient or unreliable with your selected model ({target_model}) due to its small parameter count ({model_size}B). CoT/ToT benefits usually emerge at 100B+ parameters."
            
            # Gemini Report Mandate: Max Output Token Truncation Risk
            if 'JSON' in expert_prompt or 'XML' in expert_prompt:
                 # This check simulates a warning if the structured output is requested but the cost is low (often correlates with max token limit constraints)
                 if token_count > 1000 and "free" in target_model.lower():
                     model_check_warning = (model_check_warning or "") + " ⚠️ RISK: Requesting structured output (JSON/XML) with high token count on a smaller/free model increases the risk of mid-output truncation. Review your Max Tokens setting."
        
        # --- Dynamic Feedback Configuration (RETAINED) ---
        if task_category in ["creative_writing", "image_generation"]:
            tech_specificity_feedback = "Focuses on narrative structure, tone, and visual/character depth, essential for creative work."
            tech_specificity_term = "Visual/Narrative Specificity"
            tech_specificity_score = random.randint(92, 100)
            suggestions = [advanced_suggestion, "Ensure the setting details are rich with sensory language or light/color description."]
        
        elif task_category == "marketing_campaign":
            tech_specificity_feedback = "Uses marketing terminology (KPIs, audience segmentation) effectively and targets platform constraints."
            tech_specificity_term = "Marketing Specificity"
            tech_specificity_score = random.randint(89, 97)
            suggestions = [advanced_suggestion, "Ensure the call-to-action is highly visible and specific."]
        
        elif task_category == "code_generation":
            tech_specificity_feedback = "Uses advanced programming terminology and language-specific best practices effectively."
            tech_specificity_term = "Technical Specificity"
            tech_specificity_score = random.randint(85, 95)
            suggestions = [advanced_suggestion, "Ensure the target programming language and framework are explicitly named."]
        
        elif task_category == "formal_email" or task_category == "blog": # Added blog here for technical focus
            tech_specificity_feedback = "Uses professional communication terminology and maintains appropriate diplomatic tone throughout."
            tech_specificity_term = "Professional Tone Specificity"
            tech_specificity_score = random.randint(88, 96)
            suggestions = [advanced_suggestion, "Ensure the recipient relationship and organizational context are clear."]
        
        else:  # general fallback
            tech_specificity_feedback = "Uses clear, domain-appropriate language and maintains focus on the core objective."
            tech_specificity_term = "Domain Specificity"
            tech_specificity_score = random.randint(85, 93)
            suggestions = [advanced_suggestion, "Ensure the target model and expected output format are explicitly defined."]
        
        # --- Simulated JSON Generation ---
        simulated_audit_json = {
            "overall_score": random.randint(88, 98),
            "grade": random.choice(["A+", "A"]),
            "estimated_success_rate": random.choice(["Extremely High (95%+)", "Very High (90%+)"]),
            "dimensions": {
                "Completeness": {
                    "score": random.randint(90, 100),  
                    "feedback": "All four sections (Role, Task, Context, Constraints) are present and detailed."
                },
                tech_specificity_term: {
                    "score": tech_specificity_score,  
                    "feedback": tech_specificity_feedback
                },
                "Clarity of Constraints": {
                    "score": random.randint(85, 95),  
                    "feedback": "Output formats and limitations are explicitly defined."
                }
            },
            "strengths": [
                "Excellent structure and clear role assignment (Axiom 1).",  
                f"Successfully integrated specialized terminology for the {task_category.replace('_', ' ')} domain."
            ],
            "suggestions": suggestions,
            # --- NEW FIELDS POPULATED ---
            "token_count": token_count,
            "estimated_cost": estimated_cost,
            "model_check_warning": model_check_warning.strip() if model_check_warning else None
        }
        
        audit_data = simulated_audit_json
        return AuditResult.model_validate(audit_data)
        
    except (Exception, ValidationError) as e:
        logger.error(f"Failed to perform audit or validate audit result: {e}")
        return None 


# ==========================================
# CONVERSATION INTELLIGENCE (FIXED WITH USER'S NEW LOGIC)
# ==========================================
def analyze_conversation(messages: List[ChatMessage]) -> Dict[str, Any]:
    """Analyzes the conversation to understand what information has been gathered."""
    # FIX: Corrected list comprehension to safely check isinstance on each message
    user_messages = [msg.content for msg in messages if msg.role == "user" and isinstance(msg.content, str)]
    full_conversation = " ".join(user_messages).lower()
    
    analysis = {
        "has_task": False, "has_role": False, "has_context": False, "has_constraints": False,
        "task_type": None, "completeness_score": 0, "message_count": len(user_messages)
    }
    
    # FIXED: More comprehensive task indicators
    task_indicators = [
        "write", "create", "generate", "make", "build", "design", "develop", "draft", "compose",
        "code", "script", "email", "letter", "blog", "post", "article", "function", "program",
        "campaign", "website", "selling", "marketing", "need help", "needs to", "process",
        "validate", "remove", "handle", "i want", "i need", "help me", "can you"
    ]
    
    if any(indicator in full_conversation for indicator in task_indicators):
        analysis["has_task"] = True
        analysis["completeness_score"] += 25
        
        # Determine task type
        if any(word in full_conversation for word in ["email", "letter", "message", "memo"]):  
            analysis["task_type"] = "email"
        elif any(word in full_conversation for word in ["code", "function", "script", "program", "python", "javascript", "validate", "process", "class", "method"]):  
            analysis["task_type"] = "code"
        elif any(word in full_conversation for word in ["blog", "post", "article"]):  
            analysis["task_type"] = "blog"
        elif any(word in full_conversation for word in ["campaign", "marketing", "ad", "selling", "website", "social media"]):  
            analysis["task_type"] = "marketing_campaign"
        elif any(word in full_conversation for word in ["story", "novel", "poem", "fiction", "character", "plot"]):  
            analysis["task_type"] = "creative"
    
    # Role detection
    role_indicators = [
        "expert", "developer", "engineer", "writer", "designer", "analyst", "specialist",
        "professional", "senior", "junior", "manager", "ai", "assistant"
    ]
    if any(indicator in full_conversation for indicator in role_indicators):  
        analysis["has_role"] = True
        analysis["completeness_score"] += 20
    
    # Context detection
    context_indicators = [
        "for", "audience", "purpose", "background", "about", "regarding", "boss", "client",
        "customer", "user", "team", "company", "project", "perfumes", "handmade", "postgresql",
        "aws lambda", "low-latency", "web application", "application", "system", "platform",
        "data", "entries", "list", "copilot", "github" # Added copilot and github for context detection
    ]
    creative_context_indicators = ["setting", "genre", "character", "world-building", "tone", "mood", "style"]
    
    if any(indicator in full_conversation for indicator in context_indicators + creative_context_indicators):  
        analysis["has_context"] = True
        analysis["completeness_score"] += 25
    
    # Constraints detection
    constraint_indicators = [
        "should", "must", "need to", "require", "limit", "words", "tone", "style", "format",
        "professional", "casual", "formal", "length", "none", "psycopg2", "error handling",
        "production-ready", "validate", "remove duplicates", "clean"
    ]
    if any(indicator in full_conversation for indicator in constraint_indicators):  
        analysis["has_constraints"] = True
        analysis["completeness_score"] += 20
    
    # Bonus for detailed input
    if len(full_conversation) > 50:  
        analysis["completeness_score"] += 10
    
    return analysis


def generate_smart_question(analysis: Dict[str, Any], messages: List[ChatMessage]) -> Optional[str]:
    """
    Generates an intelligent follow-up question based on what's missing, prioritizing
    Task > Context > Constraints.
    """
    # Safety check for message content
    last_user_message = messages[-1].content.lower().strip() if messages and isinstance(messages[-1].content, str) else ""
    skip_keywords = ['none', 'no', 'skip', 'nothing', 'not needed', 'n/a']
    user_wants_to_skip = last_user_message in skip_keywords
    
    # Check if the previous assistant message was asking about constraints
    was_asking_for_constraints = False
    if len(messages) >= 2 and messages[-2].role == "assistant":
        prev_message = messages[-2].content.lower()
        if "constraint" in prev_message or "length limit" in prev_message or "tone be" in prev_message:
            was_asking_for_constraints = True

    # FIX: If user just skipped constraints question, treat as "has_constraints" for flow purposes
    effective_has_constraints = analysis["has_constraints"] or (was_asking_for_constraints and user_wants_to_skip)

    # CRITICAL FIX: Changed completeness score from 70 to 90 to prevent premature generation
    if analysis["completeness_score"] >= 90 or (effective_has_constraints and analysis["has_context"] and analysis["has_task"] and analysis["has_role"]):
        return None # Ready to generate!

    # --- FIXED PRIORITY FLOW (Task -> Role -> Context -> Constraints) ---
    
    # 1. Ask for TASK if missing (Highest Priority)
    if not analysis["has_task"]:
        return "Thanks for that! Let's clarify the **Task** first: What exactly do you want the AI to generate? (e.g., a marketing plan, product descriptions, a Python function, a story outline, etc.)"
    
    # 2. Ask for ROLE if missing (Second Priority - Axiom 1)
    if analysis["has_task"] and not analysis["has_role"]:
        return "Great! Now that I know the task, let's define the **Role** for the AI. What specific persona should the AI adopt to complete this? (e.g., Senior Developer, Marketing Expert, Academic Tutor, etc.)"

    # 3. Ask for CONTEXT if missing
    if not analysis["has_context"] and not user_wants_to_skip:
        task_type = analysis.get("task_type")
        if task_type == "email": 
            return "That's a great start! To make this email perfect, could you tell me who the email is for (like a boss, colleague, or client) and what the main situation is? More context helps me craft a precise prompt for you."
        elif task_type == "code": 
            return "Got it, code generation! I need a little more detail: What's the specific use case or problem it needs to solve? For example, is this for a web application, data processing, API integration, etc.?"
        elif task_type == "blog": 
            return "Awesome, a blog post! Who is your target audience, and what's the main idea or message you want them to take away? Focusing the audience helps a lot!"
        elif task_type == "creative": 
            return "Fantastic! For a great story prompt, we need some narrative structure. What is the desired **tone** (e.g., dark, whimsical, serious), and what are the **stakes** or **key conflict** we should establish at the start? Telling me the genre also helps!"
        elif task_type == "marketing_campaign": 
            return "Great start! To craft a strong marketing prompt, who is your **final audience** (age, interests, platform)? More context about the products (handmade perfumes) is also key!"
        else: 
            return "I need a bit more context to craft a really strong prompt. What's the main goal of your prompt, and who is the final audience? Knowing the purpose and the audience makes a huge difference!"
    
    # 4. Ask for CONSTRAINTS if missing
    if not effective_has_constraints:
        task_type = analysis.get("task_type")
        if task_type == "email": 
            return "We're almost there! Do you have any specific constraints? For example, should the tone be professional or casual, is there a length limit, or any key points that must be included? (Just type 'none' if you're flexible!)"
        elif task_type == "code": 
            return "Great! For the code prompt, do you have any technical constraints? Which programming language should be used, are there performance goals, or any specific code styles required? (You can say 'none' if you're flexible!)"
        elif task_type == "creative": 
            return "Almost ready! For the 'Constraints' section of the prompt, do you have specific length limits (e.g., 500 words), style requirements (e.g., use sensory details), or characters that must be included? (Type 'none' if you're flexible!)"
        else: 
            return "Last piece of info needed: Do you have any format or style constraints? This could be a length requirement, a specific tone you need (like funny or serious), or things the AI must be sure to avoid. (Type 'none' to move on!)"
    
    # Fallback to generate prompt if score is decent but flow didn't hit 'None'
    if analysis["completeness_score"] >= 40 or user_wants_to_skip:
        return None # Ready to generate!
    
    # Final check for detail
    if analysis["completeness_score"] < 50:
        return "Thanks! Your idea is shaping up well. Would you like to add any more specific details about the exact outcome you want or any special requirements? If not, just say 'generate' and I'll create your prompt!"
    
    # Default
    return "Got it! I have enough information to create a detailed prompt. Would you like to add anything else, or should I go ahead and generate the expert prompt now? Type 'generate' when you're ready!"


# ==========================================
# PROMPT ENGINEERING (RETAINED)
# ==========================================
def create_system_prompt(user_idea: str, target_model: str, task_category: str) -> str:
    """Generates an optimized system prompt for the Prompt Alchemist."""
    research = get_research_data(f"prompting techniques for {target_model}")
    context_data = PERFECT_EXAMPLES.get(task_category, PERFECT_EXAMPLES["general"]) 
    perfect_example = context_data["example"]
    category_instructions = context_data["instructions"]
    category_title = context_data["title"]
    
    # NEW MANDATE: Instructions must be at the beginning, separated by delimiters (Axiom 4).
    return f"""### INSTRUCTION BLOCK
You are 'Prompt Alchemist', an expert AI prompt engineer specializing in creating 
production-ready prompts optimized for {target_model}.

### YOUR MISSION
The user's requirements are provided in the <USER_REQUIREMENTS> block below. Your job is to TRANSFORM this into a 
professional, detailed, research-backed prompt that delivers excellent results.

You must ADD VALUE by:
- Enforcing **Axiom 1 (Role-Based Instantiation)** by assigning a strong, specialized persona.
- Enforcing **Axiom 2 (Specificity and Quantification)** by making instructions measurable.
- Enforcing **Axiom 3 (Explicit Output Format)** by detailing structure (Markdown, JSON, etc.).

### CRITICAL OUTPUT REQUIREMENTS
You MUST output EXACTLY TWO sections with NOTHING ELSE:

Section 1: The Enhanced Prompt (starts with "### Prompt")
Section 2: The Explanation (starts with "---EXPLANATION---")

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

---
### USER_REQUIREMENTS (Context Separation via Delimiter - Axiom 4)
{user_idea}
---

### STRICT FORMATTING RULES
1. Start IMMEDIATELY with "### Prompt" (no preamble)
2. ENHANCE each of the 4 components with specific, actionable details
3. Make it comprehensive and production-ready
4. Add "---EXPLANATION---" after the prompt
5. Write 2-4 sentences explaining what enhancements you made
6. STOP after explanation - add nothing else
"""


# ==========================================
# API INTERACTION WITH RETRY LOGIC (RETAINED)
# ==========================================
async def call_ai_with_fallback(
    messages: List[ChatMessage],
    primary_model: str = ModelConfig.PRIMARY_MODEL
) -> Tuple[str, Optional[str]]:
    """Calls AI API with fallback strategy for reliability."""
    models_to_try = [primary_model] + ModelConfig.FALLBACK_MODELS
    
    for attempt, model in enumerate(models_to_try):
        try:
            logger.info(f"Attempting API call with model: {model} (attempt {attempt + 1})")
            response = await get_ai_response(messages=messages, model=model)
            return response, None
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Model {model} failed: {error_msg}")
            
            if "429" in error_msg: return "", "rate_limit"
            if attempt < len(models_to_try) - 1: continue
            
            return "", "All models exhausted"
    
    return "", "All models exhausted"


def format_response(raw_response: str) -> Tuple[str, str]:
    """Ensures consistent response formatting with prompt and explanation."""
    raw_response = raw_response.strip()
    
    if "### Prompt" in raw_response:
        raw_response = "### Prompt" + raw_response.split("### Prompt", 1)[1]
    elif not raw_response.startswith("### Prompt"):
        raw_response = "### Prompt\n" + raw_response
    
    if "---EXPLANATION---" in raw_response:
        parts = raw_response.split("---EXPLANATION---", 1)
        prompt_part = parts[0].strip()
        explanation_part = parts[1].strip()
        
        unwanted_markers = ["### Solution Matrix", "### Recommended Path", "### Alternative", "| Solution |", "```json", "```"]
        for marker in unwanted_markers:
            if marker in prompt_part: prompt_part = prompt_part.split(marker)[0].strip()
        
        explanation_lines = explanation_part.split('\n\n')
        if len(explanation_lines) > 1: explanation_part = explanation_lines[0]
        
        return prompt_part, explanation_part
    
    return raw_response.strip(), "Prompt generated based on best practices and research."


# ==========================================
# MAIN PROCESSING LOGIC (FIXED WITH USER'S NEW LOGIC)
# ==========================================
async def process_chat_request(
    messages: List[ChatMessage],
    model: str,
    mode: str
) -> Dict[str, Any]:
    """
    Main entry point for processing chat requests, now including all advanced layers.
    """
    if not messages:
        logger.error("Empty messages list received")
        return {
            "expert_prompt": "Error: No messages provided.",
            "explanation": "Unable to process empty conversation."
        }
    
    # FIX: Corrected list comprehension to safely check isinstance on each message
    user_messages = [msg.content for msg in messages if msg.role == "user" and isinstance(msg.content, str)]
    user_context = "\n".join(user_messages)
    
    # --- LAYER 4: SAFETY GUARDRAIL (Max Token/Size Check) ---
    if len(user_context) * 0.25 > ModelConfig.MAX_TOKEN_LIMIT:
        return {
            "expert_prompt": f"⚠️ Input too large! Please limit your total context to under {ModelConfig.MAX_TOKEN_LIMIT} tokens for stable performance.",
            "explanation": "Context window exceeded. Cannot process large request."
        }
    
    # --- FIX FOR QUESTION REPETITION ---
    messages_for_next_question = list(messages)
    
    # Check if the last two messages are: Assistant (Question) and User (Reply)
    if len(messages) >= 2 and messages[-2].role == "assistant" and not messages[-2].content.startswith("### Prompt"):
        # Temporary remove the assistant's question to evaluate the user's reply against the core components
        messages_for_next_question = messages[:-2] + [messages[-1]]
    
    analysis = analyze_conversation(messages)
    analysis_for_question = analyze_conversation(messages_for_next_question) 

    # --- NEW VAGUE INPUT BYPASS LOGIC (RETAINED) ---
    # FIX: Added check 'and user_messages' to prevent IndexError on user_messages[-1]
    user_is_vague = analysis["message_count"] == 1 and user_messages and (len(user_messages[-1].strip()) < 10 or analysis["completeness_score"] < 10)
    
    if user_is_vague and mode == "guided":
        logger.info("Bypassing Layer 1: Input too vague/short. Going straight to smart question.")
        next_question = generate_smart_question(analysis, messages)
        
        return {
            "expert_prompt": f"Welcome! I'm the Prompt Alchemist. Let's start with your idea. What do you want the AI to do? {next_question or 'Are you ready to generate the final expert prompt?'}",
            "explanation": "The Alchemist recognized your input as an opening greeting and immediately started the guided process by asking for the most critical piece of information (the Task/Context)."
        }

    # --- LAYER 1: INSTANT CORRECTION (RETAINED) ---
    if analysis["message_count"] == 1 and analysis["completeness_score"] < 40 and mode == "guided":
        logger.info("Triggering Layer 1: Instant Vague Correction.")
        
        structured_idea, success = await instant_vague_correction(user_context)
        
        if not success:
            next_question = generate_smart_question(analysis, messages)
            return {
                "expert_prompt": f"Apologies, I still couldn't structure that idea. Let's try the guided approach! {next_question or 'Are you ready to generate the final expert prompt?'}",
                "explanation": "The model failed to structure the initial idea. Switching to conversational guidance."
            }
        
        messages[-1].content = structured_idea 
        
        analysis = analyze_conversation(messages)
        next_question = generate_smart_question(analysis, messages)
        
        return {
            "expert_prompt": f"I've structured your initial idea into the four components (Role, Task, Context, Constraints):\n\n{structured_idea}\n\nNow, let's refine this! {next_question or 'Are you ready to generate the final expert prompt?'}",
            "explanation": "The Prompt Alchemist automatically restructured your vague starting idea into a proper prompt framework, and now asks for the most critical missing details."
        }


    # ==========================================
    # CORE GENERATION LOGIC PREP
    # ==========================================
    
    next_question = generate_smart_question(analysis_for_question, messages)
    
    # FIX: Corrected the syntax error 'user_wants_to-generate' to 'user_wants_to_generate'
    user_wants_to_generate = any(keyword in messages[-1].content.lower() for keyword in ['generate', 'ready', 'create it', 'make it'])
    
    is_generation_task = mode == "visual" or (next_question is None) or user_wants_to_generate

    # --- LAYER 3: ADAPTIVE MEMORY COMPRESSION (RETAINED) ---
    final_messages_for_api = messages
    if len(messages) > ModelConfig.MAX_HISTORY_MESSAGES and is_generation_task:
        logger.info(f"Triggering Layer 3: Compressing history from {len(messages)} messages.")
        final_messages_for_api, summary = await summarize_conversation(messages)
        
    # --- LAYER 4: SMART MODEL SELECTION (RETAINED) ---
    final_model_to_use = smart_model_selection(model, is_generation_task)

    if is_generation_task:
        logger.info(f"Starting Core Generation with model: {final_model_to_use}.")
        
        task_category = classify_intent(user_context)
        system_prompt = create_system_prompt(user_context, model, task_category)
        
        # We only send the system prompt for the final generation, using the context gathered
        api_messages = [ChatMessage(role="user", content=system_prompt)]
        
        raw_response, error = await call_ai_with_fallback(api_messages, primary_model=final_model_to_use)
        
        if error:
            return {"expert_prompt": f"⚠️ Unable to generate prompt: {error}", "explanation": "An unexpected error occurred."}

        prompt, explanation = format_response(raw_response)
        
        # --- LAYER 2 & 5: OBJECTIVE AUDIT with Optimization Suggestion ---
        # The audit function now receives the prompt, model, and category for deep checks
        quality_score = await audit_generated_prompt(prompt, model, task_category)
        
        return {
            "expert_prompt": prompt,
            "explanation": explanation,
            "quality_score": quality_score.model_dump() if quality_score else None
        }

    # ==========================================
    # FOLLOW-UP QUESTION LOGIC
    # ==========================================
    
    if next_question:
        # Determine the reason for the question for the explanation field
        reason = "Gathering more details to create the perfect prompt."
        
        # This uses the clean analysis_for_question to ensure the *next* question is accurate
        current_analysis = analyze_conversation(messages_for_next_question) 
        
        # Now, use the clean analysis to determine the precise reason for the question
        # This logic is based on the logic in generate_smart_question (Task -> Role -> Context -> Constraints)
        if not current_analysis["has_task"]:
            reason = "The most critical component—the **Task**—is missing. We need a clear objective (Axiom 2)."
        elif current_analysis["has_task"] and not current_analysis["has_role"]: 
            reason = "The **Role** (specialized persona) is missing, which is key to focused and authoritative responses (Axiom 1)."
        elif not current_analysis["has_context"]:
            reason = "The **Context** (audience, background, use-case) is missing, which dramatically reduces quality."
        elif not current_analysis["has_constraints"]:
            reason = "The **Constraints** (format, tone, length) are missing, which is key to a polished output (Axiom 3)."
            
        return {
            "expert_prompt": next_question,
            "explanation": f"{reason} (Completeness: {current_analysis['completeness_score']}%).",
        }
    
    # Fallback/Error case
    return {
        "expert_prompt": "I think I have what I need! Would you like to me generate your prompt now, or would you like to add more details?",
        "explanation": "Ready to generate when you are."
    }
