# backend/core/alchemy_engine.py

from typing import List, Dict, Optional, Tuple
from models.chat_models import ChatMessage
from services.openrouter_client import get_ai_response
import logging

# Import new modules
from core.prompt_enricher import PromptEnricher
from core.expert_templates import ExpertTemplates
from core.prompt_quality_scorer import PromptQualityScorer

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
# ENHANCED PROMPT ENGINEERING
# ==========================================
def create_ultra_enhanced_system_prompt(
    user_idea: str, 
    target_model: str,
    use_enricher: bool = True
) -> str:
    """
    Creates an extremely powerful system prompt that generates expert-level output.
    Now includes automatic enrichment and example-based learning.
    """
    research = get_research_data(f"prompting techniques for {target_model}")
    
    # Parse user input to extract components
    components = parse_user_input(user_idea)
    
    # Enrich components if enabled
    if use_enricher:
        enriched = PromptEnricher.enrich_full_prompt(
            role=components["role"],
            task=components["task"],
            context=components["context"],
            constraints=components["constraints"]
        )
    else:
        enriched = components
    
    # Get relevant example prompts for few-shot learning
    examples = get_example_prompts(components["task"])
    
    prompt = f"""You are 'Prompt Alchemist', the world's leading expert in prompt engineering with 15+ years 
of experience crafting prompts that achieve 95%+ success rates across all major AI models.

### YOUR MISSION - CRITICAL
The user has provided input that needs to be transformed into a PROFESSIONAL, EXPERT-LEVEL prompt 
that is 5-10x MORE DETAILED and EFFECTIVE than what they provided.

DO NOT simply reformat their input. You MUST ADD MASSIVE VALUE by:
✓ Expanding vague descriptions into specific, actionable details
✓ Adding professional credentials and expertise to roles
✓ Including concrete examples, metrics, and success criteria
✓ Providing comprehensive context about audience and use case
✓ Detailing constraints with quality standards and guidelines

### LEARN FROM THESE EXPERT EXAMPLES

Example 1 - Expert-Level Prompt:
{examples[0]}

Example 2 - Another Excellence Standard:
{examples[1]}

Notice how these prompts include:
- Specific credentials and years of experience
- Detailed, measurable deliverables
- Rich audience psychology and context
- Comprehensive, specific constraints
- Professional language and structure

### CRITICAL OUTPUT REQUIREMENTS
You MUST output EXACTLY TWO sections with NOTHING ELSE:

Section 1: The Enhanced Prompt (starts with "### Prompt")
Section 2: The Explanation (starts with "---EXPLANATION---")

❌ DO NOT include:
- Solution matrices or tables
- Recommended paths or strategies  
- Code blocks or JSON
- Multiple prompt versions
- Any text before "### Prompt"

### PROMPT STRUCTURE (MANDATORY)
Every prompt MUST include these four components with SUBSTANTIAL DETAIL:

**Role:** [Expand with: years of experience, specific expertise areas, credentials, specializations]
Example: "Expert Content Strategist with 12+ years optimizing content for search engines..."

**Task:** [Make specific with: exact deliverables, structure, success criteria, key elements]
Example: "Create a comprehensive 1,800-word blog post that ranks for 'project management' by..."

**Context:** [Add: target audience details, psychology, use case, environment, goals]
Example: "For B2B SaaS marketing managers aged 30-45 who struggle with team alignment..."

**Constraints:** [Include: quality standards, format requirements, tone, what to avoid, metrics]
Example: "Word count: 1,500-2,000. Tone: Conversational yet authoritative. Include 5-7 H2 subheadings..."

### USER INPUT TO TRANSFORM (Make this 5-10x better!)
{user_idea}

### ENRICHED CONTEXT (Use this to inform your enhancement)
Role: {enriched["role"]}
Task: {enriched["task"]}
Context: {enriched["context"]}
Constraints: {enriched["constraints"]}

### RESEARCH-BACKED GUIDELINES FOR {target_model.upper()}
{research}

### OUTPUT INSTRUCTIONS
1. Start IMMEDIATELY with "### Prompt" (no preamble)
2. Make each section 3-5x more detailed than user input
3. Add specific examples, numbers, and measurable criteria
4. Use professional language with industry terminology
5. Add "---EXPLANATION---" after the prompt
6. Write 2-4 sentences explaining your research-backed enhancements
7. STOP after explanation

Generate the expert-level prompt now:"""
    
    return prompt


def parse_user_input(user_idea: str) -> Dict[str, str]:
    """Parse user input to extract role, task, context, constraints"""
    import re
    
    components = {
        "role": "",
        "task": "",
        "context": "",
        "constraints": ""
    }
    
    # Try to parse structured input
    role_match = re.search(r'Role:\s*(.+?)(?=Task:|Context:|Constraint:|$)', user_idea, re.IGNORECASE | re.DOTALL)
    task_match = re.search(r'Task:\s*(.+?)(?=Role:|Context:|Constraint:|$)', user_idea, re.IGNORECASE | re.DOTALL)
    context_match = re.search(r'Context:\s*(.+?)(?=Role:|Task:|Constraint:|$)', user_idea, re.IGNORECASE | re.DOTALL)
    constraint_match = re.search(r'Constraint[s]?:\s*(.+?)(?=Role:|Task:|Context:|$)', user_idea, re.IGNORECASE | re.DOTALL)
    
    if role_match:
        components["role"] = role_match.group(1).strip()
    if task_match:
        components["task"] = task_match.group(1).strip()
    if context_match:
        components["context"] = context_match.group(1).strip()
    if constraint_match:
        components["constraints"] = constraint_match.group(1).strip()
    
    # If no structured parsing worked, treat entire input as task
    if not any(components.values()):
        components["task"] = user_idea
    
    return components


def get_example_prompts(task_description: str) -> List[str]:
    """Get 2 relevant example prompts based on task type"""
    # Detect task type
    task_lower = task_description.lower()
    
    if any(word in task_lower for word in ["blog", "article", "write", "content"]):
        template_id = "seo_blog_post"
    elif any(word in task_lower for word in ["code", "function", "program", "develop"]):
        template_id = "production_code"
    elif any(word in task_lower for word in ["analyze", "data", "research"]):
        template_id = "data_analysis"
    elif any(word in task_lower for word in ["email", "message"]):
        template_id = "professional_email"
    else:
        template_id = "seo_blog_post"  # Default
    
    # Get template
    template = ExpertTemplates.get_template(template_id)
    
    if template:
        example1 = f"""**Role:** {template['role']}
**Task:** {template['task']}
**Context:** {template['context']}
**Constraints:** {template['constraints']}"""
        
        # Create a second variation
        example2 = create_variation_example(template)
    else:
        example1 = "**Role:** Expert with 10+ years experience...\n**Task:** Specific deliverable with metrics..."
        example2 = "**Role:** Senior professional specializing in...\n**Task:** Create comprehensive output that..."
    
    return [example1, example2]


def create_variation_example(template: Dict) -> str:
    """Create a variation of a template for few-shot learning"""
    return f"""**Role:** Senior professional with extensive expertise similar to the above
**Task:** {template['task'][:200]}... [detailed continuation]
**Context:** {template['context'][:150]}... [expanded details]
**Constraints:** {template['constraints'][:200]}... [comprehensive guidelines]"""


# ==========================================
# CONVERSATION FLOW MANAGEMENT
# ==========================================
class GuidedFlowManager:
    """Manages the step-by-step guided conversation flow"""
    
    FLOW_STEPS = [
        {
            "trigger": None,
            "prompt": "Hello! I'm the Prompt Alchemist 🪄\n\nI'll help you create an expert-level prompt in 5 quick steps.\n\n**Step 1 of 5:** What would you like to create?\n\nExamples:\n• Write a blog post about AI\n• Generate Python code for data analysis\n• Create a marketing email\n• Analyze customer feedback data",
            "explanation": "Starting guided prompt creation with clear examples."
        },
        {
            "trigger": "goal or idea",
            "prompt": "Great! 🎯\n\n**Step 2 of 5:** What expertise should the AI have?\n\nExamples:\n• Senior Software Engineer\n• Digital Marketing Specialist  \n• Data Analyst\n• Content Strategist\n\n(Or type 'skip' for auto-suggestion)",
            "explanation": "Role definition ensures the AI adopts the right expertise."
        },
        {
            "trigger": "role should the AI",
            "prompt": "Perfect! 📝\n\n**Step 3 of 5:** Describe the task in detail.\n\nBe specific:\n• What exactly should be created?\n• What are the key requirements?\n• What makes this successful?\n\n(The clearer you are, the better the result!)",
            "explanation": "Detailed task definition is crucial for quality output."
        },
        {
            "trigger": "main task",
            "prompt": "Excellent! 🎨\n\n**Step 4 of 5:** Provide context:\n\n• Who is the audience?\n• What's the purpose/goal?\n• Any background information?\n• Where will this be used?\n\n(Or type 'skip' to continue)",
            "explanation": "Context helps the AI understand the bigger picture."
        },
        {
            "trigger": "context",
            "prompt": "Almost done! ⚙️\n\n**Step 5 of 5:** Any constraints or requirements?\n\nExamples:\n• Length: 800-1000 words\n• Tone: Professional and friendly\n• Format: Use bullet points\n• Avoid: Technical jargon\n\n(Type 'none' if no specific constraints)",
            "explanation": "Constraints shape the boundaries and quality standards."
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
            
            if "429" in error_msg:
                return "", "rate_limit"
            
            if attempt < len(models_to_try) - 1:
                continue
            
            return "", error_msg
    
    return "", "All models exhausted"


def format_response(raw_response: str) -> Tuple[str, str]:
    """
    Ensures consistent response formatting with prompt and explanation.
    Cleans up unwanted content.
    """
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
        
        # Remove unwanted sections
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
        
        # Clean explanation
        explanation_lines = explanation_part.split('\n\n')
        if len(explanation_lines) > 1:
            explanation_part = explanation_lines[0]
        
        return prompt_part, explanation_part
    
    return raw_response.strip(), "Prompt generated based on best practices and research."


# ==========================================
# MAIN PROCESSING LOGIC
# ==========================================
async def process_chat_request(
    messages: List[ChatMessage],
    model: str,
    mode: str
) -> Dict[str, any]:
    """
    Main entry point - now returns quality score along with prompt
    """
    if not messages:
        logger.error("Empty messages list received")
        return {
            "expert_prompt": "Error: No messages provided.",
            "explanation": "Unable to process empty conversation.",
            "quality_score": None
        }
    
    # GUIDED MODE
    if mode == "guided":
        last_msg = GuidedFlowManager.get_last_assistant_message(messages)
        current_step = GuidedFlowManager.get_current_step(last_msg)
        
        if current_step:
            return {
                "expert_prompt": current_step["prompt"],
                "explanation": current_step["explanation"],
                "quality_score": None
            }
        
        # Generate final prompt
        user_answers = [msg.content for msg in messages if msg.role == "user"]
        
        if len(user_answers) >= 5:
            assembled_idea = f"""Goal: {user_answers[-5]}
Role: {user_answers[-4]}
Task: {user_answers[-3]}
Context: {user_answers[-2]}
Constraints: {user_answers[-1]}"""
            
            system_prompt = create_ultra_enhanced_system_prompt(assembled_idea, model)
            api_messages = [ChatMessage(role="user", content=system_prompt)]
            
            raw_response, error = await call_ai_with_fallback(api_messages)
            
            if error == "rate_limit":
                return {
                    "expert_prompt": "⚠️ High traffic detected. Please wait 30-60 seconds and try again.",
                    "explanation": "Rate limit encountered.",
                    "quality_score": None
                }
            
            if error:
                return {
                    "expert_prompt": f"⚠️ Unable to generate prompt: {error}",
                    "explanation": "An unexpected error occurred.",
                    "quality_score": None
                }
            
            prompt, explanation = format_response(raw_response)
            
            # Score the prompt quality
            quality_score = PromptQualityScorer.score_prompt(prompt)
            
            return {
                "expert_prompt": prompt,
                "explanation": explanation,
                "quality_score": quality_score
            }
        
        return {
            "expert_prompt": "I need more information. Let's start over. What's your main goal?",
            "explanation": "Restarting guided flow.",
            "quality_score": None
        }
    
    # VISUAL BUILDER MODE
    else:
        last_user_message = messages[-1].content
        if not isinstance(last_user_message, str):
            last_user_message = str(last_user_message)
        
        system_prompt = create_ultra_enhanced_system_prompt(last_user_message, model)
        api_messages = [ChatMessage(role="user", content=system_prompt)]
        
        logger.info(f"Visual Builder - Generating enhanced prompt")
        
        raw_response, error = await call_ai_with_fallback(api_messages)
        
        if error == "rate_limit":
            return {
                "expert_prompt": "⚠️ Service at capacity. Please wait and retry.",
                "explanation": "High traffic volume.",
                "quality_score": None
            }
        
        if error:
            logger.error(f"Visual Builder error: {error}")
            return {
                "expert_prompt": f"⚠️ Generation failed: {error}",
                "explanation": "Please try again.",
                "quality_score": None
            }
        
        logger.info(f"Received response: {len(raw_response)} characters")
        
        prompt, explanation = format_response(raw_response)
        
        # Validate enhancement happened
        if prompt.strip() == last_user_message.strip():
            logger.warning("Response unchanged - retrying")
            # One retry with emphasis
            retry_prompt = f"{system_prompt}\n\nCRITICAL: You MUST enhance and expand, not just reformat!"
            retry_messages = [ChatMessage(role="user", content=retry_prompt)]
            retry_response, retry_error = await call_ai_with_fallback(retry_messages)
            
            if not retry_error:
                prompt, explanation = format_response(retry_response)
        
        # Score the prompt
        quality_score = PromptQualityScorer.score_prompt(prompt)
        
        return {
            "expert_prompt": prompt,
            "explanation": explanation,
            "quality_score": quality_score
        }