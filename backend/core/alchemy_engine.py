# backend/core/alchemy_engine.py

from typing import List, Dict, Optional
from models.chat_models import ChatMessage
from services.openrouter_client import get_ai_response

# ==========================================
# CONFIGURATION
# ==========================================
# You can change this model for higher reliability:
#    "openai/gpt-4o-mini" or "anthropic/claude-3.5-sonnet"
PROMPT_GENERATOR_MODEL = "alibaba/tongyi-deepresearch-30b-a3b:free"


# ==========================================
# SIMULATED RESEARCH FETCH
# ==========================================
def perform_web_search(query: str) -> str:
    """
    Simulates fetching research results about prompt-engineering
    based on the chosen model.
    """
    print(f"Performing simulated web search for: {query}")

    if "gpt" in query.lower() or "openai" in query.lower():
        return """
- Research Result 1: For GPT models, be explicit — place key instructions first.
- Research Result 2: Use clear separators (###) to organize information.
- Research Result 3: Encouraging 'step-by-step reasoning' improves accuracy.
"""
    elif "claude" in query.lower() or "anthropic" in query.lower():
        return """
- Research Result 1: Claude performs best with XML-like tags.
- Research Result 2: Positive instructions work better than prohibitions.
- Research Result 3: Pre-filling output format increases adherence.
"""
    elif "gemini" in query.lower() or "google" in query.lower():
        return """
- Research Result 1: Gemini benefits from persona-based prompts.
- Research Result 2: More context yields richer, detailed results.
- Research Result 3: Breaking tasks into substeps improves coherence.
"""
    elif "llama" in query.lower() or "meta" in query.lower():
        return """
- Research Result 1: Llama models excel with structured markdown layouts.
- Research Result 2: Examples and constraints enhance reliability.
- Research Result 3: Explicit formatting guidance reduces drift.
"""
    else:
        return """
- Research Result 1: Provide role, task, and constraints distinctly.
- Research Result 2: Include background context for relevance.
- Research Result 3: Define a clear expected output format.
"""


# ==========================================
# SYSTEM PROMPT CREATION
# ==========================================
def create_system_prompt(user_idea: str, target_model: str) -> str:
    """
    Creates a consistent, explicit system prompt that instructs
    the AI to output a fully structured and formatted expert prompt.
    """
    search_query = f"latest prompting techniques for {target_model}"
    latest_research = perform_web_search(search_query)

    system_prompt = f"""
You are 'Prompt Alchemist' — a world-class AI that designs perfectly structured prompts
optimized for language models like {target_model}.

---

### OBJECTIVE
Transform the user's idea into a **professionally structured, research-driven prompt**
that strictly follows the format below.

---

### REQUIRED OUTPUT FORMAT

### Prompt
**Role:** [Define the AI’s persona or area of expertise]
**Task:** [Describe the exact goal or action clearly]
**Context:** [Provide relevant background, environment, or audience details]
**Constraints:** [List rules, limitations, or style restrictions]

**Output Format:**
```markdown
### Solution Matrix
| Solution | Feasibility | Risk | Efficiency |
|-----------|-------------|------|-------------|

### Recommended Path
[2–3 paragraph justification with supporting evidence]
EXPLANATION
After the prompt, include a short "EXPLANATION" section
explaining how you used the latest research findings to craft the prompt.
Mention at least one specific research insight.

RESEARCH FOR YOUR REFERENCE
<research> {latest_research} </research>
USER INPUT
<idea> {user_idea} </idea>
RULES
Always begin with “### Prompt”

Always end with “---EXPLANATION---”

Do NOT include JSON, code blocks, or extra commentary.

Never omit any of the sections.

Keep explanations under 4 sentences.
"""
    return system_prompt

# ==========================================
# CORE LOGIC — PROCESS CHAT REQUEST
# ==========================================
async def process_chat_request(messages: List[ChatMessage], model: str, mode: str) -> Dict[str, str]:
    """
    Processes user interactions and generates final expert prompts.
    Supports both Guided and Visual Builder modes.
    """
    if not messages:
        return {"expert_prompt": "Error: No user message found.", "explanation": ""}

    # ---------------------------
    # GUIDED MODE: step-by-step
    # ---------------------------
    if mode == "guided":
        last_assistant_message: Optional[str] = None
        for i in range(len(messages) - 2, -1, -1):
            if messages[i].role == "assistant":
                if isinstance(messages[i].content, dict):
                    last_assistant_message = messages[i].content.get("expert_prompt", "")
                else:
                    last_assistant_message = messages[i].content
                break

        # Conversation flow logic
        if last_assistant_message is None or "start over" in last_assistant_message:
            return {
                "expert_prompt": "Hello! I'm the Prompt Alchemist. Let's get started. What's your main goal or idea?",
                "explanation": "This begins the guided prompt creation process.",
            }
        elif "goal or idea" in last_assistant_message:
            return {
                "expert_prompt": "Got it. What role should the AI take on? (e.g., 'marketing expert', 'data scientist')",
                "explanation": "Roles help tailor the AI’s persona for precision.",
            }
        elif "role should the AI take on" in last_assistant_message:
            return {
                "expert_prompt": "Perfect. What’s the main task or objective you want the AI to accomplish?",
                "explanation": "A well-defined task ensures targeted prompt generation.",
            }
        elif "main task" in last_assistant_message:
            return {
                "expert_prompt": "Good. Can you share any relevant context or background details?",
                "explanation": "Context adds clarity and relevance to the final prompt.",
            }
        elif "context" in last_assistant_message:
            return {
                "expert_prompt": "Almost done. Are there any specific constraints or formatting requirements?",
                "explanation": "Constraints refine the AI’s creative boundaries.",
            }
        else:
            # Collect all user responses
            user_answers = [msg.content for msg in messages if msg.role == "user"]
            if len(user_answers) >= 5:
                # Assuming the last 5 user messages are: idea, role, task, context, constraints
                idea, role, task, context, constraints = user_answers[-5:]
                assembled_idea = f"Role: {role}\nTask: {task}\nContext: {context}\nConstraints: {constraints}"
                system_prompt = create_system_prompt(assembled_idea, model)
                api_messages = [ChatMessage(role="user", content=system_prompt)]

                try:
                    raw_response = await get_ai_response(messages=api_messages, model=PROMPT_GENERATOR_MODEL)
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error calling model: {error_msg}")
                    if "429" in error_msg:
                        return {
                            "expert_prompt": "⚠️ Rate limit reached. Please wait and retry shortly.",
                            "explanation": "Free-tier congestion detected; please retry.",
                        }
                    return {
                        "expert_prompt": f"⚠️ Error generating prompt: {error_msg}",
                        "explanation": "Unexpected API or network error occurred.",
                    }

                # Enforce consistent formatting
                if not raw_response.strip().startswith("### Prompt"):
                    raw_response = "### Prompt\n" + raw_response.strip()
                if "---EXPLANATION---" not in raw_response:
                    raw_response += "\n\n---EXPLANATION---\nAutomatically formatted for consistency."

                if "---EXPLANATION---" in raw_response:
                    parts = raw_response.split("---EXPLANATION---", 1)
                    # Removed the "Great! Here's the final structured prompt:\n\n" prefix for a cleaner final output
                    return {
                        "expert_prompt": parts[0].strip(),
                        "explanation": parts[1].strip(),
                    }

                return {
                    "expert_prompt": raw_response,
                    "explanation": "Final prompt generated successfully.",
                }

            # If conversation logic fails
            return {
                "expert_prompt": "Something went wrong. Let's start again. What's your main goal?",
                "explanation": "Conversation restarted to ensure accuracy.",
            }

    # ---------------------------
    # VISUAL BUILDER MODE
    # ---------------------------
    else:
        last_user_message = ""
        if isinstance(messages[-1].content, str):
            last_user_message = messages[-1].content

        system_prompt = create_system_prompt(last_user_message, model)
        api_messages = [ChatMessage(role="user", content=system_prompt)]

        try:
            raw_response = await get_ai_response(messages=api_messages, model=PROMPT_GENERATOR_MODEL)
        except Exception as e:
            error_msg = str(e)
            print(f"Error calling model: {error_msg}")
            if "429" in error_msg:
                return {
                    "expert_prompt": "⚠️ Rate limit reached. Please try again later.",
                    "explanation": "Free-tier API overload detected.",
                }
            return {
                "expert_prompt": f"⚠️ Error generating prompt: {error_msg}",
                "explanation": "Unexpected backend or network error occurred.",
            }

        # Enforce structure if missing
        if not raw_response.strip().startswith("### Prompt"):
            raw_response = "### Prompt\n" + raw_response.strip()
        if "---EXPLANATION---" not in raw_response:
            raw_response += "\n\n---EXPLANATION---\nAutomatically formatted for consistency."

        expert_prompt, explanation = raw_response, "No explanation provided."
        if "---EXPLANATION---" in raw_response:
            parts = raw_response.split("---EXPLANATION---", 1)
            expert_prompt = parts[0].strip()
            explanation = parts[1].strip()

        return {"expert_prompt": expert_prompt, "explanation": explanation}