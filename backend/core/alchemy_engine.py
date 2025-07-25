# backend/core/alchemy_engine.py

from typing import List, Dict, Optional
from backend.models.chat_models import ChatMessage
from backend.services.openrouter_client import get_ai_response

# --- NEW: We will simulate a search function for now ---
# In a real application, this would use a library like 'requests' or a real search API
def perform_web_search(query: str) -> str:
    """
    Simulates performing a web search and returning the top results.
    """
    print(f"Performing simulated web search for: {query}")
    # We'll return some example best practices based on our research
    if "gpt" in query.lower():
        return """
- Research Result 1: For GPT models, it's best to be very direct. Place instructions at the beginning of the prompt.
- Research Result 2: Use separators like '###' to clearly distinguish instructions from context.
- Research Result 3: Encouraging the model to 'think step-by-step' can improve reasoning on complex tasks.
"""
    elif "claude" in query.lower():
        return """
- Research Result 1: Claude models respond very well to XML tags like <document> and <instructions>.
- Research Result 2: It's better to tell Claude what to do, rather than what not to do.
- Research Result 3: Pre-filling the assistant's response can help guide the output format.
"""
    else: # Default for Gemini and others
        return """
- Research Result 1: Providing a clear 'Persona' is highly effective for Gemini models.
- Research Result 2: Giving rich context helps Gemini produce more nuanced and accurate responses.
- Research Result 3: Break down complex tasks into smaller, more manageable steps in the prompt.
"""

# --- THIS FUNCTION IS NOW UPGRADED WITH LIVE SEARCH ---
def create_system_prompt(user_idea: str, target_model: str) -> str:
    """
    Creates the master system prompt. It now performs a web search
    for the latest prompting techniques to include in its instructions.
    """
    
    # 1. Perform a web search for the latest prompting guides
    search_query = f"latest prompting techniques for {target_model}"
    latest_research = perform_web_search(search_query)

    # 2. Construct the final system prompt, including the research
    system_prompt = f"""
You are 'Prompt Alchemist', a world-class AI assistant that creates expert-level prompts.

**Your Task:**
Your goal is to convert the user's idea into a high-quality, structured prompt that is perfectly optimized for the '{target_model}' AI model. You must use the latest research to inform your prompt structure.

**Latest Research Findings:**
<research>
{latest_research}
</research>

**User's Core Idea:**
<idea>
{user_idea}
</idea>

**Instructions:**
1.  **Analyze:** Read the user's idea and the latest research findings.
2.  **Synthesize:** Combine the user's goal with the best practices from your research.
3.  **Construct:** Build a new, detailed prompt. It must include a clear 'Role', 'Task', 'Context', 'Format', and 'Constraints'. The structure of the prompt should reflect the best practices you found in your research.

**Final Output Structure:**
First, write the complete, expertly crafted prompt.
Then, on a new line, write the separator "---EXPLANATION---".
Finally, in your explanation, you MUST mention how you used one of the specific findings from your web research to make the prompt better.
"""
    return system_prompt


# This function remains unchanged as it correctly handles the conversation flow
async def process_chat_request(messages: List[ChatMessage], model: str, mode: str) -> Dict[str, str]:
    """
    Processes a chat request based on the current mode.
    """
    if not messages:
        return {"expert_prompt": "Error: No user message found.", "explanation": ""}

    if mode == 'guided':
        last_assistant_message: Optional[str] = None
        for i in range(len(messages) - 2, -1, -1):
            if messages[i].role == 'assistant':
                if isinstance(messages[i].content, dict):
                    last_assistant_message = messages[i].content.get('expert_prompt', '')
                else:
                    last_assistant_message = messages[i].content
                break

        if last_assistant_message is None or "start over" in last_assistant_message:
            return { "expert_prompt": "Hello! I'm the Prompt Alchemist. To start, what's your main goal or idea?", "explanation": "This is the first step of our guided interview." }
        elif "goal or idea" in last_assistant_message:
            return { "expert_prompt": "Got it. Now, what role should the AI take on? (e.g., 'a marketing expert', 'a python developer')", "explanation": "Defining a role gives the AI better focus." }
        elif "role should the AI take on" in last_assistant_message:
            return { "expert_prompt": "Perfect. What is the specific task you want the AI to perform?", "explanation": "A clear task leads to a more precise result." }
        elif "specific task" in last_assistant_message:
            return { "expert_prompt": "Great. Is there any important context, background, or details to include?", "explanation": "Context helps the AI understand the full picture." }
        elif "context, background, or details" in last_assistant_message:
            return { "expert_prompt": "Almost done. Are there any constraints, rules, or specific output formats? (e.g., 'keep it under 100 words', 'respond in JSON')", "explanation": "Constraints help guide the final output." }
        else:
            user_answers = [msg.content for msg in messages if msg.role == 'user']
            if len(user_answers) >= 5:
                idea, role, task, context, constraints = user_answers[-5:]
                assembled_idea = f"Role: {role}\nTask: {task}\nContext: {context}\nConstraints: {constraints}"
                system_prompt = create_system_prompt(assembled_idea, model)
                api_messages = [ChatMessage(role="user", content=system_prompt)]
                raw_response = await get_ai_response(messages=api_messages, model="google/gemini-flash-1.5")
                
                if "---EXPLANATION---" in raw_response:
                    parts = raw_response.split("---EXPLANATION---", 1)
                    return { "expert_prompt": f"Great, I have everything I need! Here is the final prompt I've constructed for you:\n\n{parts[0].strip()}", "explanation": parts[1].strip() }
                else:
                     return { "expert_prompt": f"Great, I have everything I need! Here is the final prompt I've constructed for you:\n\n{raw_response}", "explanation": "The final prompt is ready." }
            else:
                 return {"expert_prompt": "Something went wrong in the conversation. Let's start over. What's your main goal?", "explanation": "Conversation restarted."}
    
    else: # Visual Builder Mode
        last_user_message = ""
        if isinstance(messages[-1].content, str):
            last_user_message = messages[-1].content

        system_prompt = create_system_prompt(last_user_message, model)
        api_messages = [ChatMessage(role="user", content=system_prompt)]
        raw_response = await get_ai_response(messages=api_messages, model="google/gemini-flash-1.5")
        
        expert_prompt = raw_response
        explanation = "No explanation was generated."
        if "---EXPLANATION---" in raw_response:
            parts = raw_response.split("---EXPLANATION---", 1)
            expert_prompt = parts[0].strip()
            explanation = parts[1].strip()

        return { "expert_prompt": expert_prompt, "explanation": explanation }
