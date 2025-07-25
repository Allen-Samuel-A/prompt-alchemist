# backend/services/openrouter_client.py

import os
import httpx
from dotenv import load_dotenv
from typing import List
from pathlib import Path
from backend.models.chat_models import ChatMessage

# --- Correctly load the .env file ---
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Define our app's identity ---
YOUR_SITE_URL = "http://127.0.0.1:5500"
YOUR_APP_TITLE = "Prompt Alchemist"

async def get_ai_response(messages: List[ChatMessage], model: str) -> str:
    """
    Sends a request to the OpenRouter API and gets a response.
    """
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set. Please check server configuration."

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    # These two headers are required by OpenRouter
                    "HTTP-Referer": YOUR_SITE_URL, 
                    "X-Title": YOUR_APP_TITLE,      
                },
                json={
                    "model": model,
                    "messages": [msg.model_dump() for msg in messages],
                },
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.text}")
            return f"An API error occurred: {e.response.status_code}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred while contacting the AI."