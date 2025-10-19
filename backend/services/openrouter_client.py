# backend/services/openrouter_client.py

import os
import httpx
from dotenv import load_dotenv
from typing import List
from pathlib import Path
from models.chat_models import ChatMessage


# --- Correctly load the .env file ---
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Define our app's identity ---
YOUR_SITE_URL = "https://whattoprompt.com"
YOUR_APP_TITLE = "Prompt Alchemist"

async def get_ai_response(messages: List[ChatMessage], model: str) -> str:
    """
    Sends a request to the OpenRouter API and gets a response.
    """
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY is not set. Please check server configuration.")

    async with httpx.AsyncClient() as client:
        try:
            print(f"Calling OpenRouter with model: {model}")
            response = await client.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": YOUR_SITE_URL, 
                    "X-Title": YOUR_APP_TITLE,      
                },
                json={
                    "model": model,
                    "messages": [msg.model_dump() for msg in messages],
                },
                timeout=30,
            )
            
            print(f"OpenRouter response status: {response.status_code}")
            
            if response.status_code != 200:
                error_body = response.text
                print(f"OpenRouter error body: {error_body}")
                raise Exception(f"OpenRouter returned {response.status_code}: {error_body}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            error_details = e.response.text
            print(f"HTTPStatusError: {error_details}")
            raise Exception(f"OpenRouter API error {e.response.status_code}: {error_details}")
        except Exception as e:
            print(f"Exception in get_ai_response: {str(e)}")
            raise