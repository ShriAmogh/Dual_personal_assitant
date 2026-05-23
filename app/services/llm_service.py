import os
import time
import asyncio
import httpx
import logging
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("app.llm")

# Configure the new official Google genai Client
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

class LlmService:
    def __init__(self):
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.frontier_model_name = os.environ.get("FRONTIER_MODEL", "models/gemini-flash-lite-latest")
        
        # Initialize Google GenAI client if key is available
        if GEMINI_KEY:
            logger.info("Initializing google-genai Client with GEMINI_API_KEY from environment.")
            self.client = genai.Client(api_key=GEMINI_KEY)
        else:
            logger.warning("GEMINI_API_KEY is not defined in environment. Frontier model integrations will be restricted.")
            self.client = None

    async def get_ollama_models(self) -> List[str]:
        """Fetches the list of pulled models from local Ollama instance."""
        logger.info(f"Querying local Ollama endpoint: {self.ollama_base_url}/api/tags")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    logger.info(f"Ollama connected successfully. Models found: {models}")
                    return models
                logger.warning(f"Ollama returned non-200 status code: {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Unable to connect to local Ollama daemon: {e}")
            return []

    async def generate_frontier(self, active_messages: List[Dict[str, str]], summary: str = "") -> Tuple[str, float]:
        """Calls the Frontier model (Gemini) with full error handling and latency tracking."""
        if not GEMINI_KEY or not self.client:
            logger.error("Attempted to call Frontier assistant, but GEMINI_API_KEY is missing.")
            return "Error: GEMINI_API_KEY is not configured or is invalid. Please verify your .env file.", 0.0

        # Construct System Instruction incorporating summary if present
        system_instruction = "You are a helpful, professional personal assistant."
        if summary:
            logger.info("Frontier request prepending historical conversation summary context.")
            system_instruction += f"\n\n[Background summary of the conversation so far]:\n{summary}"

        # Format messages for the google-genai SDK
        formatted_contents = []
        for msg in active_messages:
            role = "model" if msg["role"] == "assistant" else "user"
            formatted_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )

        logger.info(
            f"Invoking Frontier LLM ({self.frontier_model_name}) "
            f"with {len(formatted_contents)} active history messages."
        )
        
        start_time = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
            
            # Configure with modern system instruction
            config = types.GenerateContentConfig(
                system_instruction=system_instruction
            )
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.frontier_model_name,
                    contents=formatted_contents,
                    config=config
                )
            )
            
            latency = time.perf_counter() - start_time
            result_text = response.text.strip()
            logger.info(
                f"Frontier LLM replied successfully. "
                f"Latency: {latency:.3f}s. Response size: {len(result_text)} chars."
            )
            return result_text, latency
        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.error(f"Frontier LLM execution failed after {latency:.3f}s with error: {e}")
            return f"Error calling Gemini: {str(e)}", latency

    async def generate_oss(self, active_messages: List[Dict[str, str]], summary: str = "", model_name: str = "phi3:mini") -> Tuple[str, float]:
        """Calls the local Ollama instance with full error handling and latency tracking."""
        # Setup conversation history for Ollama
        messages_payload = []
        
        # Prepend system prompt containing history summary if available
        system_instruction = "You are a helpful, professional personal assistant."
        if summary:
            logger.info(f"Ollama request prepending historical conversation summary context.")
            system_instruction += f"\n\n[Background summary of the conversation so far]:\n{summary}"
        
        messages_payload.append({"role": "system", "content": system_instruction})
        
        # Add active sliding window messages
        for msg in active_messages:
            messages_payload.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        logger.info(
            f"Invoking local OSS Model ({model_name}) via Ollama "
            f"with {len(messages_payload) - 1} active messages."
        )

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages_payload,
                        "stream": False
                    }
                )
                
                latency = time.perf_counter() - start_time
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "").strip()
                    logger.info(
                        f"OSS model ({model_name}) replied successfully. "
                        f"Latency: {latency:.3f}s. Response size: {len(content)} chars."
                    )
                    return content, latency
                else:
                    logger.error(
                        f"Ollama returned HTTP error code {response.status_code} "
                        f"after {latency:.3f}s. Response: {response.text}"
                    )
                    return f"Error calling Ollama (HTTP {response.status_code}): {response.text}", latency
        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.error(f"Ollama execution failed after {latency:.3f}s with error: {e}")
            return f"Error calling Ollama: {str(e)}. Please check if your local Ollama server is running.", latency

    async def generate_summary(self, older_messages: List[Dict[str, str]], current_summary: str = "") -> str:
        """Asynchronously triggers Gemini to generate an updated concise summary of older history."""
        if not GEMINI_KEY or not self.client:
            logger.warning("Attempted to trigger summarization, but Frontier client is unconfigured.")
            return ""

        logger.info(
            f"Triggering asynchronous historical summarization layer via Gemini Flash Lite. "
            f"Summarizing {len(older_messages)} older turns."
        )

        prompt = "You are a conversational summarization expert.\n"
        if current_summary:
            prompt += f"Here is the existing summary of the earlier conversation:\n{current_summary}\n\n"
        
        prompt += "Please update the summary by incorporating the following subsequent turns in a concise paragraph:\n"
        for msg in older_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role_label}: {msg['content']}\n"
            
        prompt += "\nProvide ONLY the updated, concise, high-level summary of the conversation so far. Do not add conversational intro, outro, or explanation."

        try:
            loop = asyncio.get_event_loop()
            start_time = time.perf_counter()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.frontier_model_name,
                    contents=prompt
                )
            )
            updated = response.text.strip()
            logger.info(
                f"Memory summarization task completed in {time.perf_counter() - start_time:.3f}s. "
                f"Updated summary length: {len(updated)} chars."
            )
            return updated
        except Exception as e:
            logger.error(f"Failed to generate memory summary: {e}")
            return current_summary
