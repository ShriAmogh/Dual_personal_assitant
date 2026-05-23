from pydantic import BaseModel
from typing import List, Dict, Optional

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    model_type: str  # "frontier" or "oss"
    selected_model: Optional[str] = None  # Specific Ollama model name if OSS

class ChatResponse(BaseModel):
    response: str
    latency: float  # In seconds
    model_name: str
    session_id: str

class OllamaModelList(BaseModel):
    models: List[str]
