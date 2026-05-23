import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.models.schemas import ChatRequest, ChatResponse, OllamaModelList
from app.services.llm_service import LlmService
from app.services.memory_service import MemoryService
from app.services.observability_service import ObservabilityService
from app.services.guardrail_service import GuardrailService
from app.services.tool_service import ToolService

load_dotenv()

# Configure standard console logging with descriptive pattern
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app.main")

logger.info("Initializing FastAPI Dual AI Personal Assistants application...")

app = FastAPI(
    title="Dual AI Personal Assistants Dashboard",
    description="Side-by-side Open Source and Frontier Assistant workspace with hybrid summarization memory.",
    version="1.0.0"
)

# Enable CORS for local testing/development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate services
llm_service           = LlmService()
memory_service        = MemoryService()
observability_service = ObservabilityService()
guardrail_service     = GuardrailService(llm_service=llm_service)
tool_service          = ToolService()

# Ensure directories exist
os.makedirs("app/static", exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")
logger.info("Static files mounted successfully under /static.")


@app.get("/")
async def serve_index():
    """Serves the main dashboard user interface."""
    index_path = "app/static/index.html"
    if not os.path.exists(index_path):
        logger.error(f"Failed to find frontend index file at: {index_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found. Please build the frontend first."
        )
    logger.info("Serving main dashboard index.html to browser client.")
    return FileResponse(index_path)


@app.get("/api/models/ollama", response_model=OllamaModelList)
async def list_ollama_models():
    """Returns all models currently pulled on the local Ollama instance."""
    logger.info("Received request for available local Ollama models.")
    models = await llm_service.get_ollama_models()
    return OllamaModelList(models=models)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint. Handles:
    1. Input guardrail check.
    2. Storing the new user message.
    3. Tool-use enrichment (time, calculator, web search).
    4. Hybrid summarization if conversation exceeds 10 turns.
    5. Generating LLM response (Gemini or Ollama) with latency measurement.
    6. Output guardrail check.
    7. Saving the assistant's reply.
    8. Recording the trace in ObservabilityService.
    """
    logger.info(
        f"[Session: {request.session_id}] New chat query. "
        f"Model: {request.model_type} (selected: {request.selected_model or 'None'}). "
        f"Message (excerpt): {request.message[:40]}..."
    )

    # ── 1. Input Guardrail ────────────────────────────────────────────────────
    guard_input = await guardrail_service.check_input(request.message)
    if not guard_input.safe:
        logger.warning(
            f"[Session: {request.session_id}] Input BLOCKED by guardrail. "
            f"Category: {guard_input.category}"
        )
        blocked_response = f"🛡️ Message blocked ({guard_input.category}): {guard_input.reason}"
        observability_service.record(
            session_id=request.session_id,
            model_type=request.model_type,
            model_name=request.model_type,
            prompt=request.message,
            response=blocked_response,
            latency=0.0,
            blocked_by_guardrail=True,
        )
        return ChatResponse(
            response=blocked_response,
            latency=0.0,
            model_name="Guardrail",
            session_id=request.session_id,
        )

    # ── 2. Store user message ─────────────────────────────────────────────────
    await memory_service.add_message(request.session_id, "user", request.message)

    # ── 3. Tool-use enrichment ────────────────────────────────────────────────
    tool_context = await tool_service.maybe_invoke(request.message)

    # ── 4. Hybrid summarization check ────────────────────────────────────────
    session = await memory_service.get_or_create_session(request.session_id)
    total_messages = len(session.messages)

    if total_messages > 10:
        logger.info(
            f"[Session: {request.session_id}] Message count ({total_messages}) exceeds hybrid limit of 10. "
            f"Checking older context for summarization."
        )
        older_messages, current_summary = await memory_service.get_older_messages_to_summarize(request.session_id)
        if older_messages:
            logger.info(f"[Session: {request.session_id}] Triggering background context summarizer.")
            updated_summary = await llm_service.generate_summary(older_messages, current_summary)
            await memory_service.update_summary(request.session_id, updated_summary)

    # ── 5. Retrieve context ───────────────────────────────────────────────────
    active_messages, summary = await memory_service.get_context(request.session_id)

    # Inject tool result as leading context if available
    if tool_context:
        logger.info(f"[Session: {request.session_id}] Injecting tool context into prompt.")
        active_messages = [{"role": "user", "content": f"[Tool result]: {tool_context}"}] + active_messages

    # ── 6. LLM invocation ────────────────────────────────────────────────────
    if request.model_type == "frontier":
        logger.info(f"[Session: {request.session_id}] Routing query to Frontier Model.")
        response_text, latency = await llm_service.generate_frontier(active_messages, summary)
        model_name = "Gemini Flash Lite"
    elif request.model_type == "oss":
        selected_model = request.selected_model or "phi3:mini"
        logger.info(f"[Session: {request.session_id}] Routing query to OSS Model ({selected_model}).")
        response_text, latency = await llm_service.generate_oss(active_messages, summary, selected_model)
        model_name = f"Ollama ({selected_model})"
    else:
        logger.error(f"[Session: {request.session_id}] Invalid model type requested: {request.model_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model_type. Must be 'frontier' or 'oss'."
        )

    # ── 7. Output Guardrail ───────────────────────────────────────────────────
    guard_output = await guardrail_service.check_output(response_text)
    if not guard_output.safe:
        logger.warning(f"[Session: {request.session_id}] Output BLOCKED by guardrail.")
        response_text = "⚠️ The model's response was flagged and blocked by the safety filter. Please rephrase your request."

    # ── 8. Store assistant reply ──────────────────────────────────────────────
    await memory_service.add_message(request.session_id, "assistant", response_text)

    # ── 9. Record observability trace ─────────────────────────────────────────
    observability_service.record(
        session_id=request.session_id,
        model_type=request.model_type,
        model_name=model_name,
        prompt=request.message,
        response=response_text,
        latency=latency,
        memory_size=len(active_messages),
        error=response_text.startswith("Error"),
    )

    logger.info(
        f"[Session: {request.session_id}] Responding. Latency: {latency:.3f}s. "
        f"Reply (excerpt): {response_text[:40]}..."
    )

    return ChatResponse(
        response=response_text,
        latency=round(latency, 3),
        model_name=model_name,
        session_id=request.session_id
    )


@app.post("/api/session/clear")
async def clear_session_endpoint(payload: dict):
    """Clears the conversational context for a given session ID."""
    session_id = payload.get("session_id")
    if not session_id:
        logger.warning("Clear session request received without session_id payload.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required"
        )
    logger.info(f"[Session: {session_id}] Received command to clear conversational context.")
    await memory_service.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} memory cleared successfully."}


@app.get("/api/observability/traces")
async def get_traces(session_id: str = None, model_type: str = None, limit: int = 50):
    """Returns recent LLM request traces, optionally filtered by session or model type."""
    traces = observability_service.get_traces(
        session_id=session_id,
        model_type=model_type,
        limit=limit,
    )
    return {"traces": traces, "count": len(traces)}


@app.get("/api/observability/stats")
async def get_stats():
    """Returns aggregate performance statistics for both models."""
    return observability_service.get_stats()


# Configure standard console logging with descriptive pattern
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app.main")

logger.info("Initializing FastAPI Dual AI Personal Assistants application...")

app = FastAPI(
    title="Dual AI Personal Assistants Dashboard",
    description="Side-by-side Open Source and Frontier Assistant workspace with hybrid summarization memory.",
    version="1.0.0"
)

# Enable CORS for local testing/development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate services
llm_service = LlmService()
memory_service = MemoryService()

# Ensure directories exist
os.makedirs("app/static", exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")
logger.info("Static files mounted successfully under /static.")

@app.get("/")
async def serve_index():
    """Serves the main dashboard user interface."""
    index_path = "app/static/index.html"
    if not os.path.exists(index_path):
        logger.error(f"Failed to find frontend index file at: {index_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found. Please build the frontend first."
        )
    logger.info("Serving main dashboard index.html to browser client.")
    return FileResponse(index_path)

@app.get("/api/models/ollama", response_model=OllamaModelList)
async def list_ollama_models():
    """Returns all models currently pulled on the local Ollama instance."""
    logger.info("Received request for available local Ollama models.")
    models = await llm_service.get_ollama_models()
    return OllamaModelList(models=models)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint. Handles:
    1. Storing the new user message.
    2. Dynamically updating conversational summaries if conversation exceeds 10 turns.
    3. Generating LLM response (Gemini or Ollama) with latency measurement.
    4. Saving the assistant's reply.
    """
    logger.info(
        f"[Session: {request.session_id}] New chat query. "
        f"Model: {request.model_type} (selected: {request.selected_model or 'None'}). "
        f"Message (excerpt): {request.message[:40]}..."
    )

    # 1. Store the new user message
    await memory_service.add_message(request.session_id, "user", request.message)
    
    # 2. Check and handle hybrid summarization
    session = await memory_service.get_or_create_session(request.session_id)
    total_messages = len(session.messages)
    
    # If history exceeds 10 messages, update summary of older messages (everything before last 5)
    if total_messages > 10:
        logger.info(
            f"[Session: {request.session_id}] Message count ({total_messages}) exceeds hybrid limit of 10. "
            f"Checking older context for summarization."
        )
        older_messages, current_summary = await memory_service.get_older_messages_to_summarize(request.session_id)
        if older_messages:
            logger.info(f"[Session: {request.session_id}] Triggering background context summarizer.")
            # Generate new summary using Gemini
            updated_summary = await llm_service.generate_summary(older_messages, current_summary)
            await memory_service.update_summary(request.session_id, updated_summary)
            
    # 3. Retrieve context (active last 5 messages + summary if applicable)
    active_messages, summary = await memory_service.get_context(request.session_id)
    
    # 4. Invoke the selected LLM and measure latency
    if request.model_type == "frontier":
        logger.info(f"[Session: {request.session_id}] Routing query to Frontier Model.")
        response_text, latency = await llm_service.generate_frontier(active_messages, summary)
        model_name = "Gemini Flash Lite"
    elif request.model_type == "oss":
        selected_model = request.selected_model or "phi3:mini"
        logger.info(f"[Session: {request.session_id}] Routing query to OSS Model ({selected_model}).")
        response_text, latency = await llm_service.generate_oss(active_messages, summary, selected_model)
        model_name = f"Ollama ({selected_model})"
    else:
        logger.error(f"[Session: {request.session_id}] Invalid model type requested: {request.model_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model_type. Must be 'frontier' or 'oss'."
        )
        
    # 5. Store the assistant's response in history
    await memory_service.add_message(request.session_id, "assistant", response_text)
    
    logger.info(
        f"[Session: {request.session_id}] Responding. Latency: {latency:.3f}s. "
        f"Reply (excerpt): {response_text[:40]}..."
    )
    
    return ChatResponse(
        response=response_text,
        latency=round(latency, 3),
        model_name=model_name,
        session_id=request.session_id
    )

@app.post("/api/session/clear")
async def clear_session_endpoint(payload: dict):
    """Clears the conversational context for a given session ID."""
    session_id = payload.get("session_id")
    if not session_id:
        logger.warning("Clear session request received without session_id payload.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required"
        )
    logger.info(f"[Session: {session_id}] Received command to clear conversational context.")
    await memory_service.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} memory cleared successfully."}
