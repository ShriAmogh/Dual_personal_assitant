import pytest
from fastapi.testclient import TestClient
from app.main import app, memory_service
from app.services.memory_service import MemoryService

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_memory():
    # Reset sessions before each test
    memory_service.sessions = {}

def test_ollama_models_endpoint():
    """Verify the local Ollama models list endpoint works gracefully."""
    response = client.get("/api/models/ollama")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)

@pytest.mark.anyio
async def test_memory_service_sliding_window():
    """Verify that only a maximum of 5 messages are returned in the active sliding window."""
    temp_memory = MemoryService()
    session_id = "test_window"
    
    # Add 8 messages (4 turns)
    for i in range(4):
        await temp_memory.add_message(session_id, "user", f"User Msg {i}")
        await temp_memory.add_message(session_id, "assistant", f"Assistant Msg {i}")
        
    active_messages, summary = await temp_memory.get_context(session_id)
    
    # Active window should have exactly the last 5 messages
    assert len(active_messages) == 5
    assert active_messages[0]["content"] == "Assistant Msg 1"
    assert active_messages[-1]["content"] == "Assistant Msg 3"
    
    # Total messages <= 10, so summary must be empty
    assert summary == ""

@pytest.mark.anyio
async def test_memory_service_summary_trigger():
    """Verify that older messages are correctly identified for summarization once limit exceeds 10."""
    temp_memory = MemoryService()
    session_id = "test_summary"
    
    # Add 12 messages (6 turns)
    for i in range(6):
        await temp_memory.add_message(session_id, "user", f"User Msg {i}")
        await temp_memory.add_message(session_id, "assistant", f"Assistant Msg {i}")
        
    active_messages, summary = await temp_memory.get_context(session_id)
    
    # Total messages is 12 (which is > 10)
    # The active window is the last 5 messages
    assert len(active_messages) == 5
    
    # Get older messages to summarize
    older_messages, current_summary = await temp_memory.get_older_messages_to_summarize(session_id)
    
    # Older messages should consist of everything except the last 5 messages (12 - 5 = 7 messages)
    assert len(older_messages) == 7
    assert older_messages[0]["content"] == "User Msg 0"
    assert older_messages[-1]["content"] == "User Msg 3"

def test_session_clear_endpoint():
    """Verify clearing session history via endpoint."""
    session_id = "test_clear"
    
    # Post some chat message
    response = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "Hello Assistant",
        "model_type": "frontier"
    })
    # Note: If Gemini key isn't configured, endpoint will still return response (containing error msg)
    assert response.status_code == 200
    
    # Verify session exists in memory
    assert session_id in memory_service.sessions
    
    # Clear the session
    clear_response = client.post("/api/session/clear", json={"session_id": session_id})
    assert clear_response.status_code == 200
    
    # Verify session is wiped
    assert session_id not in memory_service.sessions
