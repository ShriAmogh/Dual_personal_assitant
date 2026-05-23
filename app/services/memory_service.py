import asyncio
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("app.memory")

class SessionHistory:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []  # List of {"role": "user"|"assistant", "content": "..."}
        self.summary: str = ""                    # Running summary of older messages

class MemoryService:
    def __init__(self):
        self.sessions: Dict[str, SessionHistory] = {}
        self.lock = asyncio.Lock()  # Ensure thread/async-safety for session management
        logger.info("MemoryService initialized with in-memory thread-safe storage.")

    async def get_or_create_session(self, session_id: str) -> SessionHistory:
        async with self.lock:
            if session_id not in self.sessions:
                logger.info(f"[Session: {session_id}] Creating new conversation session history.")
                self.sessions[session_id] = SessionHistory()
            return self.sessions[session_id]

    async def add_message(self, session_id: str, role: str, content: str):
        session = await self.get_or_create_session(session_id)
        async with self.lock:
            session.messages.append({"role": role, "content": content})
            logger.info(
                f"[Session: {session_id}] Appended message (role: {role}). "
                f"Total history length: {len(session.messages)} messages."
            )

    async def get_context(self, session_id: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Returns:
            Tuple containing:
            - List of active messages:
              - If total messages <= 10, returns ALL messages in full.
              - If total messages > 10, returns only the last 5 messages.
            - Running summary of older messages (empty if total messages <= 10).
        """
        session = await self.get_or_create_session(session_id)
        
        async with self.lock:
            total_messages = len(session.messages)
            
            # Active sliding window is the last 5 messages
            active_messages = session.messages[-5:] if total_messages > 5 else session.messages
            
            logger.info(
                f"[Session: {session_id}] Retrieving context. "
                f"Active window: {len(active_messages)} messages (from total of {total_messages})."
            )
            
            # If we haven't crossed 10 messages, summary is not included/needed yet
            if total_messages <= 10:
                logger.debug(f"[Session: {session_id}] Context limit <= 10, skipping background summary context.")
                return active_messages, ""
            
            # Otherwise, return active messages and the running summary
            logger.info(
                f"[Session: {session_id}] Context limit > 10. Prepending background summary "
                f"(length: {len(session.summary)} chars)."
            )
            return active_messages, session.summary

    async def update_summary(self, session_id: str, new_summary: str):
        session = await self.get_or_create_session(session_id)
        async with self.lock:
            session.summary = new_summary
            logger.info(
                f"[Session: {session_id}] Session summary updated successfully. "
                f"New summary length: {len(new_summary)} characters."
            )

    async def get_older_messages_to_summarize(self, session_id: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Gets the messages that are older than the active sliding window of 5 messages
        and the current summary, so they can be merged.
        """
        session = await self.get_or_create_session(session_id)
        async with self.lock:
            # We summarize everything preceding the last 5 active messages
            if len(session.messages) > 5:
                older = session.messages[:-5]
                logger.info(
                    f"[Session: {session_id}] Extracting {len(older)} older messages for memory summarization "
                    f"(retaining last 5 as active sliding window)."
                )
                return older, session.summary
            logger.debug(f"[Session: {session_id}] Not enough messages to trigger older history summarization.")
            return [], ""

    async def clear_session(self, session_id: str):
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"[Session: {session_id}] Memory service wiped session record from memory.")
            else:
                logger.warning(f"[Session: {session_id}] Request to clear session, but session was not found.")
