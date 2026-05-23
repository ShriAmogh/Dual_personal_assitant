"""
tool_service.py
---------------
Lightweight tool-use layer that detects when a user's message
can be enriched by a tool call and injects the result as context.

Available tools:
  - get_current_time: Returns current date and time
  - calculate:        Safe math expression evaluator
  - web_search:       DuckDuckGo Instant Answer lookup (no API key needed)
"""

import re
import math
import logging
import httpx
from datetime import datetime
from typing import Optional

logger = logging.getLogger("app.tools")


# ── Tool Implementations ──────────────────────────────────────────────────────

def get_current_time() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


def safe_calculate(expression: str) -> str:
    """
    Safely evaluates a mathematical expression.
    Restricts to numeric operations only — no builtins, no imports.
    """
    # Whitelist: only digits, operators, parens, spaces, and math constants
    allowed = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.\^eE]+$')
    # Also allow common math functions by name
    expression = expression.strip()
    expression = expression.replace('^', '**')  # convert caret to Python power

    # Map allowed math names
    safe_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "floor": math.floor,
        "ceil": math.ceil,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Could not evaluate expression: {e}"


async def web_search(query: str) -> str:
    """
    Fetches an instant answer from DuckDuckGo's API (no key required).
    Returns a short factual snippet or empty string if unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            if response.status_code == 200:
                data = response.json()
                # AbstractText is the main factual snippet
                abstract = data.get("AbstractText", "").strip()
                answer = data.get("Answer", "").strip()
                if answer:
                    return f"Web search result: {answer}"
                if abstract:
                    return f"Web search result: {abstract[:300]}"
    except Exception as e:
        logger.warning(f"[Tool: web_search] Request failed: {e}")
    return ""


# ── Intent Detection ──────────────────────────────────────────────────────────

TIME_PATTERNS = [
    r"\b(what|current).{0,15}(time|date|day)\b",
    r"\b(today|right now)\b",
    r"\bwhat\s+(day|month|year)\s+is\s+it\b",
]

CALC_PATTERNS = [
    r"\b(calculate|compute|what\s+is|solve|evaluate)\b.{0,30}[\d\+\-\*\/\(\)]+",
    r"\d+\s*[\+\-\*\/\%\^]\s*\d+",  # bare arithmetic like "15 * 42"
]

SEARCH_PATTERNS = [
    r"\b(who\s+is|what\s+is|where\s+is|when\s+(was|did|is))\b",
    r"\b(latest|current|recent)\b.{0,20}\b(news|update|price|score)\b",
    r"\b(wikipedia|search|look\s+up)\b",
]

CALC_EXTRACTOR = re.compile(r'[\d\s\+\-\*\/\%\^\(\)\.]+')


class ToolService:
    async def maybe_invoke(self, message: str) -> Optional[str]:
        """
        Detect tool intent in a user message and invoke the appropriate tool.
        Returns the tool result string to inject as context, or None.
        """
        lower = message.lower()

        # ── Time tool ─────────────────────────────────────────────────────────
        if any(re.search(p, lower) for p in TIME_PATTERNS):
            result = get_current_time()
            logger.info(f"[Tool] Invoked: get_current_time → {result}")
            return result

        # ── Calculator tool ───────────────────────────────────────────────────
        if any(re.search(p, lower) for p in CALC_PATTERNS):
            match = CALC_EXTRACTOR.search(message)
            if match:
                expr = match.group(0).strip()
                if any(op in expr for op in ['+', '-', '*', '/', '%', '^']):
                    result = safe_calculate(expr)
                    logger.info(f"[Tool] Invoked: calculate('{expr}') → {result}")
                    return result

        # ── Web search tool ───────────────────────────────────────────────────
        if any(re.search(p, lower) for p in SEARCH_PATTERNS):
            result = await web_search(message)
            if result:
                logger.info(f"[Tool] Invoked: web_search → {result[:80]}...")
                return result

        return None
