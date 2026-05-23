"""
guardrail_service.py
--------------------
Two-stage safety filter applied to every chat request:

  Stage 1 — Input Guardrail:
    • Rule-based: regex patterns for known jailbreaks and dangerous keywords
    • LLM-based: Gemini classifies borderline messages as SAFE / UNSAFE

  Stage 2 — Output Guardrail:
    • Pattern checks on generated response to catch any safety slippage

Returns a GuardrailResult indicating safe/blocked with category and reason.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger("app.guardrail")


@dataclass
class GuardrailResult:
    safe: bool
    category: str   # "safe" | "jailbreak" | "harmful" | "pii" | "hate"
    reason: str


# ── Rule-Based Pattern Library ────────────────────────────────────────────────

JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+(are|have)\s+no\s+restrictions",
    r"developer\s+mode",
    r"pretend\s+you\s+(are|have\s+no)",
    r"jailbreak",
    r"act\s+as\s+(an?\s+)?ai\s+with(out)?\s+(any\s+)?restrictions",
    r"system\s+prompt",
    r"override\s+your\s+(safety|guidelines|instructions)",
]

HARMFUL_PATTERNS = [
    r"\b(make|build|create|synthesize)\b.{0,30}\b(bomb|explosive|weapon|poison|drug|meth|fentanyl)\b",
    r"\b(hack|exploit|bypass)\b.{0,30}\b(password|account|system|database|server)\b",
    r"\b(phishing|malware|ransomware|keylogger|backdoor)\b",
    r"\b(how\s+to\s+)?(make|produce|synthesize)\b.{0,20}\b(chemical\s+weapon|nerve\s+agent)\b",
    r"step.{0,10}by.{0,10}step.{0,20}(kill|murder|stalk|harm)",
    r"\b(track|stalk)\b.{0,30}\b(without\s+them\s+knowing|secretly|covertly)\b",
]

HATE_PATTERNS = [
    r"\b(hate\s+speech|slur|racial\s+slur)\b",
    r"generate.{0,20}(hate|racist|sexist|homophobic).{0,20}(post|content|message)",
]

PII_REQUEST_PATTERNS = [
    r"\b(credit\s+card|ssn|social\s+security|passport\s+number)\b.{0,30}\b(format|template|example)\b",
]

OUTPUT_HARMFUL_PATTERNS = [
    r"here\s+(is|are)\s+(the\s+)?(instructions?|steps?).{0,30}(bomb|explosive|weapon)",
    r"(sudo|rm\s+-rf|format\s+c:)",   # dangerous shell commands
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b.{0,10}\b(password|credentials)\b",
]


def _check_patterns(text: str, patterns: List[str]) -> Optional[str]:
    """Returns the first matched pattern string, or None."""
    lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, lower, re.IGNORECASE):
            return pattern
    return None


class GuardrailService:
    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: Optional LlmService instance for LLM-based classification
                         on borderline cases. If None, only rule-based checks run.
        """
        self.llm_service = llm_service
        logger.info("GuardrailService initialized (rule-based + optional LLM judge).")

    async def check_input(self, message: str) -> GuardrailResult:
        """
        Stage 1: Validate user input before sending to LLM.
        Returns GuardrailResult indicating safe or blocked.
        """
        # ── Jailbreak detection ───────────────────────────────────────────────
        matched = _check_patterns(message, JAILBREAK_PATTERNS)
        if matched:
            logger.warning(f"[Guardrail] INPUT BLOCKED — Jailbreak pattern detected.")
            return GuardrailResult(
                safe=False,
                category="jailbreak",
                reason="Your message appears to contain a prompt injection or jailbreak attempt.",
            )

        # ── Harmful content detection ─────────────────────────────────────────
        matched = _check_patterns(message, HARMFUL_PATTERNS)
        if matched:
            logger.warning(f"[Guardrail] INPUT BLOCKED — Harmful content pattern detected.")
            return GuardrailResult(
                safe=False,
                category="harmful",
                reason="Your message contains a request for potentially harmful information.",
            )

        # ── Hate speech detection ─────────────────────────────────────────────
        matched = _check_patterns(message, HATE_PATTERNS)
        if matched:
            logger.warning(f"[Guardrail] INPUT BLOCKED — Hate speech pattern detected.")
            return GuardrailResult(
                safe=False,
                category="hate",
                reason="Your message appears to request hateful or discriminatory content.",
            )

        # ── PII solicitation detection ────────────────────────────────────────
        matched = _check_patterns(message, PII_REQUEST_PATTERNS)
        if matched:
            logger.warning(f"[Guardrail] INPUT BLOCKED — PII solicitation detected.")
            return GuardrailResult(
                safe=False,
                category="pii",
                reason="Your message appears to request sensitive personal information templates.",
            )

        # ── Optional LLM-based check for borderline messages ──────────────────
        if self.llm_service and len(message) > 20:
            try:
                result = await self._llm_safety_check(message)
                if not result.safe:
                    return result
            except Exception as e:
                # LLM judge failure is non-blocking — fail open (let message through)
                logger.warning(f"[Guardrail] LLM safety check failed: {e}. Failing open.")

        logger.debug(f"[Guardrail] Input passed all checks.")
        return GuardrailResult(safe=True, category="safe", reason="")

    async def check_output(self, response: str) -> GuardrailResult:
        """
        Stage 2: Validate model output before returning to user.
        Catches cases where the model partially complied despite instructions.
        """
        matched = _check_patterns(response, OUTPUT_HARMFUL_PATTERNS)
        if matched:
            logger.warning(f"[Guardrail] OUTPUT BLOCKED — Harmful content in model response.")
            return GuardrailResult(
                safe=False,
                category="harmful",
                reason="Model response was flagged and blocked by the safety filter.",
            )

        return GuardrailResult(safe=True, category="safe", reason="")

    async def _llm_safety_check(self, message: str) -> GuardrailResult:
        """Use Gemini as a lightweight LLM-based safety judge for ambiguous messages."""
        prompt = f"""You are a content safety classifier. Classify the following user message.

Message: "{message}"

Reply with ONLY one of these exact labels:
- SAFE: The message is a legitimate, harmless request.
- UNSAFE: The message attempts to extract harmful, dangerous, or policy-violating content.

Reply with a single JSON object: {{"label": "SAFE"|"UNSAFE", "reason": "one sentence"}}"""

        msgs = [{"role": "user", "content": prompt}]
        raw, _ = await self.llm_service.generate_frontier(msgs, summary="")

        import json
        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(clean)
            if parsed.get("label") == "UNSAFE":
                logger.warning(f"[Guardrail] LLM judge flagged input as UNSAFE: {parsed.get('reason')}")
                return GuardrailResult(
                    safe=False,
                    category="harmful",
                    reason=parsed.get("reason", "Message flagged as unsafe by content filter."),
                )
        except Exception:
            pass  # Parse failure → fail open

        return GuardrailResult(safe=True, category="safe", reason="")
