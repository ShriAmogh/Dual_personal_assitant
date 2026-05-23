"""
observability_service.py
------------------------
Lightweight in-memory observability / telemetry service.

Records every LLM request trace and exposes aggregate stats for the dashboard.
Traces are stored in memory (can be extended to SQLite or PostgreSQL).
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger("app.observability")


@dataclass
class Trace:
    """Single LLM request trace record."""
    trace_id:     str
    session_id:   str
    model_type:   str       # "frontier" | "oss"
    model_name:   str       # "Gemini Flash Lite" | "Ollama (phi3:mini)"
    prompt:       str
    response:     str
    latency:      float     # seconds
    memory_size:  int       # number of messages in active context at time of request
    timestamp:    float = field(default_factory=time.time)
    error:        bool = False
    blocked_by_guardrail: bool = False


class ObservabilityService:
    def __init__(self):
        self._traces: List[Trace] = []
        self._counter = 0
        logger.info("ObservabilityService initialized.")

    def _new_trace_id(self) -> str:
        self._counter += 1
        return f"trace_{self._counter:05d}"

    def record(
        self,
        session_id: str,
        model_type: str,
        model_name: str,
        prompt: str,
        response: str,
        latency: float,
        memory_size: int = 0,
        error: bool = False,
        blocked_by_guardrail: bool = False,
    ) -> str:
        """Record a new LLM request trace. Returns the trace_id."""
        trace = Trace(
            trace_id=self._new_trace_id(),
            session_id=session_id,
            model_type=model_type,
            model_name=model_name,
            prompt=prompt,
            response=response,
            latency=latency,
            memory_size=memory_size,
            error=error,
            blocked_by_guardrail=blocked_by_guardrail,
        )
        self._traces.append(trace)
        logger.info(
            f"[Trace: {trace.trace_id}] Recorded — model: {model_name}, "
            f"latency: {latency:.3f}s, memory_size: {memory_size}, error: {error}"
        )
        return trace.trace_id

    def get_traces(
        self,
        session_id: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Return recent traces, optionally filtered by session or model type."""
        traces = self._traces
        if session_id:
            traces = [t for t in traces if t.session_id == session_id]
        if model_type:
            traces = [t for t in traces if t.model_type == model_type]
        # Return most recent `limit` traces as dicts (without full prompt/response for brevity)
        result = []
        for t in reversed(traces[-limit:]):
            d = asdict(t)
            d["prompt"] = d["prompt"][:80] + "..." if len(d["prompt"]) > 80 else d["prompt"]
            d["response"] = d["response"][:120] + "..." if len(d["response"]) > 120 else d["response"]
            result.append(d)
        return result

    def get_stats(self) -> Dict:
        """Return aggregate statistics across all recorded traces."""
        if not self._traces:
            return {
                "total_requests": 0,
                "frontier": self._empty_model_stats(),
                "oss": self._empty_model_stats(),
            }

        def model_stats(traces):
            if not traces:
                return self._empty_model_stats()
            latencies = [t.latency for t in traces if not t.error]
            errors = sum(1 for t in traces if t.error)
            blocked = sum(1 for t in traces if t.blocked_by_guardrail)
            return {
                "total_calls": len(traces),
                "error_count": errors,
                "error_rate": round(errors / len(traces) * 100, 1),
                "guardrail_blocks": blocked,
                "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0,
                "min_latency": round(min(latencies), 3) if latencies else 0,
                "max_latency": round(max(latencies), 3) if latencies else 0,
                # Latency buckets for sparkline
                "latency_history": [round(t.latency, 3) for t in traces[-20:]],
            }

        frontier_traces = [t for t in self._traces if t.model_type == "frontier"]
        oss_traces      = [t for t in self._traces if t.model_type == "oss"]

        return {
            "total_requests": len(self._traces),
            "frontier": model_stats(frontier_traces),
            "oss": model_stats(oss_traces),
        }

    @staticmethod
    def _empty_model_stats() -> Dict:
        return {
            "total_calls": 0,
            "error_count": 0,
            "error_rate": 0,
            "guardrail_blocks": 0,
            "avg_latency": 0,
            "min_latency": 0,
            "max_latency": 0,
            "latency_history": [],
        }
