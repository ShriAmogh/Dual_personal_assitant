"""
run_evaluation.py
-----------------
Runs all 30 evaluation prompts against both the Frontier (Gemini) and OSS (Ollama/phi3:mini)
models using isolated, fresh sessions for each prompt.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/run_evaluation.py [--oss-model phi3:mini]

Outputs:
    evaluation/results/gemini_results.json
    evaluation/results/phi3_results.json
"""

import asyncio
import json
import os
import time
import argparse
import logging
from pathlib import Path

from app.services.llm_service import LlmService
from evaluation.prompts import PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluation.runner")

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def run_single(llm: LlmService, prompt_obj: dict, model_type: str, oss_model: str) -> dict:
    """Run a single prompt against one model and return structured result."""
    prompt_text = prompt_obj["prompt"]
    # Each prompt gets a fresh, isolated 1-message context — no cross-contamination
    messages = [{"role": "user", "content": prompt_text}]

    start = time.perf_counter()
    if model_type == "frontier":
        response, latency = await llm.generate_frontier(messages, summary="")
    else:
        response, latency = await llm.generate_oss(messages, summary="", model_name=oss_model)
    
    return {
        "id": prompt_obj["id"],
        "category": prompt_obj["category"],
        "prompt": prompt_text,
        "ground_truth": prompt_obj.get("ground_truth"),
        "expected": prompt_obj.get("expected"),
        "response": response,
        "latency": round(latency, 3),
    }


async def run_all(llm: LlmService, model_type: str, oss_model: str) -> list:
    """Run all prompts sequentially against one model."""
    results = []
    total = len(PROMPTS)
    model_label = "Frontier (Gemini)" if model_type == "frontier" else f"OSS ({oss_model})"

    logger.info(f"{'─'*55}")
    logger.info(f"Starting evaluation for: {model_label}")
    logger.info(f"Total prompts: {total}")
    logger.info(f"{'─'*55}")

    for i, prompt_obj in enumerate(PROMPTS, 1):
        logger.info(f"[{i:02d}/{total}] [{prompt_obj['category'].upper()}] {prompt_obj['id']} — Sending prompt...")
        result = await run_single(llm, prompt_obj, model_type, oss_model)
        results.append(result)
        logger.info(f"          ✓ Latency: {result['latency']}s | Response ({len(result['response'])} chars)")
        # Small throttle to avoid rate-limiting on Gemini free tier
        if model_type == "frontier":
            await asyncio.sleep(1.5)

    return results


async def main(oss_model: str):
    llm = LlmService()

    # ── Run Frontier ──────────────────────────────────────────────
    gemini_results = await run_all(llm, "frontier", oss_model)
    gemini_path = RESULTS_DIR / "gemini_results.json"
    with open(gemini_path, "w") as f:
        json.dump(gemini_results, f, indent=2)
    logger.info(f"\n✅ Gemini results saved to: {gemini_path}")

    # ── Run OSS ───────────────────────────────────────────────────
    oss_results = await run_all(llm, "oss", oss_model)
    oss_path = RESULTS_DIR / f"{oss_model.replace(':', '_')}_results.json"
    with open(oss_path, "w") as f:
        json.dump(oss_results, f, indent=2)
    logger.info(f"✅ OSS results saved to: {oss_path}")

    logger.info("\n🎉 Evaluation run complete. Next step: run judge.py to score results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation prompts against both models.")
    parser.add_argument(
        "--oss-model",
        default="phi3:mini",
        help="Name of the local Ollama model to evaluate (default: phi3:mini)"
    )
    args = parser.parse_args()
    asyncio.run(main(oss_model=args.oss_model))
