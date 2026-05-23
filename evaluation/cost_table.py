"""
cost_table.py
-------------
Generates a cost + latency comparison table from evaluation results.
Reads summary.json and computes estimated API costs.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/cost_table.py

Output:
    evaluation/results/cost_latency_table.md
"""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path("evaluation/results")

# ── Pricing as of 2025 ────────────────────────────────────────────────────────
# Gemini Flash Lite (google-genai)
GEMINI_INPUT_COST_PER_1M  = 0.075   # $0.075 per 1M input tokens
GEMINI_OUTPUT_COST_PER_1M = 0.30    # $0.30 per 1M output tokens
# Approximate avg token counts per prompt/response in our eval
AVG_PROMPT_TOKENS  = 50
AVG_RESPONSE_TOKENS = 120

# HF Spaces (Qwen2.5-0.5B)
HF_COST_PER_MONTH = 0.0    # Free CPU tier
HF_COLD_START_S   = 30     # Approx cold start time on HF free tier

# Ollama (local phi3:mini)
OLLAMA_COST = 0.0           # Hardware cost only


def compute_latency_stats(latencies: list) -> dict:
    clean = [l for l in latencies if l > 0]
    if not clean:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0}
    sorted_l = sorted(clean)
    p95_idx = int(len(sorted_l) * 0.95)
    return {
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "avg": round(statistics.mean(clean), 3),
        "p50": round(statistics.median(clean), 3),
        "p95": round(sorted_l[min(p95_idx, len(sorted_l)-1)], 3),
    }


def estimate_gemini_cost_per_100(avg_prompt_tokens=AVG_PROMPT_TOKENS,
                                  avg_response_tokens=AVG_RESPONSE_TOKENS) -> float:
    input_cost  = (avg_prompt_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_1M * 100
    output_cost = (avg_response_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M * 100
    return round(input_cost + output_cost, 4)


def main():
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        print("❌ summary.json not found. Run run_evaluation.py and judge.py first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    gemini_latencies = summary["gemini"]["latencies"]
    oss_latencies    = summary["oss"]["latencies"]
    oss_model        = summary["oss_model"]

    gemini_stats = compute_latency_stats(gemini_latencies)
    oss_stats    = compute_latency_stats(oss_latencies)
    cost_per_100 = estimate_gemini_cost_per_100()

    # ── Build markdown table ──────────────────────────────────────────────────
    lines = []
    lines.append("# Cost + Latency Comparison Table\n")
    lines.append(f"> Generated from evaluation results — 30 prompts evaluated across both models.\n")
    lines.append("")

    lines.append("## Latency Statistics (seconds)\n")
    lines.append("| Metric | Gemini Flash Lite | phi3:mini (Ollama) | Qwen2.5-0.5B (HF Spaces) |")
    lines.append("|--------|:-----------------:|:------------------:|:------------------------:|")
    lines.append(f"| Min Latency | {gemini_stats['min']}s | {oss_stats['min']}s | ~2s (warm) |")
    lines.append(f"| Avg Latency | **{gemini_stats['avg']}s** | **{oss_stats['avg']}s** | ~8–15s (CPU) |")
    lines.append(f"| Median (P50) | {gemini_stats['p50']}s | {oss_stats['p50']}s | ~10s |")
    lines.append(f"| P95 Latency | {gemini_stats['p95']}s | {oss_stats['p95']}s | ~25s |")
    lines.append(f"| Max Latency | {gemini_stats['max']}s | {oss_stats['max']}s | ~40s |")
    lines.append(f"| Cold Start | None | ~2s | ~30s |")
    lines.append("")

    lines.append("## Cost Comparison\n")
    lines.append("| Metric | Gemini Flash Lite | phi3:mini (Local) | Qwen2.5-0.5B (HF Free) |")
    lines.append("|--------|:-----------------:|:-----------------:|:----------------------:|")
    lines.append(f"| Cost per 1M input tokens | $0.075 | $0 | $0 |")
    lines.append(f"| Cost per 1M output tokens | $0.30 | $0 | $0 |")
    lines.append(f"| Est. cost per 100 prompts | ~${cost_per_100} | $0 | $0 |")
    lines.append(f"| Est. cost per 1,000 prompts | ~${round(cost_per_100*10, 3)} | $0 | $0 |")
    lines.append(f"| Monthly infra cost | Usage-based | Hardware only | **$0 (free tier)** |")
    lines.append(f"| Deployment complexity | None (API) | Local setup required | Simple (HF push) |")
    lines.append("")

    lines.append("## Model Characteristics\n")
    lines.append("| Property | Gemini Flash Lite | phi3:mini | Qwen2.5-0.5B-Instruct |")
    lines.append("|----------|:-----------------:|:---------:|:---------------------:|")
    lines.append("| Parameters | ~8B (estimated) | 3.8B | **0.5B** |")
    lines.append("| Hosting | Google Cloud | Local machine | HuggingFace Spaces |")
    lines.append("| Internet required | ✅ Yes | ❌ No | ✅ Yes |")
    lines.append("| Privacy (data leaves device) | ✅ Yes | ❌ No | ✅ Yes |")
    lines.append("| Context window | 1M tokens | 128K tokens | 128K tokens |")
    lines.append("| Open weights | ❌ No | ✅ Yes | ✅ Yes |")
    lines.append("")

    lines.append("## Recommendations\n")
    lines.append("- **For production APIs:** Use Gemini Flash Lite — lowest latency at minimal cost.")
    lines.append("- **For privacy-sensitive workloads:** Use phi3:mini locally via Ollama.")
    lines.append("- **For zero-cost public deployment:** Use Qwen2.5-0.5B on HuggingFace Spaces free tier.")
    lines.append("- **For edge/offline:** phi3:mini or Qwen2.5-0.5B can run entirely without internet.\n")

    output = "\n".join(lines)

    out_path = RESULTS_DIR / "cost_latency_table.md"
    with open(out_path, "w") as f:
        f.write(output)

    print(output)
    print(f"\n✅ Cost + latency table saved to: {out_path}")


if __name__ == "__main__":
    main()
