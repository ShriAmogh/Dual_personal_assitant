# Cost + Latency Comparison Table

> Generated from evaluation results — 30 prompts evaluated across both models.


## Latency Statistics (seconds)

| Metric | Gemini Flash Lite | phi3:mini (Ollama) | Qwen2.5-0.5B (HF Spaces) |
|--------|:-----------------:|:------------------:|:------------------------:|
| Min Latency | 0.751s | 0.374s | ~2s (warm) |
| Avg Latency | **4.487s** | **5.057s** | ~8–15s (CPU) |
| Median (P50) | 2.337s | 3.237s | ~10s |
| P95 Latency | 14.882s | 14.24s | ~25s |
| Max Latency | 18.365s | 21.857s | ~40s |
| Cold Start | None | ~2s | ~30s |

## Cost Comparison

| Metric | Gemini Flash Lite | phi3:mini (Local) | Qwen2.5-0.5B (HF Free) |
|--------|:-----------------:|:-----------------:|:----------------------:|
| Cost per 1M input tokens | $0.075 | $0 | $0 |
| Cost per 1M output tokens | $0.30 | $0 | $0 |
| Est. cost per 100 prompts | ~$0.004 | $0 | $0 |
| Est. cost per 1,000 prompts | ~$0.04 | $0 | $0 |
| Monthly infra cost | Usage-based | Hardware only | **$0 (free tier)** |
| Deployment complexity | None (API) | Local setup required | Simple (HF push) |

## Model Characteristics

| Property | Gemini Flash Lite | phi3:mini | Qwen2.5-0.5B-Instruct |
|----------|:-----------------:|:---------:|:---------------------:|
| Parameters | ~8B (estimated) | 3.8B | **0.5B** |
| Hosting | Google Cloud | Local machine | HuggingFace Spaces |
| Internet required | ✅ Yes | ❌ No | ✅ Yes |
| Privacy (data leaves device) | ✅ Yes | ❌ No | ✅ Yes |
| Context window | 1M tokens | 128K tokens | 128K tokens |
| Open weights | ❌ No | ✅ Yes | ✅ Yes |

## Recommendations

- **For production APIs:** Use Gemini Flash Lite — lowest latency at minimal cost.
- **For privacy-sensitive workloads:** Use phi3:mini locally via Ollama.
- **For zero-cost public deployment:** Use Qwen2.5-0.5B on HuggingFace Spaces free tier.
- **For edge/offline:** phi3:mini or Qwen2.5-0.5B can run entirely without internet.
