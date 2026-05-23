# Dual AI Personal Assistants Hub & Benchmark Suite

A premium, production-grade side-by-side web workspace and automated evaluation framework comparing hosted Frontier models and local/public Open-Source Software (OSS) personal assistants.

* **Frontier Model:** `gemini-flash-lite-latest` using the official `google-genai` SDK.
* **Open Source Model (OSS):** Local models (e.g. `phi3:mini`) powered via **Ollama**, and publicly deployed via **HuggingFace Spaces** (`microsoft/Phi-3-mini-4k-instruct`).

Built on a robust, async **FastAPI** backend with a high-fidelity, dark-mode glassmorphic single-page user playground, fully instrumented with structured observability, rule/LLM guardrails, conversational summary-buffer memory, and dynamic tool use.

---

## Key Features

* **Side-by-Side Unified Console:** Compare latency, formatting, alignment, and quality of Frontier and OSS models simultaneously in real-time.
* **Hybrid Summarization Memory:** Implements a Conversational Summary Buffer Memory:
  * Retains a strict sliding window of the last 5 messages in high-fidelity.
  * Conversations exceeding 10 messages asynchronously trigger Gemini to summarize older history, maintaining long-term memory without context bloat or scaling costs.
* **Two-Stage Safety Guardrails:** Implements a proactive `GuardrailService`:
  * **Input Validation:** Scans user inputs using regex blocklists (jailbreaks, hate, PII, harm) with a secondary LLM-as-a-judge safety check for borderline cases.
  * **Output Validation:** Intercepts generated responses to detect compliance slips, leakage, or forbidden shell/system commands.
* **Dynamic Tool Use:** Integrated `ToolService` enriches LLM context:
  * `get_current_time`: Returns the formatted date and time.
  * `calculate`: Safely parses and evaluates mathematical expressions.
  * `web_search`: Performs zero-API-key instant factual queries via DuckDuckGo.
* **Structured Observability:** Native `/api/observability/stats` and `/api/observability/traces` endpoints track request counts, latency history (for sparklines), error rates, and guardrail interception statistics.
* **Automated Evaluation Suite:** Full-featured benchmarking runner assessing 30 rigorous scenarios across factual accuracy, sensitive bias, and safety/jailbreak vectors.

---

## Repository Structure

```text
founding_aiml_assignment/
├── app/
│   ├── main.py                 # FastAPI Application entry point & pipeline orchestrator
│   ├── models/
│   │   └── schemas.py          # Pydantic validation schemas
│   ├── services/
│   │   ├── llm_service.py      # Core interfaces for google-genai and local Ollama APIs
│   │   ├── memory_service.py   # State manager for session sliding window and summary memory
│   │   ├── guardrail_service.py# Two-stage input/output safety & jailbreak defense filter
│   │   ├── tool_service.py     # Intent-based context enrichment (Time, Calculator, Web Search)
│   │   └── observability.py    # Structured in-memory telemetry and aggregate stats recorder
│   └── static/
│       ├── index.html          # Sleek glassmorphic side-by-side dashboard UI
│       ├── style.css           # Premium vanilla CSS with micro-animations & custom grid
│       └── app.js              # Client state, parallel Fetch execution, & telemetry charts
├── evaluation/
│   ├── results/
│   │   ├── charts/             # Infographic bar and boxplot visualizations
│   │   ├── evaluation_report.pdf # Publication-quality 1-page business evaluation report
│   │   ├── cost_latency_table.md # Detailed costing and response speed metrics table
│   │   ├── summary.json        # Compiled statistics from both model evaluations
│   │   └── *_scores.json       # Individual judges scoring outputs
│   ├── prompts.py              # 30-scenario test suite dataset (factual, bias, jailbreak)
│   ├── judge.py                # LLM-as-a-judge scoring wrapper via Gemini
│   ├── run_evaluation.py       # Async benchmarking runner orchestrating the 30 trials
│   └── cost_table.py           # Markdown cost and latency table generation utility
├── hf_space/                   # Publicly deployed open-source Gradio assistant
│   ├── app.py                  # Gradio 5.x interface utilizing microsoft/Phi-3-mini-4k-instruct
│   ├── requirements.txt        # Deployed environment package dependencies
│   └── README.md               # HuggingFace Space card metadata
├── tests/
│   └── test_chat.py            # Unit test suite verifying schema validation and memory buffer
├── requirements.txt            # Local Python dependencies
└── README.md                   # Main documentation
```

---

## Setup Instructions

### 1. System Requirements
Ensure you have **Python 3.10+** and **Ollama** installed on your system.

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Virtual Environment Setup & Dependencies Installation
```bash
# Create local virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 4. Fetch Local Open-Source Models
Start the Ollama daemon on your machine, then pull the target open-source model:
```bash
ollama pull phi3:mini
```

### 5. Launching the Comparative UI Workspace
Start the FastAPI backend server:
```bash
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser to access the playground.

### 6. Running Local Verification Tests
Verify all API schemas and memory window behaviors are correct:
```bash
PYTHONPATH=. ./.venv/bin/pytest
```

---

## Evaluation & Benchmarking

A custom automated **LLM-as-a-judge** framework benchmarks both assistants across **30 diverse evaluation scenarios** (10 Factual, 10 Sensitive Bias, 10 Safety/Jailbreak Prompts). Gemini Flash Lite serves as the objective judge.

### Running the Benchmarks
To run the automated suite:
```bash
# Executing prompt trials and saving raw scores + summary.json
PYTHONPATH=. ./.venv/bin/python evaluation/run_evaluation.py

# Generates the comparative Cost + Latency Markdown report
PYTHONPATH=. ./.venv/bin/python evaluation/cost_table.py
```

### Deployed Deliverables (Results Directory)
* **Comparative Report (PDF):** [`evaluation/results/evaluation_report.pdf`](evaluation/results/evaluation_report.pdf) — A publication-quality, executive business report compiling category scores, horizontal latency boxplots, and recommendations.
* **Cost + Latency Benchmarks:** [`evaluation/results/cost_latency_table.md`](evaluation/results/cost_latency_table.md) — Comprehensive infrastructure breakdown and production hosting comparisons.

---

## Bonus: Public Deployment

The Open-Source Assistant is publicly deployed on HuggingFace Spaces running **Gradio 5.16.0** and powered by `microsoft/Phi-3-mini-4k-instruct` (running CPU inference):

**Public Space Demo:** **[https://huggingface.co/spaces/Batman07/founding-oss-assistant](https://huggingface.co/spaces/Batman07/founding-oss-assistant)**

---

## Architecture Decisions

### 1. Hybrid Conversational Summary Buffer Memory
To avoid the prompt bloat associated with retaining full chat history, the application utilizes a dual-buffer approach. It retains a sliding window of the most recent 5 messages verbatim to preserve short-term flow. When history exceeds 10 turns, an asynchronous background task requests the Frontier model to compile an aggregate running summary of all older turns, which is then prepended to subsequent payloads.

### 2. Multi-Stage Guardrail Service
Rather than using a single check, safety filtering is split into two distinct stages:
* **Input stage:** Employs compiled regular expressions to block obvious jailbreak injection strings, malicious requests, PII, and unsafe topics. Borderline cases are escalated to a fast LLM-as-a-judge check.
* **Output stage:** Analyzes completions returned by the local open-source model to ensure it has not returned system paths, terminal structures, or forbidden shell strings, enforcing high alignment.

### 3. Asynchronous Non-Blocking Processing
All background operations (such as memory summarization and telemetry collection) are structured using `asyncio` tasks. This separates chat processing from metadata generation. The main loop delivers responses immediately, while memory and metrics are updated out-of-band.

### 4. Intent-Based Tool Injections
Instead of fully autonomous agent loops that execute multi-step thought processes (which are slow and prone to loops), the architecture implements an intent-driven pattern. The server performs a rapid intent-extraction check, retrieves relevant context from tools (e.g., local system time, DuckDuckGo search results), and injects it directly into the prompt context for the LLM.

---

## Trade-offs Made

### 1. In-Memory telemetries vs. Database Systems
* **Trade-off:** We stored user traces and session statistics in volatile, thread-safe memory lists rather than a persistent database system (like PostgreSQL or Redis).
* **Rationale:** This kept local environment setup extremely light and allowed the dashboard to run without external database credentials or Docker containers. The trade-off is that active statistics are reset upon server restart.

### 2. Local Ollama Execution vs. Hosted Cloud GPUs
* **Trade-off:** Running the OSS assistant locally via Ollama instead of hosted model APIs (like Hugging Face Inference Endpoints or Replicate).
* **Rationale:** Local execution is completely cost-free and provides full offline control. The trade-off is that inference speed depends entirely on host hardware, leading to varying latencies depending on host CPU/GPU cores.

### 3. Native Model Implementations for Phi-3 in Production
* **Trade-off:** Configured `trust_remote_code=False` when loading the tokenizer and model for public deployment on HuggingFace Spaces.
* **Rationale:** While custom repository configurations sometimes implement experimental attention modifications, they frequently fail under Python 3.13 due to rope-scaling parameter key mismatches. Native transformers support guarantees absolute container stability at the cost of experimental custom scripts.

### 4. Direct Tool Injection vs. Re-entrant Agent Loops
* **Trade-off:** We chose single-turn prompt enrichment over multi-turn ReAct loops.
* **Rationale:** Agent loops are highly unpredictable, can get stuck in infinite logic loops, and dramatically increase response latency. Context injection delivers robust results within a predictable latency budget.

---

## Future Roadmap & Enhancements

1. **Persistent Database Store:** Migrate in-memory session states to Redis or PostgreSQL for stateless, horizontal scaling in multi-node containers.
2. **Server-Sent Events (SSE):** Transition from long polling/batch fetches to SSE token-by-token streaming, making responses feel instantaneous.
3. **Advanced Tool Integrations:** Expand the context enrichment service to connect to live mock APIs (weather, code interpreters, custom DB lookup).
4. **Vector Database Memory:** Introduce Retrieval-Augmented Generation (RAG) using a vector DB (e.g. Chroma/Qdrant) to recall past relevant conversation details based on semantic search rather than just summaries.
