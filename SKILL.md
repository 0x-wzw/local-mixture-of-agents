---
name: local-mixture-of-agents
description: "Mixture-of-Agents (MoA) pipeline — dual-mode: local Ollama or Ollama Cloud. Parallel reference models → aggregator synthesis for enhanced reasoning."
version: 3.0.0
author: 0x-wzw
trigger: "MoA, Ollama mixture of agents, multi-model aggregation, local reasoning, cloud reasoning, K2 routing"
metadata:
  hermes:
    tags: [ollama, moa, mixture-of-agents, local, cloud, reasoning, multi-model, k2]
---

# Mixture-of-Agents (Dual-Mode: Local + Cloud)

Run a Mixture-of-Agents pipeline via **Ollama** (local) or **Ollama Cloud** (remote). Multiple diverse models generate reference responses in parallel, then an aggregator synthesizes them into a single high-quality answer.

**When to use:** Complex reasoning problems where a single model may hallucinate or miss edge cases — math, coding, architecture decisions, multi-step analysis.

**Two modes:**
| Mode | Endpoint | Auth | GPU required | Best for |
|------|----------|------|------------|----------|
| **local** (default) | `http://127.0.0.1:11434` | None | Yes (or CPU) | Sovereignty, privacy, offline |
| **cloud** | `https://ollama.com/v1` | API key | No | Speed, no hardware constraints |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  USER PROMPT                                             │
└────────────────────┬───────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                       ▼
   ┌─────────┐            ┌─────────┐     ... (N models)
   │ Model A │            │ Model B │
   └────┬────┘            └────┬────┘
        │                        │
        └────────┬───────────────┘
                 ▼
          ┌─────────────┐
          │ Aggregator  │  ← "Synthesize best elements, discard errors"
          └──────┬──────┘
                 ▼
           FINAL ANSWER
```

| Layer | Role | Local defaults | Cloud defaults |
|-------|------|---------------|----------------|
| **Reference** | Parallel generation | `llama3.3`, `qwen2.5`, `mistral`, `phi4` | `qwen3-coder:480b`, `kimi-k2.6`, `deepseek-v4-flash`, `gemma4:31b` |
| **Aggregator** | Critical synthesis | `llama3.3` | `deepseek-v4-flash` |

---

## Prerequisites

### Local mode
1. [Install Ollama](https://ollama.com/download)
2. Pull models:
   ```bash
   ollama pull llama3.3 qwen2.5 mistral phi4
   ```

### Cloud mode
1. Set API key in `~/.hermes/.env`:
   ```
   OLLAMA_API_KEY=your_key_here
   ```
   Or use env var: `export OLLAMA_API_KEY=...`

### Both modes
```bash
pip install -r requirements.txt   # aiohttp
```

---

## Quick Start

### Local mode (default)
```bash
python scripts/local_moa.py "Explain the trade-offs between REST and GraphQL"
```

### Cloud mode
```bash
MOA_MODE=cloud python scripts/local_moa.py "Explain quantum entanglement"
```

### K2-Backbone dynamic routing (optional)
```bash
python scripts/local_moa.py "Refactor this Python class" --k2 --task-type code --budget quality_first
```

---

## Programmatic Usage

```python
import asyncio
import os
import sys
sys.path.insert(0, os.path.expanduser("~/.hermes/skills/local-mixture-of-agents/scripts"))
from local_moa import mixture_of_agents_local

# Local mode (default)
result = asyncio.run(mixture_of_agents_local(
    user_prompt="Your complex question here",
    reference_models=["llama3.3", "qwen2.5"],  # optional override
    aggregator_model="llama3.3",               # optional override
    max_concurrency=2,                        # optional: limit parallelism
))
print(result["response"])

# Cloud mode
os.environ["MOA_MODE"] = "cloud"
result = asyncio.run(mixture_of_agents_local(
    user_prompt="Your complex question here",
    reference_models=["deepseek-v4-flash", "kimi-k2.6"],
    aggregator_model="deepseek-v4-flash",
))

# With K2 routing
result = asyncio.run(mixture_of_agents_local(
    user_prompt="Your complex question here",
    use_k2_routing=True,
    task_type="analysis",    # "code", "research", "creative", ...
    budget="balanced",       # "quality_first", "balanced", "cost_first"
))
```

---

## CLI Options

```bash
python scripts/local_moa.py "<prompt>" [options]

Options:
  --mode {local,cloud}        Execution mode (default: env MOA_MODE or 'local')
  --refs MODEL1,MODEL2,...   Reference models (mode-specific defaults)
  --agg MODEL                Aggregator model (mode-specific default)
  --max-conc N               Max parallel calls (default: 4)
  --k2                       Enable K2-Backbone dynamic model routing
  --task-type TYPE           Task type for K2 routing (default: analysis)
  --budget {quality_first,balanced,cost_first}
                             Budget mode for K2 routing (default: balanced)
  --timeout N                Per-request timeout in seconds (default: 120)
  --ref-temp F               Reference layer temperature (default: 0.6)
  --agg-temp F               Aggregator temperature (default: 0.4)
  --raw                      Print only the response text (no JSON wrapper)
  --list-models              List available models for the current mode and exit
  --debug                    Enable verbose debug logging
  --version                  Print version and exit
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOA_MODE` | `local` | `local` or `cloud` |
| `OLLAMA_URL` | `http://127.0.0.1:11434/v1/chat/completions` | Local API endpoint |
| `OLLAMA_TAGS_URL` | `http://127.0.0.1:11434/api/tags` | Local model list endpoint |
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` | Cloud base URL |
| `OLLAMA_API_KEY` | — | Cloud API key (required for cloud mode). Falls back to `~/.hermes/.env`, then `~/.ollama/token` if unset. |
| `MOA_REF_MAX_TOKENS` | `8000` | Max tokens per reference response |
| `MOA_AGG_MAX_TOKENS` | `16000` | Max tokens for aggregator response |
| `MOA_MAX_CONCURRENCY` | `4` | Max parallel reference calls |

Reference and aggregator temperatures default to `0.6` and `0.4` respectively and are
adjustable via the `--ref-temp` / `--agg-temp` CLI flags.

---

## K2-Backbone Integration (Optional)

When `k2_backbone` is installed, MoA can query K2's capability matrix to dynamically select models best suited for the task type and budget. This is **opt-in** — set `use_k2_routing=True` or pass `--k2`.

```python
# K2 selects models based on:
# - task_type: "code", "analysis", "research", "creative", etc.
# - budget: "quality_first", "balanced", "cost_first"
# - diversity: number of architecturally distinct models

refs, agg = get_k2_routed_models(
    task_type="code",
    budget="quality_first",
    diversity=4,
)
# Returns (['qwen3-coder:480b', 'kimi-k2.7-code', ...], 'deepseek-v4-flash')
```

---

## Failure Handling

| Stage | Behaviour |
|-------|-----------|
| **Pre-flight (local)** | Probes Ollama connectivity and model availability before any API calls. Fails fast with clear message if Ollama is down or models missing. |
| **Pre-flight (cloud)** | Validates `OLLAMA_API_KEY` is present. Fails fast if missing. |
| **Reference layer** | Failed models logged and skipped. Pipeline requires **≥1 successful reference** to proceed. |
| **Aggregator** | If aggregation fails, falls back to the longest successful reference response and returns `success: true` with `degraded: true` and the failure noted in `error`. A usable answer is always returned rather than a raw error string. |
| **All layers** | 3 retry attempts with exponential backoff per model. |

---

## Output Format

```json
{
  "success": true,
  "degraded": false,
  "response": "<synthesized final answer>",
  "models_used": {
    "reference": ["llama3.3", "qwen2.5", "mistral", "phi4"],
    "aggregator": "llama3.3"
  },
  "reference_count": 4,
  "processing_time": 42.5,
  "error": null
}
```

---

## Cost & Performance

| Metric | Local MoA | Cloud MoA | Hermes Built-in MoA |
|--------|-----------|-----------|---------------------|
| API cost | Free (your hardware) | Per-token (Ollama Cloud) | ~5× OpenRouter API calls |
| Typical latency | 30–120s (GPU) / 2–10min (CPU) | 15–60s (cloud) | 15–60s (cloud) |
| Model quality | Open-weight (7B–70B) | Frontier open-weight | Frontier (Claude, GPT, Gemini) |
| Privacy | Fully local | Cloud-transmitted | Cloud-transmitted |
| Offline | ✅ Yes | ❌ No | ❌ No |

Use **local** when privacy, cost, or offline operation are paramount.  
Use **cloud** when you lack local GPU or need faster turnaround.  
Use **Hermes built-in** when maximum reasoning quality is the priority.

---

## Files

| File | Description |
|------|-------------|
| `scripts/local_moa.py` | Full pipeline — dual-mode, K2 routing, CLI |
| `scripts/__init__.py` | Enables `from scripts.local_moa import ...` |
| `SKILL.md` | This file |
| `README.md` | GitHub-facing overview |
| `requirements.txt` | `aiohttp>=3.8.0` |
| `.env.example` | Template for environment variables |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Ollama not reachable` | Ollama not running | `ollama serve` or `sudo systemctl start ollama` |
| `Model 'X' not found` | Model not pulled | `ollama pull X` |
| `OLLAMA_API_KEY not found` | Key not set | Add to `~/.hermes/.env` or export |
| `All reference models failed` | Network issue or API downtime | Check connectivity; retry later |
| `K2-Backbone not available` | K2 not installed | `pip install k2-backbone` or ignore (falls back to defaults) |
| High memory usage | Too many parallel calls | Lower `--max-conc` or `MOA_MAX_CONCURRENCY` |
| Slow on CPU | Normal for local CPU inference | Use cloud mode or reduce reference model count |
