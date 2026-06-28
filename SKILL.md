---
name: local-mixture-of-agents
description: "Run a local Mixture-of-Agents (MoA) pipeline via Ollama — parallel reference models → aggregator synthesis for enhanced reasoning on complex problems."
version: 1.0.0
author: Agent
trigger: "local MoA, Ollama mixture of agents, multi-model aggregation, local reasoning pipeline"
metadata:
  hermes:
    tags: [ollama, moa, mixture-of-agents, local-llm, reasoning, multi-model]
---

# Local Mixture-of-Agents (Ollama)

Run a sovereign-grade Mixture-of-Agents pipeline entirely on local hardware via Ollama. This skill recreates Hermes's built-in MoA architecture using open-weight models instead of cloud frontier APIs.

**When to use:** complex reasoning problems where a single local model may hallucinate or miss edge cases — math, coding, architecture decisions, multi-step analysis. Accepts the trade-off of higher latency for improved answer quality.

---

## Architecture

| Layer | Role | Models |
|-------|------|--------|
| **Layer 1: Reference** | Parallel generation (diverse perspectives) | `llama3.3`, `qwen2.5`, `mistral`, `phi4` |
| **Layer 2: Aggregator** | Critical synthesis into final answer | `llama3.3` (configurable) |

All calls hit the Ollama OpenAI-compatible API at `http://127.0.0.1:11434/v1/chat/completions`.

---

## Prerequisites

1. **Ollama running locally**
   ```bash
   ollama serve
   ```

2. **Models pulled**
   ```bash
   ollama pull llama3.3
   ollama pull qwen2.5
   ollama pull mistral
   ollama pull phi4
   ```

3. **Python dependencies**
   ```bash
   pip install aiohttp
   ```

---

## Usage from Hermes

### Option A: Direct script execution (recommended)

```bash
python ~/.hermes/skills/local-mixture-of-agents/scripts/local_moa.py "<your complex prompt>"
```

The script prints a JSON result to stdout:

```json
{
  "success": true,
  "response": "<synthesized final answer>",
  "models_used": {
    "reference": ["llama3.3", "qwen2.5", "mistral", "phi4"],
    "aggregator": "llama3.3"
  },
  "reference_count": 4,
  "processing_time": 45.2
}
```

### Option B: Via `execute_code`

```python
import asyncio
import sys
sys.path.insert(0, "~/.hermes/skills/local-mixture-of-agents/scripts")
from local_moa import mixture_of_agents_local

result = asyncio.run(mixture_of_agents_local(
    user_prompt="Explain the trade-offs between REST and GraphQL APIs...",
    reference_models=["llama3.3", "qwen2.5"],  # optional: override defaults
    aggregator_model="llama3.3"                 # optional: override default
))

print(result["response"])
```

### Option C: Customise model lineup

Edit the constants at the top of `scripts/local_moa.py`:

```python
REFERENCE_MODELS = ["llama3.3", "qwen2.5", "mistral", "phi4"]
AGGREGATOR_MODEL = "llama3.3"
```

Or pass them at runtime as shown in Option B.

---

## Customisation Guide

| Parameter | Default | Tuning advice |
|-----------|---------|---------------|
| `REFERENCE_MODELS` | 4 models | More models = more diversity, but linearly slower. Minimum viable: 2. |
| `AGGREGATOR_MODEL` | `llama3.3` | Use your largest / best local model. This is the quality bottleneck. |
| `REFERENCE_TEMPERATURE` | `0.6` | Higher = more diverse perspectives. Lower = more consistent references. |
| `AGGREGATOR_TEMPERATURE` | `0.4` | Keep low. The aggregator must synthesize, not invent. |
| `timeout` | `120s` | Increase for slow CPUs or large context windows. |

**Model diversity matters more than raw parameter count.** If all your models share the same architecture (e.g., all Llama-based), cross-referencing provides less value than mixing families (Llama + Qwen + Mistral + Phi).

---

## Failure Handling

- The pipeline requires **≥1 successful reference model** to proceed to aggregation.
- Failed models are logged to stderr and skipped silently.
- If all references fail, the tool returns a JSON error with `success: false`.
- Each model has 3 retry attempts with exponential backoff.

---

## Cost & Performance

| Metric | Local MoA | Hermes Built-in MoA |
|--------|-----------|---------------------|
| API cost | Free (your hardware) | ~5× OpenRouter API calls |
| Typical latency | 30–120s (GPU) / 2–10min (CPU) | 15–60s (cloud) |
| Model quality | Open-weight (7B–70B) | Frontier (Claude, GPT, Gemini) |
| Privacy | Fully local | Cloud-transmitted |

Use local MoA when privacy, cost, or offline operation are paramount. Use Hermes built-in MoA when maximum reasoning quality is the priority.

---

## Files

- `scripts/local_moa.py` — Full pipeline script (Layer 1 + Layer 2)
