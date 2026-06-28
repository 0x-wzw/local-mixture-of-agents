---
name: local-mixture-of-agents
description: "Run a Mixture-of-Agents (MoA) pipeline via Ollama Cloud — parallel reference models → aggregator synthesis for enhanced reasoning on complex problems."
version: 2.0.0
author: 0x-wzw
trigger: "MoA, Ollama mixture of agents, multi-model aggregation, cloud reasoning pipeline"
metadata:
  hermes:
    tags: [ollama, moa, mixture-of-agents, ollama-cloud, reasoning, multi-model]
---

# Mixture-of-Agents (Ollama Cloud)

Run a Mixture-of-Agents pipeline via the Ollama Cloud OpenAI-compatible endpoint. Multiple diverse models generate reference responses in parallel, then an aggregator model synthesizes them into a single high-quality answer.

**When to use:** complex reasoning problems where a single model may hallucinate or miss edge cases — math, coding, architecture decisions, multi-step analysis. Cloud endpoint means no local GPU required.

---

## Architecture

| Layer | Role | Default Models |
|-------|------|----------------|
| **Layer 1: Reference** | Parallel generation (diverse perspectives) | `qwen3-coder:480b`, `kimi-k2.6`, `deepseek-v4-flash`, `gemma4:31b` |
| **Layer 2: Aggregator** | Critical synthesis into final answer | `deepseek-v4-flash` (configurable) |

All calls hit the Ollama Cloud API at `https://ollama.com/v1/chat/completions`.

API key is resolved from (in order):
1. `OLLAMA_API_KEY` environment variable
2. `~/.hermes/.env` file
3. `~/.ollama/token` file (ollama CLI token)

---

## Prerequisites

1. **Ollama Cloud API key** — set in `~/.hermes/.env`:
   ```
   OLLAMA_API_KEY=your_key_here
   ```

2. **Python dependencies**
   ```bash
   pip install aiohttp
   ```

---

## Available Ollama Cloud Models

35 models available as of June 2026. The defaults below were chosen for **architectural diversity** (Qwen + Kimi + DeepSeek + Gemma) which provides more value than using models from the same family.

**Full list:** `qwen3-coder:480b`, `deepseek-v4-pro`, `deepseek-v3.1:671b`, `qwen3-coder-next`, `gemma3:12b`, `glm-4.7`, `glm-5.1`, `kimi-k2.6`, `kimi-k2.7-code`, `nemotron-3-nano:30b`, `minimax-m2.5`, `kimi-k2.5`, `minimax-m2.1`, `ministral-3:14b`, `mistral-large-3:675b`, `gemma3:4b`, `gemma3:27b`, `nemotron-3-super`, `deepseek-v4-flash`, `ministral-3:3b`, `devstral-2:123b`, `rnj-1:8b`, `qwen3.5:397b`, `deepseek-v3.2`, `gpt-oss:20b`, `minimax-m2.7`, `devstral-small-2:24b`, `gemma4:31b`, `nemotron-3-ultra`, `gemini-3-flash-preview`, `gpt-oss:120b`, `minimax-m3`, `ministral-3:8b`, `glm-5`, `glm-5.2`

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
    "reference": ["qwen3-coder:480b", "kimi-k2.6", "deepseek-v4-flash", "gemma4:31b"],
    "aggregator": "deepseek-v4-flash"
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
    reference_models=["qwen3-coder:480b", "kimi-k2.6"],  # optional: override defaults
    aggregator_model="deepseek-v4-flash"                  # optional: override default
))

print(result["response"])
```

### Option C: Customise model lineup

Edit the constants at the top of `scripts/local_moa.py`:

```python
REFERENCE_MODELS = ["qwen3-coder:480b", "kimi-k2.6", "deepseek-v4-flash", "gemma4:31b"]
AGGREGATOR_MODEL = "deepseek-v4-flash"
```

Or pass them at runtime as shown in Option B.

---

## Customisation Guide

| Parameter | Default | Tuning advice |
|-----------|---------|---------------|
| `REFERENCE_MODELS` | 4 models | More models = more diversity, but linearly slower. Minimum viable: 2. |
| `AGGREGATOR_MODEL` | `deepseek-v4-flash` | Use your strongest model. This is the quality bottleneck. |
| `REFERENCE_TEMPERATURE` | `0.6` | Higher = more diverse perspectives. Lower = more consistent references. |
| `AGGREGATOR_TEMPERATURE` | `0.4` | Keep low. The aggregator must synthesize, not invent. |
| `timeout` | `120s` | Cloud models are fast; increase only for very long context. |

**Model diversity matters more than raw parameter count.** If all your models share the same architecture (e.g., all Llama-based), cross-referencing provides less value than mixing families (Qwen + Kimi + DeepSeek + Gemma).

---

## Failure Handling

- The pipeline requires **≥1 successful reference model** to proceed to aggregation.
- Failed models are logged to stderr and skipped silently.
- If all references fail, the tool returns a JSON error with `success: false`.
- Each model has 3 retry attempts with exponential backoff.
- If `OLLAMA_API_KEY` is not found, returns an immediate error with setup instructions.

---

## Cost & Performance

| Metric | Ollama Cloud MoA | Local Ollama MoA | Hermes Built-in MoA |
|--------|-------------------|-------------------|---------------------|
| API cost | Included in Ollama Cloud subscription | Free (your hardware) | ~5× OpenRouter API calls |
| Typical latency | 10–60s (cloud) | 30–120s (GPU) / 2–10min (CPU) | 15–60s (cloud) |
| Model quality | Open-weight (up to 480B) | Open-weight (7B–70B) | Frontier (Claude, GPT, Gemini) |
| Privacy | Cloud-transmitted | Fully local | Cloud-transmitted |
| Local GPU required | No | Yes | No |

---

## Files

- `scripts/local_moa.py` — Full pipeline script (Layer 1 + Layer 2)