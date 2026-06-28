# Local Mixture-of-Agents (Ollama)

A sovereign-grade Mixture-of-Agents pipeline that runs entirely on local hardware via [Ollama](https://ollama.com). Recreates the research-backed 2-layer MoA architecture using open-weight models — no cloud APIs, no tokens, no data exfiltration.

## What is MoA?

[Mixture-of-Agents](https://arxiv.org/abs/2406.04692) (Wang et al., 2024) is a multi-layer LLM collaboration technique:

1. **Layer 1 (Reference):** Multiple models generate independent answers in parallel.
2. **Layer 2 (Aggregator):** A strong model critically evaluates and synthesizes the best elements into one refined answer.

This repo implements that architecture for local Ollama models.

## Quick Start

```bash
# 1. Install Ollama and pull models
ollama pull llama3.3 qwen2.5 mistral phi4

# 2. Install Python dependency
pip install -r requirements.txt

# 3. Run
python scripts/local_moa.py "Explain the trade-offs between REST and GraphQL"
```

## Default Model Stack

| Role | Models |
|------|--------|
| **Reference** (Layer 1) | `llama3.3`, `qwen2.5`, `mistral`, `phi4` |
| **Aggregator** (Layer 2) | `llama3.3` |

## Features

- **Fully local** — no API keys, no cloud dependency
- **Failure-tolerant** — requires only ≥1 successful reference to proceed
- **Configurable** — model lineup, concurrency, token budgets via env vars or CLI flags
- **Pre-flight checks** — validates Ollama connectivity and model availability before calling
- **Bounded concurrency** — prevents memory thrashing on CPU-only systems
- **Aggregator failure detection** — never silently returns an error string as the answer

## CLI Options

```bash
python scripts/local_moa.py "<prompt>" [options]

  --refs MODEL1,MODEL2,...   Reference models (default: llama3.3,qwen2.5,mistral,phi4)
  --agg MODEL                Aggregator model (default: llama3.3)
  --max-conc N               Max parallel calls (default: 4)
  --debug                    Enable verbose debug logging
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://127.0.0.1:11434/v1/chat/completions` | Ollama API endpoint |
| `OLLAMA_TAGS_URL` | `http://127.0.0.1:11434/api/tags` | Ollama model list endpoint |
| `MOA_REF_MAX_TOKENS` | `8000` | Max tokens per reference response |
| `MOA_AGG_MAX_TOKENS` | `16000` | Max tokens for aggregator response |
| `MOA_MAX_CONCURRENCY` | `4` | Max parallel reference calls |

## Programmatic Usage

```python
import asyncio
import os
import sys
sys.path.insert(0, os.path.expanduser("~/.hermes/skills/local-mixture-of-agents/scripts"))
from local_moa import mixture_of_agents_local

result = asyncio.run(mixture_of_agents_local(
    user_prompt="Your complex question here",
    reference_models=["llama3.3", "qwen2.5"],
    aggregator_model="llama3.3",
    max_concurrency=2,
))
print(result["response"])
```

## Output Format

```json
{
  "success": true,
  "response": "<synthesized final answer>",
  "models_used": {
    "reference": ["llama3.3", "qwen2.5"],
    "aggregator": "llama3.3"
  },
  "reference_count": 2,
  "processing_time": 42.5,
  "error": null
}
```

## When to Use

- Complex math, coding, or architecture problems where a single model may hallucinate
- Offline or air-gapped environments
- Privacy-sensitive workloads
- Cost-constrained scenarios

## When NOT to Use

- Simple questions (single model is faster and sufficient)
- Speed-critical applications (local MoA is slower than cloud)
- When you have access to Hermes built-in MoA and maximum quality is the priority

## License

MIT
