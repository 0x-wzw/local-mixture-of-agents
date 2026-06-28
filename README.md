# Local Mixture-of-Agents (Dual-Mode)

A production-ready Mixture-of-Agents pipeline that runs **locally via Ollama** or **remotely via Ollama Cloud**. Implements the research-backed 2-layer MoA architecture using open-weight models — with optional K2-Backbone dynamic model routing.

## What is MoA?

[Mixture-of-Agents](https://arxiv.org/abs/2406.04692) (Wang et al., 2024) is a multi-layer LLM collaboration technique:

1. **Layer 1 (Reference):** Multiple models generate independent answers in parallel.
2. **Layer 2 (Aggregator):** A strong model critically evaluates and synthesizes the best elements into one refined answer.

This repo implements that architecture with **dual-mode support** (local + cloud) and **optional K2-Backbone integration** for dynamic model selection.

## Quick Start

### Local mode (default)
```bash
# 1. Install Ollama and pull models
ollama pull llama3.3 qwen2.5 mistral phi4

# 2. Install dependency
pip install -r requirements.txt

# 3. Run
python scripts/local_moa.py "Explain the trade-offs between REST and GraphQL"
```

### Cloud mode
```bash
# Set API key
export OLLAMA_API_KEY=*** # Or add to ~/.hermes/.env

# Run
MOA_MODE=cloud python scripts/local_moa.py "Explain quantum entanglement"
```

### With K2-Backbone routing
```bash
python scripts/local_moa.py "Refactor this class" --k2 --task-type code --budget quality_first
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  USER PROMPT                                │
└──────────────┬──────────────────────────────┘
               │
   ┌───────────┴───────────┐
   ▼                       ▼
┌─────────┐         ┌─────────┐     ...
│ Model A │         │ Model B │
└────┬────┘         └────┬────┘
     │                   │
     └────────┬──────────┘
              ▼
       ┌─────────────┐
       │ Aggregator  │
       └──────┬──────┘
              ▼
        FINAL ANSWER
```

## Features

- **Dual-mode** — switch between local Ollama (sovereign, free) and Ollama Cloud (fast, no GPU) via `MOA_MODE` env var
- **K2-Backbone integration** — dynamic model selection via capability matrix (opt-in, falls back gracefully)
- **Failure-tolerant** — requires only ≥1 successful reference; failed models skipped with logging
- **Pre-flight checks** — validates connectivity and model availability before calling
- **Bounded concurrency** — `asyncio.Semaphore` prevents memory thrashing
- **Aggregator failure detection** — never silently returns an error string as the answer
- **Structured logging** — timestamped DEBUG/INFO/WARNING/ERROR levels

## CLI Options

```bash
python scripts/local_moa.py "<prompt>" [options]

  --mode {local,cloud}       Execution mode (default: env MOA_MODE or 'local')
  --refs MODEL1,MODEL2,...   Reference models (mode-specific defaults)
  --agg MODEL                Aggregator model (mode-specific default)
  --max-conc N               Max parallel calls (default: 4)
  --timeout N                Per-request timeout in seconds (default: 120)
  --k2                       Enable K2-Backbone dynamic routing
  --task-type TYPE           Task type for K2 routing
  --budget {quality_first,balanced,cost_first}  Budget mode for K2
  --debug                    Enable verbose debug logging
  --list-models              List available models and exit
```

### List available models

```bash
# Cloud models
python scripts/local_moa.py --mode cloud --list-models

# Local models
python scripts/local_moa.py --mode local --list-models
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOA_MODE` | `local` | `local` or `cloud` |
| `OLLAMA_URL` | `http://127.0.0.1:11434/v1/chat/completions` | Local endpoint |
| `OLLAMA_TAGS_URL` | `http://127.0.0.1:11434/api/tags` | Local model list |
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` | Cloud base URL |
| `OLLAMA_API_KEY` | — | Cloud API key |
| `MOA_REF_MAX_TOKENS` | `8000` | Max tokens per reference |
| `MOA_AGG_MAX_TOKENS` | `16000` | Max tokens for aggregator |
| `MOA_MAX_CONCURRENCY` | `4` | Max parallel calls |

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

## Integration: Standalone Multi-Agent Platform (Nexys)

The MoA pipeline can be embedded in a **standalone platform** where each agent runs its own MoA configuration. See the working implementation:

- **Repo:** https://github.com/0x-wzw/nexys
- **Adapter:** `unified_platform/adapters/moa_enabled_agent.py`
- **Demo:** `test_moa_platform.py`

### Platform Architecture

```
┌─────────────────────────────────────────────────────┐
│  MoAEnabledAgentService (IAgentService)               │
│  Manages N agents, each with own MoA engine         │
├─────────────────────────────────────────────────────┤
│  Agent Alpha  │  Agent Beta  │  Agent Gamma        │
│  ───────────  │  ──────────  │  ───────────        │
│  Local MoA    │  Cloud MoA   │  Local MoA          │
│  Code focus   │  Research    │  Analysis           │
│  Cost-first   │  Quality     │  Balanced           │
└─────────────────────────────────────────────────────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              CONVERGENCE (synthesis)
```

### How It Works

1. **Create platform service:** `MoAEnabledAgentService()`
2. **Spawn agents:** Each agent gets its own `MoAProfile` (mode, models, budget)
3. **Dispatch in parallel:** `asyncio.gather(*[agent.execute(task) for agent in agents])`
4. **Each agent runs MoA:** Its own Layer 1 (references) + Layer 2 (aggregator)
5. **BOUNDARY gate:** Quality check before returning
6. **CONVERGENCE:** Merge all outputs into one coherent answer

### Example

```python
from unified_platform.adapters.moa_enabled_agent import MoAEnabledAgentService
from unified_platform.interfaces import Task, AgentConfig, ResourceLimit, CoordinationStrategy

platform = MoAEnabledAgentService()

# Spawn 3 agents with different MoA configs
await platform.create_agent(AgentConfig(
    id="alpha", capabilities=["code"],
    resources=ResourceLimit(token_limit=200000, memory_mb=1024, timeout_seconds=600),
    metadata={"moa_mode": "local", "moa_refs": ["qwen2.5", "llama3.3"],
              "moa_agg": "llama3.3", "moa_budget": "cost_first"},
))

await platform.create_agent(AgentConfig(
    id="beta", capabilities=["research"],
    resources=ResourceLimit(token_limit=200000, memory_mb=1024, timeout_seconds=600),
    metadata={"moa_mode": "cloud", "moa_refs": ["kimi-k2.6", "gemma4:31b"],
              "moa_agg": "kimi-k2.6", "moa_budget": "quality_first"},
))

# Dispatch same task to all in parallel
task = Task(objective="Explain REST vs GraphQL", context={}, constraints={})
results = await asyncio.gather(*[
    platform.dispatch_task(aid, task) for aid in ["alpha", "beta"]
])

# Converge
converged = await platform.coordinate(
    agent_ids=["alpha", "beta"],
    strategy=CoordinationStrategy.CONSENSUS,
    objective=task.objective,
)
```

## When to Use

- Complex math, coding, or architecture problems where a single model may hallucinate
- Offline or air-gapped environments (local mode)
- Privacy-sensitive workloads (local mode)
- Cost-constrained scenarios (local mode)
- No local GPU available (cloud mode)

## When NOT to Use

- Simple questions (single model is faster and sufficient)
- Speed-critical applications (MoA is inherently slower)
- When you have access to Hermes built-in MoA and maximum quality is the priority

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Ollama not reachable` | Run `ollama serve` |
| `Model 'X' not found` | Run `ollama pull X` |
| `OLLAMA_API_KEY not found` | Add to `~/.hermes/.env` or export |
| High memory usage | Lower `--max-conc` |
| Slow on CPU | Normal — use cloud mode or fewer models |

## License

MIT
