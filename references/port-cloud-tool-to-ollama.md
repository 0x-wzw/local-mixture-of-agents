# Porting Cloud-Locked Hermes Tools to Local Ollama

**Context:** Several Hermes built-in tools (e.g., `mixture_of_agents`, `image_generate`, `delegate_task` when routing through cloud providers) are hard-wired to cloud APIs (OpenRouter, OpenAI, FAL.ai, etc.). When you need a sovereign, offline, or zero-cost equivalent, the general pattern below applies.

## The Porting Pattern

### 1. Identify the cloud dependency
Read the source (`~/.hermes/hermes-agent/tools/<tool>.py`) and look for:
- Hard-coded API client calls (`_get_openrouter_client()`, `openai.chat.completions.create()`)
- API key requirements (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`)
- Model IDs that are provider-prefixed (`anthropic/claude-opus-4.6`, `google/gemini-2.5-pro`)

### 2. Recreate the architecture, not the implementation
- Preserve the **control flow** (parallel calls → aggregation, sequential chains, retry loops)
- Replace **model IDs** with Ollama-compatible names (strip provider prefixes)
- Replace **cloud API client** with `aiohttp` calls to `http://127.0.0.1:11434/v1/chat/completions`
- Replace **API auth** (headers with keys) with none (Ollama local has no auth)

### 3. Add production hardening that cloud tools get for free
Cloud tools run on managed infra with implicit safeguards. Local tools must add these explicitly:

| Cloud implicit | Local explicit |
|---------------|----------------|
| Managed token limits | Add `max_tokens` to payload |
| Managed concurrency | Add `asyncio.Semaphore` or thread-pool limit |
| Pre-warmed model availability | Add `_check_ollama_connectivity()` probe |
| Structured error codes | Parse HTTP status + raise typed exceptions |
| Observability | Add structured logging (timestamped levels, not raw `print`) |
| Graceful degradation | Handle partial failure (e.g., ≥1 of 4 models succeeds) |

### 4. Common Ollama API shape

```python
payload = {
    "model": "llama3.3",           # Ollama model name, no provider prefix
    "messages": [{"role": "..."}],
    "temperature": 0.6,
    "max_tokens": 8000,            # ADD THIS — not optional locally
    "stream": False,               # Must be False for JSON response
}
```

### 5. Environment variable convention
For any local tool, expose these for configuration without code edits:
- `OLLAMA_URL` — default `http://127.0.0.1:11434/v1/chat/completions`
- `OLLAMA_TAGS_URL` — default `http://127.0.0.1:11434/api/tags`
- Tool-specific caps: `<TOOL>_MAX_TOKENS`, `<TOOL>_MAX_CONCURRENCY`

## Pitfalls

- **No `max_tokens` = runaway generation.** Local models don't have implicit output limits. Always cap.
- **`~` in `sys.path` doesn't expand.** Use `os.path.expanduser()` when constructing paths.
- **Truncation degrades multi-model synthesis.** If aggregating responses, pass full outputs (respecting token budget) rather than arbitrary character truncation. The aggregator needs coherent input.
- **Aggregator failure must be checked.** A cloud API throws an exception; local Ollama may return an error string that your code treats as a valid answer. Check `response.startswith("[ERROR")` or parse HTTP status.
- **Model diversity matters.** If all local models share the same architecture (e.g., all Llama-based), the MoA quality uplift is smaller than mixing Claude + Gemini + GPT families. Pick diverse open families: Llama (Meta), Qwen (Alibaba), Mistral (European), Phi (Microsoft).

## Verification Checklist

Before calling a ported local tool "done":
1. [ ] Ollama reachable (`curl http://127.0.0.1:11434/api/tags`)
2. [ ] Required models pulled (`ollama list` shows them)
3. [ ] Script runs single-model baseline successfully
4. [ ] Multi-model pipeline runs with ≥1 reference succeeding
5. [ ] Aggregator failure returns `success: false` (not an error string as answer)
6. [ ] `--max-conc 1` runs sequentially without deadlock
7. [ ] `--debug` produces readable structured logs
