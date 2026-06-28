#!/usr/bin/env python3
"""
Mixture-of-Agents (MoA) — Dual-Mode: Local Ollama + Ollama Cloud
=================================================================
Architecture: Layer 1 (parallel reference models) → Layer 2 (aggregator synthesis)

Modes:
  • LOCAL (default): Uses http://127.0.0.1:11434 — no auth, fully sovereign.
  • CLOUD: Uses https://ollama.com/v1 — requires OLLAMA_API_KEY.

Switch modes via env var: MOA_MODE=local (default) | MOA_MODE=cloud

Optional K2-Backbone integration for dynamic model selection via capability matrix.
Falls back gracefully when K2 is not installed.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp

# ── Logging ────────────────────────────────────────────────────────────────
logger = logging.getLogger("local_moa")


def _setup_logging(verbose: bool = False) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ── K2-Backbone Integration (optional) ──────────────────────────────────────
try:
    from k2_backbone.router.moa_router import MoARouter  # type: ignore

    _K2_AVAILABLE = True
except ImportError:
    _K2_AVAILABLE = False

_K2_ROUTER: Optional[object] = None


def get_k2_routed_models(
    task_type: str = "analysis",
    budget: str = "balanced",
    diversity: int = 4,
) -> tuple[List[str], str]:
    """
    Query K2 MoARouter for dynamically selected models.
    Falls back to hardcoded defaults when K2 unavailable.
    """
    global _K2_ROUTER

    if not _K2_AVAILABLE:
        logger.info("K2-Backbone not available; using hardcoded defaults")
        return REFERENCE_MODELS, AGGREGATOR_MODEL

    if _K2_ROUTER is None:
        _K2_ROUTER = MoARouter()  # type: ignore[operator]

    refs = _K2_ROUTER.select_reference_models(  # type: ignore[union-attr]
        task_type=task_type, count=diversity, budget=budget
    )
    agg = _K2_ROUTER.select_aggregator(task_type=task_type, budget=budget)  # type: ignore[union-attr]

    logger.info("K2-Routed refs: %s | agg: %s", ", ".join(refs), agg)
    return refs, agg


# ── Mode & Endpoint ────────────────────────────────────────────────────────
MODE = os.environ.get("MOA_MODE", "local").strip().lower()
if MODE not in ("local", "cloud"):
    raise ValueError(f"MOA_MODE must be 'local' or 'cloud', got '{MODE}'")

LOCAL_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
LOCAL_TAGS_URL = os.environ.get("OLLAMA_TAGS_URL", "http://127.0.0.1:11434/api/tags")
CLOUD_BASE = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
CLOUD_URL = f"{CLOUD_BASE}/chat/completions"

OLLAMA_URL = CLOUD_URL if MODE == "cloud" else LOCAL_URL

# ── API Key (cloud only) ──────────────────────────────────────────────────
def _load_api_key() -> str:
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        return key
    hermes_env = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(hermes_env):
        with open(hermes_env, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OLLAMA_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    token_file = os.path.expanduser("~/.ollama/token")
    if os.path.exists(token_file):
        with open(token_file, encoding="utf-8") as f:
            return f.read().strip()
    return ""


API_KEY = _load_api_key()

# ── Defaults by mode ───────────────────────────────────────────────────────
if MODE == "cloud":
    REFERENCE_MODELS = [
        "qwen3-coder:480b",
        "kimi-k2.6",
        "deepseek-v4-flash",
        "gemma4:31b",
    ]
    AGGREGATOR_MODEL = "deepseek-v4-flash"
else:
    REFERENCE_MODELS = [
        "llama3.3",
        "qwen2.5",
        "mistral",
        "phi4",
    ]
    AGGREGATOR_MODEL = "llama3.3"

# ── Token budgets ──────────────────────────────────────────────────────────
REFERENCE_MAX_TOKENS = int(os.environ.get("MOA_REF_MAX_TOKENS", "8000"))
AGGREGATOR_MAX_TOKENS = int(os.environ.get("MOA_AGG_MAX_TOKENS", "16000"))

# ── Concurrency ────────────────────────────────────────────────────────────
MAX_CONCURRENCY = int(os.environ.get("MOA_MAX_CONCURRENCY", "4"))

# ── Temperatures ───────────────────────────────────────────────────────────
REFERENCE_TEMPERATURE = 0.6
AGGREGATOR_TEMPERATURE = 0.4

# ── Prompts ─────────────────────────────────────────────────────────────────
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


# ── Helpers ─────────────────────────────────────────────────────────────────
def _auth_headers() -> Dict[str, str]:
    if MODE == "cloud":
        return {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
    return {"Content-Type": "application/json"}


def _check_local_ollama() -> List[str]:
    """Pre-flight: verify Ollama is reachable and return available models."""
    try:
        import urllib.request

        req = urllib.request.Request(LOCAL_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {LOCAL_TAGS_URL}: {e}")


def _resolve_available_models(desired: List[str], available: List[str]) -> List[str]:
    """Filter desired models to those actually pulled locally."""
    resolved = []
    for m in desired:
        # Ollama model names may include tags (e.g., "llama3.3:latest")
        if m in available or any(a.startswith(m + ":") for a in available):
            resolved.append(m)
        else:
            logger.warning("Model '%s' not found locally — skipping", m)
    return resolved


# ── Core API call ───────────────────────────────────────────────────────────
async def ollama_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: List[dict],
    temperature: float = 0.6,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = _auth_headers()

    for attempt in range(3):
        try:
            async with session.post(
                OLLAMA_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug("Attempt %d for %s failed: %s", attempt + 1, model, e)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"[ERROR: {model} failed after 3 attempts: {e}]"
    return f"[ERROR: {model} unreachable]"


# ── Layer 1: Reference ──────────────────────────────────────────────────────
async def run_reference_layer(
    session: aiohttp.ClientSession,
    user_prompt: str,
    models: Optional[List[str]] = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> List[str]:
    models = models or REFERENCE_MODELS
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call(m: str) -> str:
        async with semaphore:
            return await ollama_chat(
                session=session,
                model=m,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=REFERENCE_TEMPERATURE,
                max_tokens=REFERENCE_MAX_TOKENS,
            )

    tasks = [_call(m) for m in models]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    valid: List[str] = []
    for model, response in zip(models, responses):
        if isinstance(response, Exception):
            logger.error("❌ %s: %s", model, response)
        elif isinstance(response, str) and response.startswith("[ERROR"):
            logger.warning("⚠️ %s: Failed", model)
        elif isinstance(response, str):
            logger.info("✅ %s: %d chars", model, len(response))
            valid.append(response)

    return valid


# ── Layer 2: Aggregator ───────────────────────────────────────────────────
async def run_aggregator(
    session: aiohttp.ClientSession,
    user_prompt: str,
    reference_responses: List[str],
    aggregator_model: Optional[str] = None,
) -> str:
    model = aggregator_model or AGGREGATOR_MODEL

    # Pass full responses (no truncation) — token budget governs length
    response_text = "\n\n".join(
        f"--- Response {i + 1} ---\n{r}"
        for i, r in enumerate(reference_responses)
    )

    system_prompt = f"{AGGREGATOR_SYSTEM_PROMPT}\n\n{response_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return await ollama_chat(
        session=session,
        model=model,
        messages=messages,
        temperature=AGGREGATOR_TEMPERATURE,
        max_tokens=AGGREGATOR_MAX_TOKENS,
    )


# ── Main pipeline ──────────────────────────────────────────────────────────
async def mixture_of_agents_local(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None,
    max_concurrency: int = MAX_CONCURRENCY,
    use_k2_routing: bool = False,
    task_type: str = "analysis",
    budget: str = "balanced",
) -> Dict[str, Any]:
    """
    Run MoA pipeline (local or cloud).

    Returns:
        {
            "success": bool,
            "response": str,
            "models_used": {"reference": [...], "aggregator": str},
            "reference_count": int,
            "processing_time": float,
            "error": str | None,
        }
    """
    start = time.time()

    # Pre-flight checks
    if MODE == "cloud":
        if not API_KEY:
            return {
                "success": False,
                "response": "OLLAMA_API_KEY not found. Set it in ~/.hermes/.env or as an environment variable.",
                "models_used": {"reference": reference_models or REFERENCE_MODELS, "aggregator": None},
                "reference_count": 0,
                "processing_time": 0.0,
                "error": "Missing OLLAMA_API_KEY",
            }
    else:
        try:
            available = _check_local_ollama()
        except RuntimeError as e:
            return {
                "success": False,
                "response": str(e),
                "models_used": {"reference": [], "aggregator": None},
                "reference_count": 0,
                "processing_time": 0.0,
                "error": str(e),
            }
        if reference_models is None:
            reference_models = REFERENCE_MODELS
        resolved_refs = _resolve_available_models(reference_models, available)
        if len(resolved_refs) < len(reference_models):
            logger.info("Resolved %d/%d models", len(resolved_refs), len(reference_models))
        reference_models = resolved_refs

    # K2 routing
    if use_k2_routing and reference_models is None:
        ref_models, agg_model = get_k2_routed_models(task_type=task_type, budget=budget, diversity=4)
        reference_models = ref_models
        aggregator_model = aggregator_model or agg_model

    resolved_refs = reference_models if reference_models is not None else REFERENCE_MODELS
    agg_model_resolved = aggregator_model if aggregator_model is not None else AGGREGATOR_MODEL

    async with aiohttp.ClientSession() as session:
        logger.info("Layer 1: Querying %d reference models (max_concurrency=%d)...", len(resolved_refs), max_concurrency)
        references = await run_reference_layer(session, user_prompt, resolved_refs, max_concurrency)

        if len(references) < 1:
            elapsed = round(time.time() - start, 2)
            return {
                "success": False,
                "response": "All reference models failed. Cannot proceed with aggregation.",
                "models_used": {"reference": resolved_refs, "aggregator": None},
                "reference_count": 0,
                "processing_time": elapsed,
                "error": "All reference models failed",
            }

        logger.info("Layer 2: Aggregating %d responses via %s...", len(references), agg_model_resolved)
        final = await run_aggregator(session, user_prompt, references, agg_model_resolved)

        if final.startswith("[ERROR"):
            elapsed = round(time.time() - start, 2)
            return {
                "success": False,
                "response": final,
                "models_used": {"reference": resolved_refs, "aggregator": agg_model_resolved},
                "reference_count": len(references),
                "processing_time": elapsed,
                "error": f"Aggregator failed: {final}",
            }

        elapsed = round(time.time() - start, 2)
        logger.info("MoA completed in %ss", elapsed)
        return {
            "success": True,
            "response": final,
            "models_used": {"reference": resolved_refs, "aggregator": agg_model_resolved},
            "reference_count": len(references),
            "processing_time": elapsed,
            "error": None,
        }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixture-of-Agents pipeline via Ollama (local or cloud)",
    )
    parser.add_argument("prompt", help="The user prompt / question")
    parser.add_argument("--mode", choices=["local", "cloud"], default=os.environ.get("MOA_MODE", "local"),
                        help="Execution mode (default: env MOA_MODE or 'local')")
    parser.add_argument("--refs", default=None, help="Comma-separated reference models (default: mode-specific)")
    parser.add_argument("--agg", default=None, help="Aggregator model (default: mode-specific)")
    parser.add_argument("--max-conc", type=int, default=None, help="Max parallel calls (default: 4)")
    parser.add_argument("--k2", action="store_true", help="Enable K2-Backbone dynamic routing")
    parser.add_argument("--task-type", default="analysis", help="Task type for K2 routing")
    parser.add_argument("--budget", default="balanced", choices=["quality_first", "balanced", "cost_first"],
                        help="Budget mode for K2 routing")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")

    args = parser.parse_args()
    _setup_logging(args.debug)

    # Override mode via CLI if provided
    if args.mode != MODE:
        os.environ["MOA_MODE"] = args.mode
        logger.info("Switched mode to: %s", args.mode)

    ref_models = args.refs.split(",") if args.refs else None
    max_conc = args.max_conc or MAX_CONCURRENCY

    result = asyncio.run(mixture_of_agents_local(
        user_prompt=args.prompt,
        reference_models=ref_models,
        aggregator_model=args.agg,
        max_concurrency=max_conc,
        use_k2_routing=args.k2,
        task_type=args.task_type,
        budget=args.budget,
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
