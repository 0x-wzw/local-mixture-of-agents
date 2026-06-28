#!/usr/bin/env python3
"""
Local Mixture-of-Agents (MoA) using Ollama
Architecture: Layer 1 (parallel reference models) → Layer 2 (aggregator synthesis)

Based on: "Mixture-of-Agents Enhances Large Language Model Capabilities"
by Junlin Wang et al. (arXiv:2406.04692)
"""

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from typing import Dict, Any, List, Optional

import aiohttp

# ── Configuration ──────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://127.0.0.1:11434/api/tags")

# Layer 1: Reference models (diverse local models for best coverage)
REFERENCE_MODELS = [
    "llama3.3",      # general reasoning
    "qwen2.5",       # math / coding
    "mistral",       # concise / structured
    "phi4",          # analytical
]

# Layer 2: Aggregator (your strongest local model)
AGGREGATOR_MODEL = "llama3.3"

# Token budgets
REFERENCE_MAX_TOKENS = int(os.getenv("MOA_REF_MAX_TOKENS", "8000"))
AGGREGATOR_MAX_TOKENS = int(os.getenv("MOA_AGG_MAX_TOKENS", "16000"))

# Concurrency limit (prevent memory thrashing on CPU-only systems)
MAX_CONCURRENCY = int(os.getenv("MOA_MAX_CONCURRENCY", "4"))

# Temperatures
REFERENCE_TEMPERATURE = 0.6
AGGREGATOR_TEMPERATURE = 0.4

# Prompts
AGGREGATOR_SYSTEM_PROMPT = (
    "You have been provided with a set of responses from various open-source models "
    "to the latest user query. Your task is to synthesize these responses into a single, "
    "high-quality response. It is crucial to critically evaluate the information provided "
    "in these responses, recognizing that some of it may be biased or incorrect. Your response "
    "should not simply replicate the given answers but should offer a refined, accurate, and "
    "comprehensive reply to the instruction. Ensure your response is well-structured, coherent, "
    "and adheres to the highest standards of accuracy and reliability.\n\n"
    "Responses from models:"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ── Pre-flight checks ──────────────────────────────────────────
def _check_ollama_connectivity(timeout: int = 10) -> List[str]:
    """Probe Ollama and return available model names."""
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_TAGS_URL}: {e}")


def _resolve_available_models(desired: List[str], available: List[str]) -> List[str]:
    """Match desired models against those actually pulled."""
    available_base = {m.split(":")[0]: m for m in available}
    resolved = []
    for d in desired:
        if d in available_base:
            resolved.append(available_base[d])
        else:
            logger.warning("Model '%s' not found in Ollama — skipping", d)
    return resolved


# ── Core API caller ──────────────────────────────────────────
async def ollama_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: List[dict],
    temperature: float = 0.6,
    max_tokens: int = 8000,
    timeout: int = 120,
) -> str:
    """Call Ollama API with retry logic and exponential backoff."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    for attempt in range(3):
        try:
            logger.info("Querying %s (attempt %s/3)", model, attempt + 1)
            async with session.post(
                OLLAMA_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        except aiohttp.ClientResponseError as e:
            logger.warning("%s HTTP %s (attempt %s): %s", model, e.status, attempt + 1, e.message)
            if e.status == 404:
                return f"[ERROR: {model} not found (HTTP 404)]"
        except asyncio.TimeoutError:
            logger.warning("%s timeout (attempt %s)", model, attempt + 1)
        except Exception as e:
            logger.warning("%s error (attempt %s): %s", model, attempt + 1, e)

        if attempt < 2:
            sleep_time = min(2 ** (attempt + 1), 60)
            logger.info("Retrying %s in %ss...", model, sleep_time)
            await asyncio.sleep(sleep_time)

    return f"[ERROR: {model} failed after 3 attempts]"


# ── Layer 1: Reference models ─────────────────────────────────
async def run_reference_layer(
    session: aiohttp.ClientSession,
    user_prompt: str,
    models: Optional[List[str]] = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> List[str]:
    """Run reference models in parallel with bounded concurrency."""
    models = models or REFERENCE_MODELS
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call_with_limit(m: str) -> str:
        async with semaphore:
            return await ollama_chat(
                session=session,
                model=m,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=REFERENCE_TEMPERATURE,
                max_tokens=REFERENCE_MAX_TOKENS,
            )

    tasks = [_call_with_limit(m) for m in models]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    valid: List[str] = []
    for model, response in zip(models, responses):
        if isinstance(response, Exception):
            logger.error("❌ %s: %s", model, response)
        elif response.startswith("[ERROR"):
            logger.warning("⚠️  %s: Failed — %s", model, response)
        else:
            logger.info("✅ %s: %s chars", model, len(response))
            valid.append(response)

    return valid


# ── Layer 2: Aggregator ──────────────────────────────────────
async def run_aggregator(
    session: aiohttp.ClientSession,
    user_prompt: str,
    reference_responses: List[str],
    aggregator_model: Optional[str] = None,
) -> str:
    """Synthesize reference outputs into final answer."""
    model = aggregator_model or AGGREGATOR_MODEL

    # Build aggregator prompt with full responses (respects token budget via max_tokens)
    response_text = "\n\n".join([
        f"--- Response {i + 1} ---\n{r}"
        for i, r in enumerate(reference_responses)
    ])

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


# ── Main pipeline ──────────────────────────────────────────────
async def mixture_of_agents_local(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> Dict[str, Any]:
    """
    Run local MoA pipeline.

    Args:
        user_prompt: The complex query to process.
        reference_models: Override default reference models.
        aggregator_model: Override default aggregator model.
        max_concurrency: Max parallel reference calls (default: 4).

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
    ref_models = reference_models or REFERENCE_MODELS
    agg_model = aggregator_model or AGGREGATOR_MODEL

    try:
        # Pre-flight: verify Ollama and resolve available models
        logger.info("Pre-flight: checking Ollama connectivity...")
        available = _check_ollama_connectivity()
        resolved_refs = _resolve_available_models(ref_models, available)
        resolved_agg = _resolve_available_models([agg_model], available)

        if not resolved_refs:
            raise RuntimeError(
                f"None of the requested reference models are available. "
                f"Pulled models: {[m.split(':')[0] for m in available]}"
            )

        if not resolved_agg:
            raise RuntimeError(
                f"Aggregator model '{agg_model}' not available. "
                f"Pulled models: {[m.split(':')[0] for m in available]}"
            )

        agg_model_resolved = resolved_agg[0]
        logger.info("Using %s/%s reference models", len(resolved_refs), len(ref_models))
        logger.info("Aggregator: %s", agg_model_resolved)

        async with aiohttp.ClientSession() as session:
            # Layer 1: Parallel references
            logger.info("Layer 1: Querying reference models (max_concurrency=%s)...", max_concurrency)
            references = await run_reference_layer(
                session, user_prompt, resolved_refs, max_concurrency
            )

            if len(references) < 1:
                raise RuntimeError(
                    "All reference models failed. Cannot proceed with aggregation."
                )

            # Layer 2: Aggregation
            logger.info("Layer 2: Aggregating %s responses via %s...", len(references), agg_model_resolved)
            final = await run_aggregator(session, user_prompt, references, agg_model_resolved)

            if final.startswith("[ERROR"):
                raise RuntimeError(f"Aggregator failed: {final}")

            elapsed = round(time.time() - start, 2)
            logger.info("MoA completed in %ss", elapsed)

            return {
                "success": True,
                "response": final,
                "models_used": {
                    "reference": resolved_refs,
                    "aggregator": agg_model_resolved,
                },
                "reference_count": len(references),
                "processing_time": elapsed,
                "error": None,
            }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        logger.error("MoA failed after %ss: %s", elapsed, e)
        return {
            "success": False,
            "response": "",
            "models_used": {
                "reference": ref_models,
                "aggregator": agg_model,
            },
            "reference_count": 0,
            "processing_time": elapsed,
            "error": str(e),
        }


# ── CLI ────────────────────────────────────────────────────────
def _print_usage():
    print("Usage: python local_moa.py '<user prompt>' [options]")
    print("\nOptions:")
    print("  --refs MODEL1,MODEL2,...   Reference models (default: llama3.3,qwen2.5,mistral,phi4)")
    print("  --agg MODEL                  Aggregator model (default: llama3.3)")
    print("  --max-conc N                 Max parallel calls (default: 4)")
    print("  --debug                      Enable debug logging")
    print("\nExamples:")
    print('  python local_moa.py "Explain quantum entanglement"')
    print('  python local_moa.py "Trade-offs between REST and GraphQL" --refs llama3.3,mistral --agg llama3.3')


async def _cli():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    args = sys.argv[1:]
    prompt = args[0]
    refs_arg = None
    agg_arg = None
    max_conc = MAX_CONCURRENCY

    i = 1
    while i < len(args):
        if args[i] == "--refs" and i + 1 < len(args):
            refs_arg = [m.strip() for m in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--agg" and i + 1 < len(args):
            agg_arg = args[i + 1].strip()
            i += 2
        elif args[i] == "--max-conc" and i + 1 < len(args):
            max_conc = int(args[i + 1])
            i += 2
        elif args[i] == "--debug":
            logging.getLogger().setLevel(logging.DEBUG)
            i += 1
        else:
            i += 1

    result = await mixture_of_agents_local(
        user_prompt=prompt,
        reference_models=refs_arg,
        aggregator_model=agg_arg,
        max_concurrency=max_conc,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    asyncio.run(_cli())
