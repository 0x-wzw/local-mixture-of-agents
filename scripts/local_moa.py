#!/usr/bin/env python3
"""
Mixture-of-Agents (MoA) using Ollama Cloud
Architecture: Layer 1 (parallel reference models) → Layer 2 (aggregator synthesis)

Targets the Ollama Cloud OpenAI-compatible endpoint (https://ollama.com/v1).
API key is read from the OLLAMA_API_KEY environment variable (loaded from ~/.hermes/.env
when running inside Hermes, or set in your shell for standalone use).
"""

import asyncio
import json
import os
import aiohttp
import sys
from typing import List, Optional

# ── Endpoint ──────────────────────────────────────────────────────────────
# Override at runtime via OLLAMA_BASE_URL env var if needed.
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1") + "/chat/completions"

# ── API Key ───────────────────────────────────────────────────────────────
# Try multiple sources: env var → ~/.hermes/.env → ~/.ollama/token
def _load_api_key() -> str:
    # 1. Already in environment
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        return key

    # 2. Load from ~/.hermes/.env (Hermes home)
    hermes_env = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(hermes_env):
        with open(hermes_env, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OLLAMA_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    # 3. Load from ~/.ollama/token (ollama CLI token file)
    token_file = os.path.expanduser("~/.ollama/token")
    if os.path.exists(token_file):
        with open(token_file, encoding="utf-8") as f:
            return f.read().strip()

    return ""

API_KEY = _load_api_key()

# ── Layer 1: Reference models (diverse cloud models for best coverage) ────
# Chosen for architectural diversity: Qwen (coding), Kimi (reasoning),
# DeepSeek (general), Gemma (analytical). All available on Ollama Cloud.
REFERENCE_MODELS = [
    "qwen3-coder:480b",     # coding / structured reasoning
    "kimi-k2.6",            # general reasoning
    "deepseek-v4-flash",     # fast general-purpose
    "gemma4:31b",           # analytical / concise
]

# ── Layer 2: Aggregator (your strongest cloud model) ──────────────────────
AGGREGATOR_MODEL = "deepseek-v4-flash"  # fast, high-quality synthesis

# ── Prompts ────────────────────────────────────────────────────────────────
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


async def ollama_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: List[dict],
    temperature: float = 0.6,
    timeout: int = 120
) -> str:
    """Call Ollama Cloud API with retry logic."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            async with session.post(
                OLLAMA_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"[ERROR: {model} failed after 3 attempts: {str(e)}]"


async def run_reference_layer(
    session: aiohttp.ClientSession,
    user_prompt: str,
    models: Optional[List[str]] = None
) -> List[str]:
    """Layer 1: Parallel reference model calls."""
    models = models or REFERENCE_MODELS

    tasks = [
        ollama_chat(
            session=session,
            model=m,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.6  # encourage diversity
        )
        for m in models
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failures
    valid: List[str] = []
    for model, response in zip(models, responses):
        if isinstance(response, Exception):
            print(f"❌ {model}: {response}", file=sys.stderr)
        elif isinstance(response, str) and response.startswith("[ERROR"):
            print(f"⚠️  {model}: Failed", file=sys.stderr)
        elif isinstance(response, str):
            print(f"✅ {model}: {len(response)} chars", file=sys.stderr)
            valid.append(response)

    return valid


async def run_aggregator(
    session: aiohttp.ClientSession,
    user_prompt: str,
    reference_responses: List[str],
    aggregator_model: Optional[str] = None
) -> str:
    """Layer 2: Synthesize all reference outputs into final answer."""
    model = aggregator_model or AGGREGATOR_MODEL

    # Build the aggregated system prompt
    response_text = "\n".join([
        f"{i+1}. {r[:500]}..." if len(r) > 500 else f"{i+1}. {r}"
        for i, r in enumerate(reference_responses)
    ])

    system_prompt = f"{AGGREGATOR_SYSTEM_PROMPT}\n\n{response_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return await ollama_chat(
        session=session,
        model=model,
        messages=messages,
        temperature=0.4  # focused synthesis
    )


async def mixture_of_agents_local(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None
) -> dict:
    """
    Run Ollama Cloud MoA pipeline.

    Returns:
        {
            "success": bool,
            "response": str,
            "models_used": {"reference": [...], "aggregator": str},
            "reference_count": int,
            "processing_time": float
        }
    """
    import time
    start = time.time()

    if not API_KEY:
        return {
            "success": False,
            "response": "OLLAMA_API_KEY not found. Set it in ~/.hermes/.env or as an environment variable.",
            "models_used": {"reference": reference_models or REFERENCE_MODELS, "aggregator": None},
            "reference_count": 0,
            "processing_time": 0.0
        }

    async with aiohttp.ClientSession() as session:
        # Layer 1: Parallel references
        print(f"🔹 Layer 1: Querying {len(reference_models or REFERENCE_MODELS)} reference models via Ollama Cloud...", file=sys.stderr)
        references = await run_reference_layer(session, user_prompt, reference_models)

        if len(references) < 1:
            return {
                "success": False,
                "response": "All reference models failed. Cannot proceed with aggregation.",
                "models_used": {"reference": reference_models or REFERENCE_MODELS, "aggregator": None},
                "reference_count": 0,
                "processing_time": time.time() - start
            }

        # Layer 2: Aggregation
        print(f"🔹 Layer 2: Aggregating {len(references)} responses via {aggregator_model or AGGREGATOR_MODEL}...", file=sys.stderr)
        final = await run_aggregator(session, user_prompt, references, aggregator_model)

        elapsed = time.time() - start

        return {
            "success": True,
            "response": final,
            "models_used": {
                "reference": reference_models or REFERENCE_MODELS,
                "aggregator": aggregator_model or AGGREGATOR_MODEL
            },
            "reference_count": len(references),
            "processing_time": round(elapsed, 2)
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python local_moa.py '<user prompt>'")
        print("\nExamples:")
        print('  python local_moa.py "Explain quantum entanglement"')
        print('  python local_moa.py "Trade-offs between REST and GraphQL"')
        sys.exit(1)

    prompt = sys.argv[1]
    result = asyncio.run(mixture_of_agents_local(prompt))
    print(json.dumps(result, indent=2, ensure_ascii=False))