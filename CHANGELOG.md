# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [3.3.0] — 2026-06-30

### Added
- **`--prompt-file` CLI flag** — read the prompt from a file or stdin (`-` for stdin),
  enabling piping and multi-line prompts.
- **`--temperature` alias** for `--ref-temp` — shorter flag for the most common override.
- **`list_available_models()`** async function — query the configured endpoint for
  available models. Added to `__all__` as part of the public API.
- **Per-model attribution** in the aggregator prompt — each reference response is
  tagged with its model name (e.g. `--- Response 1 (model: kimi-k2.6) ---`) so the
  aggregator can critically weight per the system prompt's instruction.
- **Expanded `__all__`** — now exports `list_available_models`, `REFERENCE_TEMPERATURE`,
  `AGGREGATOR_TEMPERATURE`, `MAX_CONCURRENCY`, and `__version__`.

### Fixed
- **K2 routing in cloud mode** — when `use_k2_routing=True` and no explicit models
  are provided, K2 can now override cloud defaults. Previously the cloud pre-flight
  assigned `reference_models` before the K2 guard check, making the `is None`
  condition always `False`.

## [3.2.0] — 2026-06-29

### Added
- **`degraded` field** in the `mixture_of_agents_local()` result schema. It is `True`
  only when the aggregator failed and a reference response was used as a fallback, so
  callers can distinguish a clean synthesis from a degraded one (the field is always
  present on every return path).
- **`pyproject.toml`** — PEP 517/518 build config, `pip install .` support, a
  `local-moa` console entry point, a `dev` extras group, pytest (`asyncio_mode=auto`)
  and ruff configuration.
- **`requirements-dev.txt`** — declares the test/lint toolchain
  (`pytest`, `pytest-asyncio`, `ruff`, `pytest-cov`).
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — runs ruff + pytest with coverage
  across Python 3.9–3.12. Tests are hermetic (network mocked); no running Ollama needed.
- **Expanded test suite** covering `ollama_chat` (success, 429 retry, 4xx error string,
  exhausted-retry `MoAModelError`), `run_reference_layer` failure filtering,
  `run_aggregator` prompt assembly, and the aggregator-fallback `degraded` path.

### Fixed
- Aggregator-failure fallback no longer reports a degraded result as indistinguishable
  from success — the new `degraded` flag makes the contract honest. Documentation in
  `SKILL.md` that claimed the tool "never silently returns an error string" has been
  corrected to describe the actual fallback behavior.

### Documentation
- `SKILL.md`: documented the `--raw`, `--list-models`, `--ref-temp`, and `--agg-temp`
  CLI options, the reference/aggregator temperature defaults (0.6 / 0.4), and the
  `~/.ollama/token` API-key fallback.
- `README.md`: added the `degraded` field to the documented JSON output schema.
