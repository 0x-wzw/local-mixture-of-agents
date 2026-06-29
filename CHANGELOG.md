# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

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
