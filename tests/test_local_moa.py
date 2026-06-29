"""Tests for local_moa.py — unit tests that don't require Ollama running."""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from local_moa import (
    __version__,
    _load_api_key,
    _resolve_available_models,
    mixture_of_agents_local,
    get_k2_routed_models,
    ollama_chat,
    run_reference_layer,
    run_aggregator,
    MoAModelError,
)


class _CtxResp:
    """Minimal stand-in for an aiohttp response used as an async context manager."""

    def __init__(self, status=200, json_data=None, text_data="", headers=None, exc=None):
        self.status = status
        self._json = json_data or {}
        self._text = text_data
        self.headers = headers or {}
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=self.status, message="err"
            )


class _FakeSession:
    """Returns queued _CtxResp objects on successive .post() calls (last one repeats)."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


class TestVersion:
    def test_version_string(self):
        assert isinstance(__version__, str)
        assert __version__.count(".") >= 1  # semver-ish

    def test_version_not_empty(self):
        assert __version__


class TestModelResolution:
    """Test dynamic model/endpoint resolution based on MODE."""

    def test_local_defaults(self):
        import local_moa
        old = local_moa.MODE
        local_moa.MODE = "local"
        refs = local_moa._get_reference_models()
        agg = local_moa._get_aggregator_model()
        url = local_moa._get_ollama_url()
        assert "llama3.3" in refs
        assert agg == "llama3.3"
        assert "127.0.0.1" in url
        local_moa.MODE = old

    def test_cloud_defaults(self):
        import local_moa
        old = local_moa.MODE
        local_moa.MODE = "cloud"
        refs = local_moa._get_reference_models()
        agg = local_moa._get_aggregator_model()
        url = local_moa._get_ollama_url()
        assert "qwen3-coder:480b" in refs
        assert agg == "deepseek-v4-flash"
        assert "ollama.com" in url
        local_moa.MODE = old


class TestResolveAvailableModels:
    """Test local model name matching logic."""

    def test_exact_match(self):
        result = _resolve_available_models(["llama3.3"], ["llama3.3"])
        assert result == ["llama3.3"]

    def test_tag_variant_match(self):
        result = _resolve_available_models(["llama3.3"], ["llama3.3:latest"])
        assert result == ["llama3.3"]

    def test_no_match(self):
        result = _resolve_available_models(["nonexistent"], ["llama3.3"])
        assert result == []

    def test_mixed(self):
        result = _resolve_available_models(
            ["llama3.3", "mistral", "nonexistent"],
            ["llama3.3:latest", "mistral:7b"],
        )
        assert "llama3.3" in result
        assert "mistral" in result
        assert "nonexistent" not in result


class TestApiKeyLoading:
    """Test API key resolution from multiple sources."""

    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "env_key_123")
        # Also need to make sure ~/.hermes/.env doesn't override
        monkeypatch.setattr(os.path, "expanduser", lambda p: "/nonexistent")
        assert _load_api_key() == "env_key_123"

    def test_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda p: "/nonexistent")
        assert _load_api_key() == ""


class TestK2Fallback:
    """Test K2-Backbone graceful fallback."""

    def test_k2_unavailable_returns_defaults(self):
        import local_moa
        old_available = local_moa._K2_AVAILABLE
        local_moa._K2_AVAILABLE = False
        refs, agg = get_k2_routed_models()
        assert refs == local_moa.REFERENCE_MODELS
        assert agg == local_moa.AGGREGATOR_MODEL
        local_moa._K2_AVAILABLE = old_available


class TestOllamaChat:
    """Test the core API call: success, retries, error handling."""

    @pytest.fixture(autouse=True)
    def _local_mode(self):
        import local_moa
        old = local_moa.MODE
        local_moa.MODE = "local"
        yield
        local_moa.MODE = old

    @pytest.mark.asyncio
    async def test_success(self):
        payload = {"choices": [{"message": {"content": "hello world"}}]}
        session = _FakeSession(_CtxResp(status=200, json_data=payload))
        out = await ollama_chat(session, "model-x", [{"role": "user", "content": "hi"}])
        assert out == "hello world"

    @pytest.mark.asyncio
    async def test_429_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        payload = {"choices": [{"message": {"content": "after retry"}}]}
        session = _FakeSession(
            _CtxResp(status=429, headers={"Retry-After": "0"}),
            _CtxResp(status=200, json_data=payload),
        )
        out = await ollama_chat(session, "model-x", [{"role": "user", "content": "hi"}])
        assert out == "after retry"
        assert session.calls == 2

    @pytest.mark.asyncio
    async def test_4xx_returns_error_string(self):
        session = _FakeSession(_CtxResp(status=401, text_data="unauthorized"))
        out = await ollama_chat(session, "model-x", [{"role": "user", "content": "hi"}])
        assert out.startswith("[ERROR")
        assert "401" in out

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        session = _FakeSession(_CtxResp(exc=aiohttp.ClientConnectionError("boom")))
        with pytest.raises(MoAModelError):
            await ollama_chat(session, "model-x", [{"role": "user", "content": "hi"}])


class TestReferenceLayer:
    """Test that the reference layer keeps only valid responses."""

    @pytest.mark.asyncio
    async def test_filters_errors_and_exceptions(self):
        import local_moa

        async def fake_chat(session, model, messages, **kwargs):
            if model == "good":
                return "valid answer"
            if model == "errstr":
                return "[ERROR: errstr HTTP 500]"
            raise MoAModelError("boom", 3, RuntimeError("net"))

        with patch.object(local_moa, "ollama_chat", side_effect=fake_chat):
            result = await run_reference_layer(
                session=MagicMock(),
                user_prompt="q",
                models=["good", "errstr", "boom"],
            )
        assert result == ["valid answer"]


class TestAggregator:
    """Test aggregator prompt assembly and pass-through."""

    @pytest.mark.asyncio
    async def test_builds_prompt_and_returns(self):
        import local_moa
        captured = {}

        async def fake_chat(session, model, messages, **kwargs):
            captured["model"] = model
            captured["messages"] = messages
            return "SYNTHESIZED"

        with patch.object(local_moa, "ollama_chat", side_effect=fake_chat):
            out = await run_aggregator(
                session=MagicMock(),
                user_prompt="the question",
                reference_responses=["alpha", "beta"],
                aggregator_model="agg-model",
            )
        assert out == "SYNTHESIZED"
        assert captured["model"] == "agg-model"
        system_prompt = captured["messages"][0]["content"]
        assert "alpha" in system_prompt and "beta" in system_prompt
        assert captured["messages"][1]["content"] == "the question"


class TestMixtureOfAgents:
    """Integration-level tests with mocked API calls."""

    @pytest.mark.asyncio
    async def test_missing_api_key_cloud_mode(self, monkeypatch):
        import local_moa
        old_mode = local_moa.MODE
        old_key = local_moa.API_KEY
        local_moa.MODE = "cloud"
        local_moa.API_KEY = ""

        result = await mixture_of_agents_local("test prompt")
        assert result["success"] is False
        assert result["degraded"] is False
        assert "OLLAMA_API_KEY" in result["response"]
        assert result["error"] == "Missing OLLAMA_API_KEY"

        local_moa.MODE = old_mode
        local_moa.API_KEY = old_key

    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        import local_moa
        old_mode = local_moa.MODE
        local_moa.MODE = "local"
        monkeypatch.setattr(local_moa, "_check_local_ollama", lambda: ["m1", "m2"])
        monkeypatch.setattr(
            local_moa, "run_reference_layer", AsyncMock(return_value=["ref one", "ref two longer"])
        )
        monkeypatch.setattr(local_moa, "run_aggregator", AsyncMock(return_value="FINAL ANSWER"))

        result = await mixture_of_agents_local("q", reference_models=["m1", "m2"], aggregator_model="agg")
        assert result["success"] is True
        assert result["degraded"] is False
        assert result["response"] == "FINAL ANSWER"
        assert result["error"] is None
        local_moa.MODE = old_mode

    @pytest.mark.asyncio
    async def test_aggregator_failure_degrades(self, monkeypatch):
        import local_moa
        old_mode = local_moa.MODE
        local_moa.MODE = "local"
        monkeypatch.setattr(local_moa, "_check_local_ollama", lambda: ["m1", "m2"])
        monkeypatch.setattr(
            local_moa, "run_reference_layer", AsyncMock(return_value=["short", "the longest reference"])
        )
        monkeypatch.setattr(
            local_moa, "run_aggregator", AsyncMock(return_value="[ERROR: agg HTTP 500]")
        )

        result = await mixture_of_agents_local("q", reference_models=["m1", "m2"], aggregator_model="agg")
        assert result["success"] is True
        assert result["degraded"] is True
        assert result["response"] == "the longest reference"  # longest reference used as fallback
        assert "Aggregator failed" in result["error"]
        local_moa.MODE = old_mode

    @pytest.mark.asyncio
    async def test_all_references_fail(self, monkeypatch):
        import local_moa
        old_mode = local_moa.MODE
        local_moa.MODE = "local"
        monkeypatch.setattr(local_moa, "_check_local_ollama", lambda: ["m1", "m2"])
        monkeypatch.setattr(local_moa, "run_reference_layer", AsyncMock(return_value=[]))

        result = await mixture_of_agents_local("q", reference_models=["m1", "m2"])
        assert result["success"] is False
        assert result["degraded"] is False
        assert result["error"] == "All reference models failed"
        local_moa.MODE = old_mode
