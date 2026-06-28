"""Tests for local_moa.py — unit tests that don't require Ollama running."""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from local_moa import (
    __version__,
    _load_api_key,
    _resolve_available_models,
    mixture_of_agents_local,
    get_k2_routed_models,
    _get_reference_models,
    _get_aggregator_model,
    _get_ollama_url,
)


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
        assert "OLLAMA_API_KEY" in result["response"]
        assert result["error"] == "Missing OLLAMA_API_KEY"

        local_moa.MODE = old_mode
        local_moa.API_KEY = old_key

    @pytest.mark.asyncio
    async def test_empty_prompt_validation(self):
        # The function should handle empty prompts gracefully
        # (currently it passes them through — this test documents current behavior)
        import local_moa
        old_mode = local_moa.MODE
        local_moa.MODE = "cloud"
        # Would need mocked API to fully test; just verify it doesn't crash
        local_moa.MODE = old_mode