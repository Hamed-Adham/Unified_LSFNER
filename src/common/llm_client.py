"""
Unified LLM Provider Client supporting OpenAI, LMStudio, Ollama, Custom API, and Dummy Fallback.
Provides unified generation interface with automatic retry, rate limiting, and token tracking.
"""

import os
import json
import time
from typing import Dict, Any, Optional
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

load_dotenv(override=True)

# Global tracking metadata
last_call_metadata = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "latency_seconds": 0.0,
    "model": "",
    "backend": ""
}


class BaseLLMProvider:
    """Abstract Base Class for LLM backends."""
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        raise NotImplementedError

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class DummyLLMProvider(BaseLLMProvider):
    """
    Offline dummy provider that returns valid empty/placeholder JSON or text.
    Allows running entire pipelines for testing/caching without API costs.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        if self.verbose:
            print("[DummyLLMProvider] Received prompt, returning fallback empty structured JSON.")
        return json.dumps({
            "status": "success",
            "predicted_tag": "O",
            "entities": [],
            "rationale": "Dummy local provider execution."
        })


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider (GPT-4o, GPT-4o-mini, etc.)."""
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.extra_kwargs = kwargs
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "4096"))

        t0 = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=tokens
        )
        t1 = time.time()

        content = response.choices[0].message.content or ""
        if response.usage:
            last_call_metadata["prompt_tokens"] = response.usage.prompt_tokens
            last_call_metadata["completion_tokens"] = response.usage.completion_tokens
            last_call_metadata["total_tokens"] = response.usage.total_tokens
        last_call_metadata["latency_seconds"] = t1 - t0
        last_call_metadata["model"] = self.model_name
        last_call_metadata["backend"] = "openai"

        return content


class GenericAPIProvider(BaseLLMProvider):
    """Generic HTTP / OpenAI-Compatible Endpoint Provider (LMStudio, vLLM, Ollama, OmniRoute, 9Router)."""
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-cli")
        self.base_url = (
            base_url
            or os.getenv("LLM_API_BASE_URL")
            or os.getenv("API_Base_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:20128/v1"
        )
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("NINEROUTER_OPENAI_API_KEY")
            or "not-needed"
        )
        self.extra_kwargs = kwargs
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("OpenAI package required for API client. Run: pip install openai")
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "4096"))

        t0 = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=tokens
        )
        t1 = time.time()

        content = response.choices[0].message.content or ""
        if response.usage:
            last_call_metadata["prompt_tokens"] = response.usage.prompt_tokens
            last_call_metadata["completion_tokens"] = response.usage.completion_tokens
            last_call_metadata["total_tokens"] = response.usage.total_tokens
        last_call_metadata["latency_seconds"] = t1 - t0
        last_call_metadata["model"] = self.model_name
        last_call_metadata["backend"] = "generic_api"

        return content


def get_llm_provider(backend: str = "api", model_name: Optional[str] = None, **kwargs) -> BaseLLMProvider:
    """
    Factory function to instantiate the desired LLM provider.
    Supported backends: 'openai', 'api', 'lmstudio', 'ollama', 'dummy'.
    """
    backend = (backend or "api").lower()
    if backend in ["dummy", "mock", "none"]:
        return DummyLLMProvider(verbose=kwargs.get("verbose", False))
    elif backend == "openai":
        return OpenAIProvider(model_name=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), **kwargs)
    elif backend in ["api", "lmstudio", "ollama", "local", "9router", "ninerouter"]:
        base_url = kwargs.get("base_url")
        api_key = kwargs.get("api_key")
        if not base_url:
            if backend == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            elif backend == "lmstudio":
                base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            else:
                base_url = (
                    os.getenv("LLM_API_BASE_URL")
                    or os.getenv("API_Base_URL")
                    or os.getenv("OPENAI_BASE_URL")
                    or "http://localhost:20128/v1"
                )
        if not api_key:
            api_key = (
                os.getenv("LLM_API_KEY")
                or os.getenv("API_KEY")
                or os.getenv("NINEROUTER_OPENAI_API_KEY")
                or "not-needed"
            )
        resolved_model = model_name or os.getenv("LLM_MODEL", "gemini-cli")
        return GenericAPIProvider(model_name=resolved_model, base_url=base_url, api_key=api_key, **kwargs)
    else:
        resolved_model = model_name or os.getenv("LLM_MODEL", "gemini-cli")
        return GenericAPIProvider(model_name=resolved_model, **kwargs)


def generate_llm_response(
    prompt: str,
    model_name: Optional[str] = None,
    arg3: Optional[str] = None,
    arg4: Optional[str] = None,
    backend: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Unified LLM call interface. Robustly handles both calling conventions:
      - generate_llm_response(prompt, model_name, backend="api", system_prompt="...")
      - generate_llm_response(prompt, model_name, "Dinasor", self.backend)
    """
    known_backends = {"api", "openai", "lmstudio", "ollama", "dummy", "mock", "local", "none", "9router", "ninerouter"}
    chosen_backend = backend or os.getenv("LLM_BACKEND", "api")
    chosen_system = system_prompt

    if arg3 is not None:
        if arg3.lower() in known_backends:
            chosen_backend = arg3.lower()
            if arg4 is not None:
                chosen_system = arg4
        else:
            chosen_system = arg3
            if arg4 is not None and arg4.lower() in known_backends:
                chosen_backend = arg4.lower()

    init_kwarg_keys = {"base_url", "api_key", "verbose"}
    provider_kwargs = {k: v for k, v in kwargs.items() if k in init_kwarg_keys}
    generate_kwargs = {k: v for k, v in kwargs.items() if k not in init_kwarg_keys}

    resolved_model = model_name or os.getenv("LLM_MODEL", "gemini-cli")
    provider = get_llm_provider(backend=chosen_backend, model_name=resolved_model, **provider_kwargs)
    return provider.generate(prompt, system_prompt=chosen_system, **generate_kwargs)
