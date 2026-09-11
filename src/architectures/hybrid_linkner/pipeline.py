import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any, Union
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

import sys


# Ensure root directory is on sys.path for package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from src.architectures.hybrid_linkner.spanner_model import SpanNERModel
except ImportError:
    from .spanner_model import SpanNERModel
from src.architectures.hybrid_linkner.prompt_templates import (
    build_lsf_linkner_prompt,
    build_lsf_linkner_batch_prompt,
    LSF_CATEGORY_DEFINITIONS,
    get_category_definitions
)
from src.common.label_mapping import (
    OLD_TO_NEW_LABELS,
    NEW_TO_OLD_LABELS,
    to_new_label,
    to_old_label,
    map_label,
    canonicalize_label,
    map_prediction_item,
    map_predictions,
    ALL_KNOWN_LABELS
)
from src.common.config import LinkNERRunConfig, DEFAULT_CONFIG
from src.architectures.gating_nn.prompt_templates import (
    build_single_call_prompt,
    parse_single_call_llm_response,
    format_inlined_abstract,
    format_entities_evidence_v1,
    format_entities_evidence_v2
)
from src.architectures.gating_nn.span_features import run_nms


# Novelty / Anomaly scoring via LOFNoveltyDetector (hierarchical Qwen 8B -> BGE-M3 -> Local RoBERTa default)
try:
    from src.architectures.hybrid_linkner.novelty_detector import (
        LOFNoveltyDetector,
        build_lof_novelty_detector,
        LocalTransformersEmbedder,
        SentenceTransformerEmbedder,
        FallbackEmbedder,
        get_default_embedder,
    )
    NOVELTY_AVAILABLE = True
except Exception as _novelty_import_err:
    NOVELTY_AVAILABLE = False
    _novelty_import_err = None

# Load environment variables from .env (override=True ensures dynamic hot-reloading)
load_dotenv(override=True)


# =====================================================================
# 1. Pluggable LLM Providers (OpenAI, Custom Endpoint, Fallback)
# =====================================================================

class BaseLLMProvider:
    """Base interface for LLMs in LinkNER."""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class DummyLLMProvider(BaseLLMProvider):
    """Fallback provider when API key is missing or for offline testing."""
    def __init__(self, verbose=True):
        self.verbose = verbose

    def generate(self, prompt: str) -> str:
        if self.verbose:
            print("\n------------------- LSF LLM Prompt (Mock Mode) -------------------")
            print(prompt)
            print("------------------------------------------------------------------")
            
        if "physical environments" in prompt.lower():
            return "Environmental_exposures"
        elif "pa) is a health behavior" in prompt.lower():
            return "None of the above"
        elif "vitamins e and c" in prompt.lower() or "antioxidant" in prompt.lower():
            return "Nutrition"
        else:
            return "None of the above"

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI API Provider (GPT-4o, Qwen, OmniRoute, custom endpoints)."""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, timeout: float = 30.0, temperature: float = None, max_tokens: int = None):
        import openai
        api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("API_Base_URL") or os.getenv("OPENAI_BASE_URL")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("Invalid OpenAI API key. Please update your .env file with a valid API_KEY.")
            
        self.timeout = timeout
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        self.model_name = model_name or os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NAME", "deepseek-web/deepseek-chat")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.0")) if temperature is None else temperature
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "256")) if max_tokens is None else max_tokens
        self.fallback = DummyLLMProvider(verbose=False)

    def generate(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        temp = self.temperature if temperature is None else temperature
        max_t = self.max_tokens if max_tokens is None else max_tokens
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise biomedical NER entity classifier."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temp,
                "timeout": self.timeout
            }
            if max_t is not None:
                kwargs["max_tokens"] = max_t
            top_p_env = os.getenv("LLM_TOP_P")
            if top_p_env:
                kwargs["top_p"] = float(top_p_env)
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                # If reasoning model hit token limit or returned content in reasoning
                reasoning = getattr(choice.message, "reasoning", "") or ""
                content = reasoning.strip() if reasoning else ""
            return content.strip()
        except Exception as e:
            print(f"⚠️ [LLM Provider Error] Provider {self.model_name} @ {self.client.base_url} failed ({type(e).__name__}: {e}). Falling back to Mock LLM.")
            return self.fallback.generate(prompt)

class OpenRouterLLMProvider(BaseLLMProvider):
    """OpenRouter API Provider (supports model routing & provider order preferences)."""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, providers: list = None, timeout: float = 30.0, temperature: float = None, max_tokens: int = None):
        import openai
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash-0731")
        if providers:
            self.providers = providers
        else:
            prov_str = os.getenv("OPENROUTER_PROVIDERS", "DeepInfra,Decart")
            self.providers = [p.strip() for p in prov_str.split(",") if p.strip()] if prov_str else []

        self.timeout = timeout
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.0")) if temperature is None else temperature
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048")) if max_tokens is None else max_tokens
        self.fallback = DummyLLMProvider(verbose=False)

    def generate(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        temp = self.temperature if temperature is None else temperature
        max_t = self.max_tokens if max_tokens is None else max_tokens
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise biomedical NER entity classifier."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temp,
                "timeout": self.timeout
            }
            if max_t is not None:
                kwargs["max_tokens"] = max_t
            if self.providers:
                kwargs["extra_body"] = {
                    "provider": {
                        "order": self.providers
                    }
                }
            top_p_env = os.getenv("LLM_TOP_P")
            if top_p_env:
                kwargs["top_p"] = float(top_p_env)
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                # Handle reasoning models where output might be in reasoning field when truncated
                reasoning = getattr(choice.message, "reasoning", "") or ""
                content = reasoning.strip() if reasoning else ""
            return content.strip()
        except Exception as e:
            print(f"⚠️ [LLM Provider Error] OpenRouter Provider {self.model_name} @ {self.client.base_url} failed ({type(e).__name__}: {e}). Falling back to Mock LLM.")
            return self.fallback.generate(prompt)

class OllamaLLMProvider(BaseLLMProvider):
    """Ollama API Provider (supports OpenAI compatibility and native /api/chat fallback)."""
    def __init__(self, api_key: str = "ollama", base_url: str = None, model_name: str = None, timeout: float = 60.0):
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL_NAME", "qwen3.6-35b")
        self.timeout = timeout
        self.fallback = DummyLLMProvider(verbose=False)

    def generate(self, prompt: str) -> str:
        import json
        import urllib.request
        # Method 1: OpenAI API compatibility layer
        try:
            import openai
            v1_url = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
            client = openai.OpenAI(api_key=self.api_key, base_url=v1_url, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise biomedical NER entity classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                timeout=self.timeout
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e1:
            # Method 2: Direct native Ollama HTTP API (/api/chat) fallback
            try:
                native_url = self.base_url.replace("/v1", "").rstrip("/") + "/api/chat"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a precise biomedical NER entity classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                req = urllib.request.Request(
                    native_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        return content
            except Exception as e2:
                print(f"⚠️ [Ollama LLM Provider Error] OpenAI call failed ({e1}) and Native /api/chat call failed ({e2}).")
            
            return self.fallback.generate(prompt)

class CustomAPILLMProvider(BaseLLMProvider):
    """Custom HTTP Endpoint Provider (vLLM, Ollama, LM Studio, etc.)."""
    def __init__(self, api_url: str = None, api_key: str = None, model_name: str = None):
        import requests
        self.api_url = api_url or os.getenv("CUSTOM_LLM_API_URL")
        self.api_key = api_key or os.getenv("CUSTOM_LLM_API_KEY") or os.getenv("API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "qwen3.7-plus")

    def generate(self, prompt: str) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        res = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

def get_default_llm_provider():
    """Automatically picks the best LLM provider based on .env config (with active endpoint health checks)."""
    import urllib.request
    load_dotenv(override=True)

    forced_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_key = os.getenv("OLLAMA_API_KEY", "ollama")
    ollama_model = os.getenv("OLLAMA_MODEL_NAME", "qwen3.6-35b")

    unsloth_key = os.getenv("UNSLOTH_STUDIO_API_KEY")
    unsloth_base = os.getenv("UNSLOTH_STUDIO_BASE_URL", "http://127.0.0.1:8888/v1")
    unsloth_model = os.getenv("UNSLOTH_MODEL_NAME", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF")

    nv_key = os.getenv("NVIDIA_API_KEY")
    nv_base = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    api_key = os.getenv("NINEROUTER_OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_Base_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "http://localhost:20128/v1"
    model_name = os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NAME") or "deepseek/deepseek-v4-flash-0731"
    custom_url = os.getenv("CUSTOM_LLM_API_URL")

    # -------------------------------------------------------------
    # 1. Explicit / Forced Provider Selection (from LLM_PROVIDER env)
    # -------------------------------------------------------------
    if forced_provider in ("9router", "ninerouter", "omniroute", "gateway", "local_gateway"):
        g_key = os.getenv("NINEROUTER_OPENAI_API_KEY") or api_key or "sk-local-gateway"
        g_base = base_url or "http://localhost:20128/v1"
        g_model = os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NAME") or "gemini-cli"
        print(f"==========================================================================")
        print(f"  🚀 LLM Provider: 9Router / OmniRoute Local Gateway (Forced via LLM_PROVIDER={forced_provider})")
        print(f"  📍 Endpoint: {g_base} | Model: {g_model}")
        print(f"==========================================================================")
        return OpenAILLMProvider(api_key=g_key, base_url=g_base, model_name=g_model)

    if forced_provider == "openrouter":
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        openrouter_model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NAME", "deepseek/deepseek-v4-flash-0731")
        prov_str = os.getenv("OPENROUTER_PROVIDERS", "DeepInfra,Decart")
        providers = [p.strip() for p in prov_str.split(",") if p.strip()] if prov_str else []
        print(f"==========================================================================")
        print(f"  🌐 LLM Provider: OpenRouter Cloud API (Forced via LLM_PROVIDER=openrouter)")
        print(f"  📍 Endpoint: {openrouter_base} | Model: {openrouter_model}")
        print(f"  🔀 Providers Preference Order: {providers}")
        print(f"==========================================================================")
        return OpenRouterLLMProvider(api_key=openrouter_key, base_url=openrouter_base, model_name=openrouter_model, providers=providers)

    if forced_provider == "ollama":
        print(f"==========================================================================")
        print(f"  🦙 LLM Provider: Local Ollama (Forced via LLM_PROVIDER=ollama)")
        print(f"  📍 Endpoint: {ollama_base} | Model: {ollama_model}")
        print(f"==========================================================================")
        return OllamaLLMProvider(api_key=ollama_key, base_url=ollama_base, model_name=ollama_model)

    if forced_provider == "unsloth":
        print(f"==========================================================================")
        print(f"  🟢 LLM Provider: Unsloth UI Backend (Forced via LLM_PROVIDER=unsloth)")
        print(f"  📍 Endpoint: {unsloth_base} | Model: {unsloth_model}")
        print(f"==========================================================================")
        return OpenAILLMProvider(api_key=unsloth_key or "sk-unsloth", base_url=unsloth_base, model_name=unsloth_model)

    if forced_provider == "nvidia":
        print(f"==========================================================================")
        print(f"  ☁️  LLM Provider: NVIDIA Cloud API (Forced via LLM_PROVIDER=nvidia)")
        print(f"  📍 Endpoint: {nv_base} | Model: {model_name}")
        print(f"==========================================================================")
        return OpenAILLMProvider(api_key=nv_key, base_url=nv_base, model_name=model_name)

    if forced_provider in ("dummy", "mock", "offline"):
        print("[Info] Running LinkNER with Mock LLM Provider (LLM_PROVIDER=dummy).")
        return DummyLLMProvider()

    # -------------------------------------------------------------
    # 2. Auto-Discovery & Health-Checked Fallback Cascade
    # -------------------------------------------------------------
    # Priority 1: 9Router / OmniRoute Local Gateway (http://localhost:20128)
    if "20128" in str(base_url) or "localhost:20128" in str(base_url):
        try:
            native_check = base_url.rstrip("/") + "/models"
            req = urllib.request.Request(native_check, headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"==========================================================================")
                    print(f"  🚀 LLM Provider: 9Router / OmniRoute Local Gateway (Status: ONLINE)")
                    print(f"  📍 Endpoint: {base_url} | Model: {model_name}")
                    print(f"==========================================================================")
                    return OpenAILLMProvider(api_key=api_key, base_url=base_url, model_name=model_name)
        except Exception:
            pass

    # Priority 2: OpenRouter Cloud API
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key and openrouter_key.startswith("sk-or-"):
        openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        openrouter_model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash-0731")
        prov_str = os.getenv("OPENROUTER_PROVIDERS", "DeepInfra,Decart")
        providers = [p.strip() for p in prov_str.split(",") if p.strip()] if prov_str else []
        print(f"==========================================================================")
        print(f"  🌐 LLM Provider: OpenRouter Cloud API (Status: ONLINE)")
        print(f"  📍 Endpoint: {openrouter_base} | Model: {openrouter_model}")
        print(f"  🔀 Providers Preference Order: {providers}")
        print(f"==========================================================================")
        return OpenRouterLLMProvider(api_key=openrouter_key, base_url=openrouter_base, model_name=openrouter_model, providers=providers)

    # Priority 3: Local Ollama Instance (http://localhost:11434)
    if ollama_base:
        try:
            native_check = ollama_base.replace("/v1", "").rstrip("/") + "/api/tags"
            req = urllib.request.Request(native_check)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"==========================================================================")
                    print(f"  🦙 LLM Provider: Local Ollama Instance (Status: ONLINE)")
                    print(f"  📍 Endpoint: {ollama_base} | Model: {ollama_model}")
                    print(f"==========================================================================")
                    return OllamaLLMProvider(api_key=ollama_key, base_url=ollama_base, model_name=ollama_model)
        except Exception:
            pass

    # Priority 4: OpenAI / Standard OpenAI-Compatible API
    if api_key and api_key != "your_openai_api_key_here":
        try:
            print(f"==========================================================================")
            print(f"  🌐 LLM Provider: OpenAI-Compatible API (Status: ONLINE)")
            print(f"  📍 Endpoint: {base_url} | Model: {model_name}")
            print(f"==========================================================================")
            return OpenAILLMProvider(api_key=api_key, base_url=base_url, model_name=model_name)
        except Exception as e:
            print(f"[Warning] Failed to initialize OpenAI provider: {e}. Falling back to Mock LLM.")
            return DummyLLMProvider()

    elif custom_url:
        return CustomAPILLMProvider(api_url=custom_url)
    else:
        print("[Info] No active API provider found in .env. Running LinkNER with Mock LLM Provider.")
        return DummyLLMProvider()

# Backward compatibility alias
get_llm_provider = get_default_llm_provider

# =====================================================================
# 2. Integrated LinkNER RDC Pipeline
# =====================================================================

class LinkNERPipeline:
    """
    Unified Hybrid LinkNER Pipeline.
    Integrates SpanNER neural recognition, LOF novelty estimation, multi-operator gating,
    and single-pass LLM arbitration with neuro-symbolic evidence prompts (v1 with label, v2 blind, or legacy).
    """

    def __init__(
        self,
        spanner_model_path: str = None,
        spanner_checkpoint_path: str = None,
        dataset_path: str = None,
        uncertainty_method: str = None,
        uncertainty_threshold: float = None,
        category_thresholds: dict = None,
        llm_provider: BaseLLMProvider = None,
        novelty_scorer=None,
        w_novelty: float = None,
        w_uncertainty: float = None,
        c1: float = None,
        c2: float = None,
        novelty_threshold: float = None,
        margin_threshold: float = None,
        use_margin_safeguard: bool = None,
        gating_mode: str = None,
        prompt_version: str = None,
        use_nms: bool = None,
        mcd_passes: int = None,
        arbitrate_by_uncertainty: bool = True,
        use_new_labels: bool = None,
        config: Optional[LinkNERRunConfig] = None,
    ):
        self.config = config
        spanner_model_path = spanner_model_path or os.path.join(PROJECT_ROOT, "models/NER_Model/trained_NER_model")
        spanner_checkpoint_path = spanner_checkpoint_path or os.path.join(PROJECT_ROOT, "models/SpanNER_LSF/best_spanner_160train.pt")
        
        # Dataset path resolution with fallback candidates
        candidate_paths = [
            dataset_path,
            os.path.join(PROJECT_ROOT, "data/processed/spanner_dataset.json"),
            os.path.join(PROJECT_ROOT, "data/Extra/processed/spanner_dataset.json"),
            os.path.join(PROJECT_ROOT, "data/Extra/spanner_dataset.json")
        ]
        resolved_dataset_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_dataset_path = p
                break

        # Set target ontology schema (use_new_labels=True for relabeled test set)
        env_new_labels = os.getenv("USE_NEW_LABELS", "false").lower() in ("true", "1", "yes")
        if use_new_labels is not None:
            self.use_new_labels = use_new_labels
        elif config is not None:
            self.use_new_labels = config.use_new_labels
        else:
            self.use_new_labels = env_new_labels

        # Resolve device (defaults to CPU on macOS to support Monte Carlo Dropout without MPS limitation)
        env_device = os.getenv("DEVICE", "").lower()
        if env_device:
            self.device = torch.device(env_device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Pre-inspect checkpoint to guarantee exact classifier weight alignment (e.g. 10 vs 11 classes)
        self.checkpoint_state = None
        ckpt_num_classes = None
        if os.path.exists(spanner_checkpoint_path):
            self.checkpoint_state = torch.load(spanner_checkpoint_path, map_location=self.device)
            if "classifier.3.weight" in self.checkpoint_state:
                ckpt_num_classes = self.checkpoint_state["classifier.3.weight"].shape[0]

        # Load dataset label mapping matching checkpoint shape
        if ckpt_num_classes == 10:
            if self.use_new_labels:
                self.id2label = {
                    0: 'O',
                    1: 'Personal_care_products_and_cosmetic_procedures',
                    2: 'Substance_use',
                    3: 'Environmental_exposures',
                    4: 'Mental_health_practices',
                    5: 'Non_physical_leisure_time_activities',
                    6: 'Nutrition',
                    7: 'Physical_activities',
                    8: 'Sleep',
                    9: 'Socioeconomic_factors'
                }
            else:
                self.id2label = {
                    0: 'O',
                    1: 'Beauty_and_Cleaning',
                    2: 'Drugs',
                    3: 'Environmental_exposures',
                    4: 'Mental_health_practices',
                    5: 'Non_physical_leisure_time_activities',
                    6: 'Nutrition',
                    7: 'Physical_activity',
                    8: 'Sleep',
                    9: 'Socioeconomic_factors'
                }
            self.num_classes = 10
        elif resolved_dataset_path and os.path.exists(resolved_dataset_path):
            with open(resolved_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.id2label = {int(k): v for k, v in data["id2label"].items()}
            self.num_classes = len(self.id2label)
        else:
            # Standard SpanNER LSF 11-class mapping fallback
            self.id2label = {
                0: 'O',
                1: 'Beauty_and_Cleaning',
                2: 'Drugs',
                3: 'Environmental_exposures',
                4: 'Lifestyle_factor',
                5: 'Mental_health_practices',
                6: 'Non_physical_leisure_time_activities',
                7: 'Nutrition',
                8: 'Physical_activity',
                9: 'Sleep',
                10: 'Socioeconomic_factors'
            }
            self.num_classes = ckpt_num_classes or len(self.id2label)


        # Unreliability Scoring Configuration:
        # Unreliability = w_novelty * novelty + w_uncertainty * uncertainty (or PROB_OR, COPULA)
        self.novelty_scorer = novelty_scorer
        if self.novelty_scorer is not None:
            if w_novelty is not None:
                self.w_novelty = w_novelty
            elif config is not None:
                self.w_novelty = config.w_novelty
            elif c1 is not None:
                self.w_novelty = c1
            else:
                self.w_novelty = float(os.getenv("WEIGHT_NOVELTY", os.getenv("NOVELTY_C1", "0.30")))

            if w_uncertainty is not None:
                self.w_uncertainty = w_uncertainty
            elif config is not None:
                self.w_uncertainty = config.w_uncertainty
            elif c2 is not None:
                self.w_uncertainty = c2
            else:
                self.w_uncertainty = float(os.getenv("WEIGHT_UNCERTAINTY", os.getenv("NOVELTY_C2", "0.70")))
        else:
            self.w_novelty = 0.0
            self.w_uncertainty = 1.0

        # Backward compatibility aliases
        self.c1 = self.w_novelty
        self.c2 = self.w_uncertainty
        self.novelty_threshold = novelty_threshold if novelty_threshold is not None else float(os.getenv("NOVELTY_THRESHOLD", "0.60"))
        
        # Gating Mode Operator
        if gating_mode is not None:
            self.gating_mode = gating_mode.lower()
        elif config is not None:
            self.gating_mode = config.gating_operator.lower()
        else:
            self.gating_mode = (os.getenv("GATING_MODE", "prob_or")).lower()

        # Prompt template version ('v1_with_label', 'v2_no_label', 'legacy')
        if prompt_version is not None:
            self.prompt_version = prompt_version.lower()
        elif config is not None:
            self.prompt_version = config.prompt_template_version.lower()
        else:
            self.prompt_version = (os.getenv("PROMPT_VERSION", "v1_with_label")).lower()

        # NMS policy (True = apply Longest-Span NMS, False = drop NMS)
        if use_nms is not None:
            self.use_nms = use_nms
        elif config is not None:
            self.use_nms = config.use_nms
        else:
            self.use_nms = os.getenv("USE_NMS", "false").lower() in ("true", "1", "yes")

        # Margin Safeguard Veto Threshold
        if margin_threshold is not None:
            self.margin_threshold = margin_threshold
        elif config is not None:
            self.margin_threshold = config.margin_veto_threshold
        else:
            self.margin_threshold = float(os.getenv("ARBITRATION_MARGIN_THRESHOLD", "0.50"))

        if use_margin_safeguard is not None:
            self.use_margin_safeguard = use_margin_safeguard
        elif config is not None:
            self.use_margin_safeguard = config.use_margin_safeguard
        else:
            self.use_margin_safeguard = True

        self.mcd_passes = mcd_passes if mcd_passes is not None else (config.mcd_passes if config else 5)
        self.arbitrate_by_uncertainty = arbitrate_by_uncertainty

        if uncertainty_method is not None:
            self.uncertainty_method = uncertainty_method.lower()
        elif config is not None:
            self.uncertainty_method = config.uncertainty_metric.lower()
        else:
            self.uncertainty_method = (os.getenv("UNCERTAINTY_METHOD", "margin")).lower()
        
        env_threshold = os.getenv("UNRELIABILITY_THRESHOLD") or os.getenv("UNCERTAINTY_THRESHOLD")
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold
        elif config is not None:
            self.uncertainty_threshold = config.threshold
        else:
            self.uncertainty_threshold = float(env_threshold or 0.35)
        self.unreliability_threshold = self.uncertainty_threshold
        
        # Load Category-Specific Thresholds mapping if available
        cat_thresh_path = os.path.join(PROJECT_ROOT, "reports/category_thresholds.json")
        if category_thresholds is not None:
            self.category_thresholds = category_thresholds
        elif os.path.exists(cat_thresh_path):
            with open(cat_thresh_path, "r", encoding="utf-8") as f:
                self.category_thresholds = json.load(f)
        else:
            self.category_thresholds = {}
        
        self.llm_provider = llm_provider or get_default_llm_provider()
        
        # Load SpanNER Model
        self.spanner_model = SpanNERModel(encoder_path=spanner_model_path, num_classes=self.num_classes, max_span_width=6)
        state_dict = self.checkpoint_state if self.checkpoint_state is not None else torch.load(spanner_checkpoint_path, map_location=self.device)
        self.spanner_model.load_state_dict(state_dict)
        self.spanner_model.to(self.device)
        self.spanner_model.eval()
        self.checkpoint_state = None

        self.tokenizer = AutoTokenizer.from_pretrained(spanner_model_path)


    def _compute_all_uncertainties(self, logits, input_ids=None, attention_mask=None, mcd_passes=5, temperature=1.8, active_method="pe", compute_all=False):
        """
        Computes uncertainty scores across batch [B, Num_Spans] or single [Num_Spans].
        When compute_all=False (default), lazily computes only the requested active_method for maximum speed:
        1. Least Confidence (LC): 1 - max(p)
        2. Prediction Entropy (PE): sum(-p * log(p)) / log(num_classes) [Normalized to [0,1]]
        3. Monte Carlo Dropout (MCD): avg entropy across M stochastic forward passes
        4. Evidential Neural Network (ENN): u = C / S, where e_c = Softplus(logit_c), S = sum(e_c + 1)
        5. Margin Uncertainty (Margin): 1 - (top1_prob - top2_prob) (computed with MCD passes if input_ids given)
        """
        import math
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)  # Shape: [B, Num_Spans, Num_Classes]

        probs = F.softmax(logits, dim=-1)  # [B, Num_Spans, Num_Classes]
        max_probs, pred_label_ids = torch.max(probs, dim=-1)  # [B, Num_Spans]
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, :, 0]
        top2_probs = sorted_probs[:, :, 1]
        top1_label_ids = sorted_indices[:, :, 0]
        top2_label_ids = sorted_indices[:, :, 1]
        margins = top1_probs - top2_probs
        p_o = probs[:, :, 0]

        res = {}
        act = (active_method or "pe").lower()

        # Method 1: Least Confidence (LC)
        if compute_all or act == "lc":
            u_lc = 1.0 - max_probs  # [B, Num_Spans]
            res["lc"] = {
                "uncertainty": u_lc,
                "max_probs": max_probs,
                "pred_labels": pred_label_ids,
                "margins": margins,
                "top1_probs": top1_probs,
                "top2_probs": top2_probs,
                "top1_labels": top1_label_ids,
                "top2_labels": top2_label_ids,
                "p_o": p_o
            }

        # Method 2: Prediction Entropy (PE)
        if compute_all or act == "pe":
            raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # [B, Num_Spans]
            max_entropy = math.log(self.num_classes)
            u_pe_norm = raw_entropy / max_entropy  # [B, Num_Spans]
            res["pe"] = {
                "uncertainty": u_pe_norm,
                "raw_entropy": raw_entropy,
                "max_probs": max_probs,
                "pred_labels": pred_label_ids,
                "margins": margins,
                "top1_probs": top1_probs,
                "top2_probs": top2_probs,
                "top1_labels": top1_label_ids,
                "top2_labels": top2_label_ids,
                "p_o": p_o
            }

        # Method 3: Monte Carlo Dropout (MCD)
        if compute_all or act == "mcd":
            if input_ids is not None:
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask is not None and attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)

                self.spanner_model.train()  # Activate dropout layers
                mcd_probs_list = []
                with torch.no_grad():
                    for _ in range(mcd_passes):
                        outputs_m = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                        p_m = F.softmax(outputs_m["logits"], dim=-1)
                        mcd_probs_list.append(p_m)
                self.spanner_model.eval()

                mcd_stack = torch.stack(mcd_probs_list, dim=0)  # [M, B, Num_Spans, Num_Classes]
                mcd_entropies = -torch.sum(mcd_stack * torch.log(mcd_stack + 1e-9), dim=-1)  # [M, B, Num_Spans]
                avg_mcd_entropy = torch.mean(mcd_entropies, dim=0)  # [B, Num_Spans]
                max_entropy = math.log(self.num_classes)
                u_mcd_norm = avg_mcd_entropy / max_entropy

                # MCD expected probabilities across passes
                mean_mcd_probs = torch.mean(mcd_stack, dim=0)  # [B, Num_Spans, Num_Classes]
                m_sorted_probs, m_sorted_indices = torch.sort(mean_mcd_probs, dim=-1, descending=True)
                mcd_top1 = m_sorted_probs[:, :, 0]
                mcd_top2 = m_sorted_probs[:, :, 1]
                mcd_top1_lbl = m_sorted_indices[:, :, 0]
                mcd_top2_lbl = m_sorted_indices[:, :, 1]
                mcd_margins = mcd_top1 - mcd_top2
                mcd_max_probs, mcd_pred_labels = torch.max(mean_mcd_probs, dim=-1)
                mcd_po = mean_mcd_probs[:, :, 0]
            else:
                raw_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                max_entropy = math.log(self.num_classes)
                u_mcd_norm = raw_entropy / max_entropy
                avg_mcd_entropy = raw_entropy
                mcd_max_probs, mcd_pred_labels = max_probs, pred_label_ids
                mcd_margins = margins
                mcd_top1, mcd_top2 = top1_probs, top2_probs
                mcd_top1_lbl, mcd_top2_lbl = top1_label_ids, top2_label_ids
                mcd_po = p_o

            res["mcd"] = {
                "uncertainty": u_mcd_norm,
                "raw_entropy": avg_mcd_entropy,
                "max_probs": mcd_max_probs,
                "pred_labels": mcd_pred_labels,
                "margins": mcd_margins,
                "top1_probs": mcd_top1,
                "top2_probs": mcd_top2,
                "top1_labels": mcd_top1_lbl,
                "top2_labels": mcd_top2_lbl,
                "p_o": mcd_po
            }

        # Method 4: Evidential Neural Network (ENN)
        if compute_all or act == "enn":
            evidence = F.softplus(logits)  # [B, Num_Spans, Num_Classes], e_c >= 0
            alpha = evidence + 1.0  # Dirichlet parameters (uniform prior = 1.0)
            S = torch.sum(alpha, dim=-1)  # Total Dirichlet strength [B, Num_Spans]
            enn_expected_probs = alpha / S.unsqueeze(-1)
            enn_max_probs, enn_pred_labels = torch.max(enn_expected_probs, dim=-1)
            u_enn = self.num_classes / S  # u = C / S in [0, 1]
            res["enn"] = {
                "uncertainty": u_enn,
                "max_probs": enn_max_probs,
                "pred_labels": enn_pred_labels,
                "dirichlet_strength": S,
                "margins": margins,
                "top1_probs": top1_probs,
                "top2_probs": top2_probs,
                "top1_labels": top1_label_ids,
                "top2_labels": top2_label_ids,
                "p_o": p_o
            }

        # Method 5: Margin Uncertainty (Margin / Margin-MCD)
        if compute_all or act == "margin":
            if input_ids is not None and mcd_passes > 1:
                self.spanner_model.train()
                mcd_probs_list = []
                with torch.no_grad():
                    for _ in range(mcd_passes):
                        outputs_m = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                        p_m = F.softmax(outputs_m["logits"], dim=-1)
                        mcd_probs_list.append(p_m)
                self.spanner_model.eval()

                mcd_stack = torch.stack(mcd_probs_list, dim=0)  # [M, B, Num_Spans, Classes]
                mean_mcd_probs = torch.mean(mcd_stack, dim=0)   # [B, Num_Spans, Classes]
                m_sorted_probs, m_sorted_indices = torch.sort(mean_mcd_probs, dim=-1, descending=True)
                mcd_top1 = m_sorted_probs[:, :, 0]
                mcd_top2 = m_sorted_probs[:, :, 1]
                mcd_top1_lbl = m_sorted_indices[:, :, 0]
                mcd_top2_lbl = m_sorted_indices[:, :, 1]
                mcd_margins = mcd_top1 - mcd_top2
                u_margin = 1.0 - mcd_margins
                mcd_max_probs, mcd_pred_labels = torch.max(mean_mcd_probs, dim=-1)
                mcd_po = mean_mcd_probs[:, :, 0]
                res["margin"] = {
                    "uncertainty": u_margin,
                    "max_probs": mcd_max_probs,
                    "pred_labels": mcd_pred_labels,
                    "margins": mcd_margins,
                    "top1_probs": mcd_top1,
                    "top2_probs": mcd_top2,
                    "top1_labels": mcd_top1_lbl,
                    "top2_labels": mcd_top2_lbl,
                    "p_o": mcd_po
                }
            else:
                scaled_logits = logits / float(temperature)
                scaled_probs = F.softmax(scaled_logits, dim=-1)
                s_probs, s_indices = torch.sort(scaled_probs, dim=-1, descending=True)
                t1_prob = s_probs[:, :, 0]
                t2_prob = s_probs[:, :, 1]
                u_margin = 1.0 - (t1_prob - t2_prob)  # [B, Num_Spans]
                res["margin"] = {
                    "uncertainty": u_margin,
                    "max_probs": max_probs,
                    "pred_labels": pred_label_ids,
                    "margins": margins,
                    "top1_probs": top1_probs,
                    "top2_probs": top2_probs,
                    "top1_labels": top1_label_ids,
                    "top2_labels": top2_label_ids,
                    "p_o": p_o
                }

        return res

    def _novelty_batch(self, texts):
        """Embed candidate span texts and return calibrated novelty scores in [0, 1]."""
        if not texts or self.novelty_scorer is None:
            return np.array([], dtype=float)
        vecs = self.novelty_scorer.embed(texts)
        return self.novelty_scorer.novelty(vecs)

    def _escalate_and_arbitrate_abstract(
        self,
        text: str,
        uncertain_candidates: list,
        method: str,
        enable_few_shot: bool = True,
        arbitrate_by_uncertainty: bool = None,
        use_new_labels: bool = None,
        prompt_version: str = None,
        margin_threshold: float = None,
        use_margin_safeguard: bool = None,
    ) -> list:
        """
        Executes consolidated single-pass LLM prompt for all uncertain/novel spans in an abstract,
        then arbitrates the response with margin safeguard against false deletions.
        """
        if not uncertain_candidates:
            return []

        active_use_new_labels = self.use_new_labels if use_new_labels is None else use_new_labels
        active_prompt_ver = (prompt_version or self.prompt_version or "v1_with_label").lower()
        active_margin_thresh = self.margin_threshold if margin_threshold is None else margin_threshold
        active_use_veto = self.use_margin_safeguard if use_margin_safeguard is None else use_margin_safeguard
        category_defs = get_category_definitions(use_new_labels=active_use_new_labels)

        # Branch 1: Single Consolidated Prompt (Prompt v1 with Label Evidence, Prompt v2 without Label, or Prompt v1 with Allowed New Spans)
        if active_prompt_ver in ("v1", "v1_with_label", "v2", "v2_no_label", "v1_allowed_new_span", "v1_new_spans", "v1_allowed_new_spans"):
            if active_prompt_ver in ("v1_allowed_new_span", "v1_new_spans", "v1_allowed_new_spans"):
                ver_key = "v1_allowed_new_span"
            elif active_prompt_ver in ("v1", "v1_with_label"):
                ver_key = "v1"
            else:
                ver_key = "v2"

            prompt = build_single_call_prompt(text, uncertain_candidates, version=ver_key)
            try:
                raw_response = self.llm_provider.generate(prompt)
            except Exception as e:
                print(f"⚠️ [Warning] Generative LLM API call error: {e}. Falling back to Primary SpanNER labels.")
                raw_response = "{}"

            allow_new = ver_key == "v1_allowed_new_span"
            parsed_res = parse_single_call_llm_response(raw_response, uncertain_candidates, return_new_spans=allow_new)
            if allow_new:
                parsed_list, new_discovered = parsed_res
            else:
                parsed_list, new_discovered = parsed_res, []

            arbitrated_entities = []

            for cand, parsed in zip(uncertain_candidates, parsed_list):
                raw_llm_label = parsed.get("llm_label", "O")
                llm_label = canonicalize_label(raw_llm_label, use_new_labels=active_use_new_labels)
                cand_spanner_label = canonicalize_label(cand["spanner_label"], use_new_labels=active_use_new_labels)
                cand_margin = float(cand.get("margin", cand.get("prob", 0.0)))
                u_local = cand.get("unreliability", cand.get("u_score", 0.0))
                u_spanner = cand.get("u_score", 0.0)
                novelty = cand.get("novelty")
                rationale = parsed.get("llm_rationale", "")

                # Margin-Based Arbitration Safeguard Veto
                if active_use_veto and llm_label == "O" and cand_margin >= active_margin_thresh and cand_spanner_label.lower() != "o":
                    final_label = cand_spanner_label
                    decision_source = f"SpanNER Margin Veto (Protected Decisive Margin: {cand_margin:.4f} >= {active_margin_thresh:.2f} against LLM Deletion)"
                else:
                    final_label = llm_label
                    decision_source = f"Generative LLM Resolution (Prompt {ver_key.upper()} with Evidence)"

                ent_dict = {
                    "span_text": cand["span_text"],
                    "start_char": cand["start_char"],
                    "end_char": cand["end_char"],
                    "label": final_label,
                    "spanner_label": cand_spanner_label,
                    "llm_label": llm_label,
                    "u_spanner": u_spanner,
                    "u_local": u_local,
                    "margin": f"{cand_margin:.4f}",
                    "unreliability": f"{u_local:.4f}",
                    "escalated_to_llm": True,
                    "source": decision_source,
                    "confidence": f"{cand['prob']:.4f}",
                    "uncertainty": f"{u_spanner:.4f}",
                    "uncertainty_method": method.upper(),
                    "rationale": rationale,
                    "llm_response": rationale or raw_llm_label
                }
                if novelty is not None:
                    ent_dict["novelty"] = f"{novelty:.4f}"
                ent_dict["combined"] = f"{u_local:.4f}"
                arbitrated_entities.append(ent_dict)

            # Discover and append new unseen spans
            if new_discovered:
                import re
                for new_ent in new_discovered:
                    m_text = new_ent.get("entity", "").strip()
                    m_label = canonicalize_label(new_ent.get("final_label", "O"), use_new_labels=active_use_new_labels)
                    if not m_text or m_label == "O":
                        continue
                    # Find span offsets in abstract text
                    pattern = re.escape(m_text)
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        s_char, e_char = match.start(), match.end()
                        already_present = any(
                            e["start_char"] == s_char and e["end_char"] == e_char
                            for e in arbitrated_entities
                        )
                        if not already_present:
                            new_dict = {
                                "span_text": text[s_char:e_char],
                                "start_char": s_char,
                                "end_char": e_char,
                                "label": m_label,
                                "spanner_label": "O",
                                "llm_label": m_label,
                                "u_spanner": 1.0,
                                "u_local": 1.0,
                                "margin": "0.0000",
                                "unreliability": "1.0000",
                                "escalated_to_llm": True,
                                "source": "Generative LLM Unseen Span Discovery (v1_allowed_new_span)",
                                "confidence": "1.0000",
                                "uncertainty": "1.0000",
                                "uncertainty_method": method.upper(),
                                "rationale": new_ent.get("rationale", "Discovered unseen LSF entity by LLM"),
                                "llm_response": new_ent.get("rationale", m_label)
                            }
                            arbitrated_entities.append(new_dict)
                            break  # Register first clean occurrence

            return arbitrated_entities

        # Branch 2: Legacy SLIMER-style numbered prompt
        else:
            prompt = build_lsf_linkner_batch_prompt(text, uncertain_candidates, enable_few_shot=enable_few_shot, use_new_labels=active_use_new_labels)
            try:
                raw_response = self.llm_provider.generate(prompt)
            except Exception as e:
                print(f"[Warning] Generative LLM API call error: {e}. Falling back to Primary SpanNER labels.")
                raw_response = ""

            # Parse the batch response
            lines = raw_response.split('\n')
            response_map = {}
            import re
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^(\d+)([a-z])?[\.\:\-]?\s*(.*)', line, re.IGNORECASE)
                if match:
                    idx = int(match.group(1)) - 1
                    rest_text = match.group(3)
                    if idx not in response_map:
                        response_map[idx] = []
                    sub_text_match = re.search(r'["\']([^"\']+)["\']', rest_text)
                    sub_text = sub_text_match.group(1) if sub_text_match else None
                    response_map[idx].append((sub_text, rest_text))

            if not response_map:
                raw_clean = raw_response.strip().lower()
                if "none of the above" in raw_clean or raw_clean == "o":
                    for i in range(len(uncertain_candidates)):
                        response_map[i] = [(None, "O")]
                else:
                    for cat in category_defs.keys():
                        if cat.lower() in raw_clean:
                            for i in range(len(uncertain_candidates)):
                                response_map[i] = [(None, cat)]
                            break

            arbitrated_entities = []
            for i, cand in enumerate(uncertain_candidates):
                entries = response_map.get(i, [(None, "")])

                for sub_text, llm_response_line in entries:
                    llm_label = "O"
                    for cat in category_defs.keys():
                        if cat.lower() in llm_response_line.lower():
                            llm_label = cat
                            break
                    if llm_label == "O":
                        for cat in ALL_KNOWN_LABELS:
                            if cat != "O" and cat.lower() in llm_response_line.lower():
                                llm_label = canonicalize_label(cat, use_new_labels=active_use_new_labels)
                                break

                    cand_spanner_label = canonicalize_label(cand["spanner_label"], use_new_labels=active_use_new_labels)
                    cand_margin = float(cand.get("margin", cand.get("prob", 0.0)))
                    u_local = cand["unreliability"]
                    u_spanner = cand["u_score"]
                    novelty = cand.get("novelty")

                    if active_use_veto and llm_label == "O" and cand_margin >= active_margin_thresh and cand_spanner_label.lower() != "o":
                        final_label = cand_spanner_label
                        decision_source = f"SpanNER Margin Veto (Protected Decisive Margin: {cand_margin:.4f} >= {active_margin_thresh:.2f} against LLM Deletion)"
                    else:
                        final_label = llm_label
                        decision_source = f"Generative LLM Resolution (Resolved High Unreliability {method.upper()}: {u_local:.4f})"

                    parent_start = cand["start_char"]
                    parent_end = cand["end_char"]
                    parent_text = cand["span_text"]

                    if sub_text and sub_text in parent_text:
                        rel_idx = parent_text.find(sub_text)
                        char_start = parent_start + rel_idx
                        char_end = char_start + len(sub_text)
                        span_text = sub_text
                    else:
                        char_start = parent_start
                        char_end = parent_end
                        span_text = parent_text

                    ent_dict = {
                        "span_text": span_text,
                        "start_char": char_start,
                        "end_char": char_end,
                        "label": final_label,
                        "spanner_label": cand_spanner_label,
                        "llm_label": llm_label,
                        "u_spanner": u_spanner,
                        "u_local": u_local,
                        "margin": f"{cand_margin:.4f}",
                        "unreliability": f"{u_local:.4f}",
                        "escalated_to_llm": True,
                        "source": decision_source,
                        "confidence": f"{cand['prob']:.4f}",
                        "uncertainty": f"{cand['u_score']:.4f}",
                        "uncertainty_method": method.upper(),
                        "llm_response": llm_response_line
                    }
                    if novelty is not None:
                        ent_dict["novelty"] = f"{novelty:.4f}"
                    ent_dict["combined"] = f"{u_local:.4f}"
                    arbitrated_entities.append(ent_dict)

            return arbitrated_entities

    def predict_batch(
        self,
        texts: list,
        batch_size: int = None,
        uncertainty_method: str = None,
        uncertainty_threshold: float = None,
        novelty_threshold: float = None,
        gating_mode: str = None,
        w_novelty: float = None,
        w_uncertainty: float = None,
        mcd_passes: int = None,
        use_nms: bool = None,
        prompt_version: str = None,
        margin_threshold: float = None,
        use_margin_safeguard: bool = None,
        enable_few_shot: bool = True,
        arbitrate_by_uncertainty: bool = None,
        use_new_labels: bool = None,
        verbose: bool = True,
        doc_ids: list = None,
        precomputed_candidates: Any = None
    ) -> list:
        """
        Runs the complete LinkNER RDC workflow in batch mode across multiple abstracts:
        1. Recognition: SpanNER evaluates batches of B abstracts (default B = 5).
        2. Filtering: Preserves candidate mentions (with optional NMS).
        3. Detection: Parallel individual LOF novelty scoring per candidate entity.
        4. Gating: Applies chosen gating operator (PROB_OR, WEIGHTED, OR, AND, COPULA).
        5. Classification: Exactly 1 consolidated batch LLM call per abstract for all escalated spans.
        """
        if not texts:
            return []

        active_bs = batch_size or getattr(self.config, "batch_size", 5)
        active_use_new_labels = self.use_new_labels if use_new_labels is None else use_new_labels
        active_gating_mode = (gating_mode or self.gating_mode or "prob_or").lower()
        active_nov_thresh = novelty_threshold if novelty_threshold is not None else self.novelty_threshold
        active_w_nov = self.w_novelty if w_novelty is None else w_novelty
        active_w_unc = self.w_uncertainty if w_uncertainty is None else w_uncertainty
        active_use_nms = self.use_nms if use_nms is None else use_nms
        active_prompt_ver = prompt_version or self.prompt_version or "v1_with_label"
        active_margin_thresh = margin_threshold if margin_threshold is not None else self.margin_threshold
        active_use_veto = use_margin_safeguard if use_margin_safeguard is not None else self.use_margin_safeguard
        active_mcd_passes = mcd_passes or self.mcd_passes

        method = (uncertainty_method or self.uncertainty_method).lower()
        if method not in ["lc", "pe", "mcd", "enn", "margin"]:
            raise ValueError(f"Unknown uncertainty method: '{method}'. Choose from ['lc', 'pe', 'mcd', 'enn', 'margin'].")

        default_thresh = 0.35 if uncertainty_threshold is None else uncertainty_threshold
        threshold = default_thresh if uncertainty_threshold is not None else self.uncertainty_threshold

        all_abstract_results = []
        total_abstracts = len(texts)
        total_batches = (total_abstracts + active_bs - 1) // active_bs

        if verbose:
            if self.novelty_scorer is not None or precomputed_candidates is not None:
                if active_gating_mode in ("or", "max", "disjunctive"):
                    nov_desc = f"LOF Novelty Detector [OR Gating: Unc>={threshold} OR Nov>={active_nov_thresh}]"
                elif active_gating_mode in ("and", "min", "conjunctive"):
                    nov_desc = f"LOF Novelty Detector [AND Gating: Unc>={threshold} AND Nov>={active_nov_thresh}]"
                elif active_gating_mode in ("prob_or", "union"):
                    nov_desc = f"LOF Novelty Detector [PROB_OR Union Gating: 1-(1-N)*(1-U) >= {threshold:.2f}]"
                elif active_gating_mode in ("copula", "correlated"):
                    nov_desc = f"LOF Novelty Detector [Correlated Copula Gating: U + N - 1.5*(U*N) >= {threshold:.2f}]"
                else:
                    nov_desc = f"LOF Novelty Detector [Weighted Convex Gating: {active_w_nov:.2f}*Nov + {active_w_unc:.2f}*Unc >= {threshold:.2f}]"
            else:
                nov_desc = "Novelty Disabled (Pure Uncertainty)"
            schema_desc = "New Relabeled Ontology" if active_use_new_labels else "Legacy Ontology"
            nms_desc = "ENABLED (Longest-Span First)" if active_use_nms else "DISABLED (Raw Candidates Preserved)"
            print("=" * 96)
            print(f"  🚀 LinkNER Batch Engine: Processing {total_abstracts} abstracts in {total_batches} batches (B={active_bs})")
            print(f"  ⚡ Method: {method.upper()} | Gating Strategy: {nov_desc}")
            print(f"  🎯 NMS Policy: {nms_desc} | Prompt: {active_prompt_ver}")
            print(f"  🏷️  Schema: {schema_desc} (use_new_labels={active_use_new_labels})")
            print("=" * 96)

        # Process input abstracts in chunks of batch_size (default B = 5)
        for chunk_start in range(0, total_abstracts, active_bs):
            chunk_texts = texts[chunk_start : chunk_start + active_bs]
            chunk_doc_ids = doc_ids[chunk_start : chunk_start + active_bs] if doc_ids else [f"doc_{i+1}" for i in range(chunk_start, chunk_start + len(chunk_texts))]
            current_bs = len(chunk_texts)
            batch_num = (chunk_start // active_bs) + 1

            if verbose:
                print(f"\n==========================================================================")
                print(f"📦 [Batch {batch_num}/{total_batches}] Processing abstracts {chunk_start + 1}-{chunk_start + current_bs} of {total_abstracts} (Size: {current_bs})")
                print(f"==========================================================================")

            chunk_pending_by_batch = [[] for _ in range(current_bs)]

            if precomputed_candidates is not None:
                if verbose and chunk_start == 0:
                    print("  ⚡ [Stage 1: Precomputed Cache] Bypassing neural forward pass; consuming cached candidate spans directly.")
                for b_idx in range(current_bs):
                    global_idx = chunk_start + b_idx
                    doc_key = chunk_doc_ids[b_idx] if (doc_ids and b_idx < len(chunk_doc_ids)) else None
                    cands = None
                    if isinstance(precomputed_candidates, dict):
                        if doc_key and doc_key in precomputed_candidates:
                            cands = precomputed_candidates[doc_key]
                        elif global_idx in precomputed_candidates:
                            cands = precomputed_candidates[global_idx]
                        elif str(global_idx) in precomputed_candidates:
                            cands = precomputed_candidates[str(global_idx)]
                    elif isinstance(precomputed_candidates, list) and global_idx < len(precomputed_candidates):
                        cands = precomputed_candidates[global_idx]

                    if isinstance(cands, dict) and "candidates" in cands:
                        cands = cands["candidates"]
                    if cands is None:
                        cands = []

                    for s_idx, c in enumerate(cands):
                        s_text = c.get("span_text", c.get("entity", "")).strip()
                        if not s_text:
                            continue
                        c_start = int(c.get("start_char", c.get("char_start", 0)))
                        c_end = int(c.get("end_char", c.get("char_end", 0)))
                        raw_lbl = c.get("predicted_label", c.get("spanner_label", "O"))
                        spanner_lbl = to_new_label(raw_lbl) if active_use_new_labels else raw_lbl
                        raw_top2 = c.get("second_best_label", c.get("top2_label", "O"))
                        top2_lbl = to_new_label(raw_top2) if active_use_new_labels else raw_top2

                        prob_val = float(c.get("prob", c.get("confidence", 1.0)))
                        u_val = float(c.get("uncertainty", c.get("u_score", 0.0)))
                        margin_val = float(c.get("margin", prob_val))
                        top2_p = float(c.get("top2_prob", 0.0))
                        po_val = float(c.get("p_background_o", c.get("p_o", 0.0)))
                        nov_val = float(c.get("novelty_score", c.get("novelty", 0.0)))
                        lbl_id = 0 if spanner_lbl == "O" else 1

                        cand_item = {
                            "b_idx": b_idx,
                            "s_idx": s_idx,
                            "start_char": c_start,
                            "end_char": c_end,
                            "char_start": c_start,
                            "char_end": c_end,
                            "span_text": s_text,
                            "label_id": lbl_id,
                            "prob": prob_val,
                            "confidence": prob_val,
                            "u_score": u_val,
                            "uncertainty": u_val,
                            "margin": margin_val,
                            "top2_prob": top2_p,
                            "p_background_o": po_val,
                            "prob_o": po_val,
                            "predicted_label": spanner_lbl,
                            "spanner_label": spanner_lbl,
                            "second_best_label": top2_lbl,
                            "novelty": nov_val,
                            "novelty_score": nov_val
                        }
                        chunk_pending_by_batch[b_idx].append(cand_item)
            else:
                # Stage 1: Batch Tokenization & SpanNER Neural Evaluation
                encoding = self.tokenizer(
                    chunk_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True,
                    return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)  # [B, Seq_Len]
                attention_mask = encoding["attention_mask"].to(self.device)  # [B, Seq_Len]
                offsets = encoding["offset_mapping"]  # [B, Seq_Len, 2]

                with torch.no_grad():
                    outputs = self.spanner_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"]  # [B, Num_Spans, Num_Classes]
                    candidate_spans = outputs["candidate_spans"]

                # Compute uncertainties lazily for active method
                unc_dict = self._compute_all_uncertainties(
                    logits,
                    input_ids=input_ids if method in ("mcd", "margin") else None,
                    attention_mask=attention_mask if method in ("mcd", "margin") else None,
                    mcd_passes=active_mcd_passes,
                    active_method=method,
                    compute_all=False
                )

                active_data = unc_dict[method]
                pred_label_ids = active_data["pred_labels"]  # [B, Num_Spans]
                max_probs = active_data["max_probs"]  # [B, Num_Spans]
                active_uncertainties = active_data["uncertainty"]  # [B, Num_Spans]
                active_margins = active_data.get("margins", max_probs)  # [B, Num_Spans]
                top2_lbl_ids = active_data.get("top2_labels", pred_label_ids)
                top2_probs_val = active_data.get("top2_probs", max_probs)
                p_o_val = active_data.get("p_o", max_probs)

                if verbose:
                    total_spans_eval = current_bs * len(candidate_spans)
                    print(f"  ⚡ [Stage 1: SpanNER] Forward pass evaluated {total_spans_eval} span predictions across {current_bs} abstracts.")

                # Stage 1b: Fast GPU-Side Boolean Masking
                keep_mask = (pred_label_ids != 0) | (active_uncertainties >= threshold)
                b_indices, s_indices = torch.where(keep_mask)

                if len(b_indices) > 0:
                    b_idx_arr = b_indices.cpu().numpy()
                    s_idx_arr = s_indices.cpu().numpy()
                    label_id_arr = pred_label_ids[b_indices, s_indices].cpu().numpy()
                    u_score_arr = active_uncertainties[b_indices, s_indices].cpu().numpy()
                    max_prob_arr = max_probs[b_indices, s_indices].cpu().numpy()
                    margin_arr = active_margins[b_indices, s_indices].cpu().numpy() if isinstance(active_margins, torch.Tensor) else max_prob_arr
                    top2_id_arr = top2_lbl_ids[b_indices, s_indices].cpu().numpy() if isinstance(top2_lbl_ids, torch.Tensor) else label_id_arr
                    top2_p_arr = top2_probs_val[b_indices, s_indices].cpu().numpy() if isinstance(top2_probs_val, torch.Tensor) else max_prob_arr
                    po_arr = p_o_val[b_indices, s_indices].cpu().numpy() if isinstance(p_o_val, torch.Tensor) else max_prob_arr
                    offsets_np = offsets.cpu().numpy()

                    for i in range(len(b_idx_arr)):
                        b_idx = int(b_idx_arr[i])
                        s_idx = int(s_idx_arr[i])
                        start, end, _ = candidate_spans[s_idx]

                        char_start = int(offsets_np[b_idx, start, 0])
                        char_end = int(offsets_np[b_idx, end, 1])
                        span_text = chunk_texts[b_idx][char_start:char_end].strip()
                        if not span_text:
                            continue

                        raw_lbl = self.id2label[int(label_id_arr[i])]
                        spanner_lbl = to_new_label(raw_lbl) if active_use_new_labels else raw_lbl
                        raw_top2 = self.id2label[int(top2_id_arr[i])]
                        top2_lbl = to_new_label(raw_top2) if active_use_new_labels else raw_top2

                        cand_item = {
                            "b_idx": b_idx,
                            "s_idx": s_idx,
                            "start_char": char_start,
                            "end_char": char_end,
                            "char_start": char_start,
                            "char_end": char_end,
                            "span_text": span_text,
                            "label_id": int(label_id_arr[i]),
                            "prob": float(max_prob_arr[i]),
                            "confidence": float(max_prob_arr[i]),
                            "u_score": float(u_score_arr[i]),
                            "uncertainty": float(u_score_arr[i]),
                            "margin": float(margin_arr[i]),
                            "top2_prob": float(top2_p_arr[i]),
                            "p_background_o": float(po_arr[i]),
                            "prob_o": float(po_arr[i]),
                            "predicted_label": spanner_lbl,
                            "spanner_label": spanner_lbl,
                            "second_best_label": top2_lbl,
                        }
                        chunk_pending_by_batch[b_idx].append(cand_item)

            # Apply NMS conditionally per abstract
            chunk_pending = []
            for b_idx in range(current_bs):
                spans_for_doc = chunk_pending_by_batch[b_idx]
                if active_use_nms and len(spans_for_doc) > 0:
                    try:
                        nms_kept = run_nms(spans_for_doc)
                    except Exception:
                        nms_kept = spans_for_doc
                else:
                    nms_kept = spans_for_doc
                chunk_pending.extend(nms_kept)

            # Stage 2: Batch Novelty Scoring
            if chunk_pending:
                all_have_nov = all("novelty_score" in p and p.get("novelty_score") is not None for p in chunk_pending)
                if all_have_nov and precomputed_candidates is not None:
                    nov_scores = np.array([float(p["novelty_score"]) for p in chunk_pending], dtype=float)
                    if verbose:
                        print(f"  🔍 [Stage 2: LOF Novelty] Reusing {len(chunk_pending)} precomputed novelty scores from cache.")
                elif self.novelty_scorer is not None:
                    all_span_texts = [p["span_text"] for p in chunk_pending]
                    all_span_vecs = self.novelty_scorer.embed(all_span_texts)
                    nov_scores = self.novelty_scorer.novelty(all_span_vecs)
                    if verbose:
                        print(f"  🔍 [Stage 2: LOF Novelty] Embedded and scored {len(chunk_pending)} non-background mentions in parallel.")
                else:
                    nov_scores = np.zeros(len(chunk_pending))
                    if verbose:
                        print(f"  🔍 [Stage 2: Gating] Scored {len(chunk_pending)} candidate mentions against unreliability thresholds (zero novelty).")
            else:
                nov_scores = np.zeros(0)

            # Partition spans per abstract into Confident (Local) vs Uncertain (Escalate to LLM)
            abstract_extracted = [[] for _ in range(current_bs)]
            abstract_uncertain = [[] for _ in range(current_bs)]

            for p_idx, item in enumerate(chunk_pending):
                b_idx = item["b_idx"]
                label_id = item["label_id"]
                prob = item["prob"]
                u_score = item["u_score"]
                novelty = float(nov_scores[p_idx]) if len(nov_scores) > p_idx else 0.0

                item["novelty"] = novelty
                item["novelty_score"] = novelty
                spanner_label = item["spanner_label"]

                # Category-specific uncertainty threshold lookup
                unc_thresh = (
                    self.category_thresholds.get(spanner_label) or
                    self.category_thresholds.get(to_old_label(spanner_label)) or
                    threshold
                )

                # Gating Strategy Resolution
                has_novelty_signal = (self.novelty_scorer is not None) or (len(nov_scores) > 0 and np.any(nov_scores > 0)) or (precomputed_candidates is not None)
                if has_novelty_signal:
                    if active_gating_mode in ("or", "max", "disjunctive"):
                        escalate = (u_score >= unc_thresh) or (novelty >= active_nov_thresh)
                        unreliability = max(u_score, novelty)
                    elif active_gating_mode in ("and", "min", "conjunctive"):
                        escalate = (u_score >= unc_thresh) and (novelty >= active_nov_thresh)
                        unreliability = min(u_score, novelty)
                    elif active_gating_mode in ("prob_or", "union"):
                        # Probabilistic Risk Union
                        unreliability = 1.0 - (1.0 - novelty) * (1.0 - u_score)
                        escalate = unreliability >= threshold
                    elif active_gating_mode in ("copula", "correlated"):
                        copula_beta = float(os.getenv("COPULA_BETA", "0.5"))
                        unreliability = float(np.clip(u_score + novelty - (1.0 + copula_beta) * (u_score * novelty), 0.0, 1.0))
                        escalate = unreliability >= threshold
                    elif active_gating_mode == "harmonic":
                        unreliability = 2.0 * (novelty * u_score) / (novelty + u_score + 1e-6)
                        escalate = unreliability >= threshold
                    else:  # Default: Weighted Convex Sum
                        unreliability = active_w_nov * novelty + active_w_unc * u_score
                        escalate = unreliability >= threshold
                else:
                    unreliability = u_score
                    escalate = unreliability >= unc_thresh

                item["unreliability"] = unreliability
                item["combined"] = unreliability
                item["escalated_to_llm"] = escalate

                if not escalate:
                    # Confident SpanNER Prediction (Accepted Locally)
                    if label_id != 0:
                        ent = {
                            "span_text": item["span_text"],
                            "start_char": item["char_start"],
                            "end_char": item["char_end"],
                            "label": spanner_label,
                            "spanner_label": spanner_label,
                            "llm_label": "❌ (Not Escalated)",
                            "escalated_to_llm": False,
                            "source": f"Primary SpanNER (High Confidence, Method: {method.upper()})",
                            "confidence": f"{prob:.4f}",
                            "uncertainty": f"{u_score:.4f}",
                            "uncertainty_method": method.upper(),
                            "unreliability": f"{unreliability:.4f}",
                        }
                        if novelty is not None:
                            ent["novelty"] = f"{novelty:.4f}"
                        ent["combined"] = f"{unreliability:.4f}"
                        abstract_extracted[b_idx].append(ent)
                else:
                    # High Unreliability (Marked for LLM Escalation)
                    abstract_uncertain[b_idx].append(item)

            # Stage 3: Consolidated LLM Calls with Concurrent Multi-Threaded Dispatch
            num_escalated_docs = sum(1 for u in abstract_uncertain if u)
            total_escalated_spans = sum(len(u) for u in abstract_uncertain)
            if verbose:
                if num_escalated_docs > 0:
                    print(f"  🤖 [Stage 3: LLM Escalation] Escalating {total_escalated_spans} unreliable spans across {num_escalated_docs} abstracts concurrently ({active_prompt_ver})...")
                else:
                    print(f"  ✨ [Stage 3: LLM Escalation] 0 abstracts required escalation (100% accepted locally).")

            escalation_tasks = []
            for b_idx in range(current_bs):
                if abstract_uncertain[b_idx]:
                    escalation_tasks.append((b_idx, chunk_texts[b_idx], abstract_uncertain[b_idx]))

            if escalation_tasks:
                from concurrent.futures import ThreadPoolExecutor
                max_workers = min(len(escalation_tasks), 10)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_bidx = {
                        executor.submit(
                            self._escalate_and_arbitrate_abstract,
                            text,
                            cand_list,
                            method,
                            enable_few_shot,
                            self.arbitrate_by_uncertainty,
                            active_use_new_labels,
                            active_prompt_ver,
                            active_margin_thresh,
                            active_use_veto
                        ): b_idx
                        for b_idx, text, cand_list in escalation_tasks
                    }
                    for future in future_to_bidx:
                        b_idx = future_to_bidx[future]
                        try:
                            res_entities = future.result()
                            abstract_extracted[b_idx].extend(res_entities)
                            if verbose:
                                num_e = len(abstract_uncertain[b_idx])
                                num_r = len(res_entities)
                                num_pos = sum(1 for e in res_entities if e.get("label", "O") != "O")
                                num_pruned = sum(1 for e in res_entities if e.get("label", "O") == "O")
                                num_new = sum(1 for e in res_entities if "Unseen Span Discovery" in e.get("source", ""))
                                new_str = f", {num_new} new discovered" if num_new > 0 else ""
                                print(f"     -> Abstract {b_idx + 1}: Escalated {num_e} spans -> {num_r} arbitrated decisions ({num_pos} positive kept, {num_pruned} pruned to 'O'{new_str}).")

                        except Exception as exc:
                            print(f"⚠️ [Error in LLM Escalation Worker] Abstract {b_idx + 1} generated an exception: {exc}")

            # Merge and sort entities by character start position, deduplicating identical span offsets
            for b_idx in range(current_bs):
                ents = abstract_extracted[b_idx]
                ents.sort(key=lambda x: x["start_char"])
                seen_spans = set()
                deduped_ents = []
                for e in ents:
                    key = (e["start_char"], e["end_char"])
                    if key not in seen_spans:
                        seen_spans.add(key)
                        deduped_ents.append(e)
                all_abstract_results.append(deduped_ents)

        if verbose:
            print(f"\n🏁 [LinkNER Batch Engine] Execution complete across all {total_abstracts} abstracts.")

        return all_abstract_results

    def predict(
        self,
        text: str,
        uncertainty_method: str = None,
        uncertainty_threshold: float = None,
        novelty_threshold: float = None,
        gating_mode: str = None,
        w_novelty: float = None,
        w_uncertainty: float = None,
        mcd_passes: int = None,
        use_nms: bool = None,
        prompt_version: str = None,
        margin_threshold: float = None,
        use_margin_safeguard: bool = None,
        enable_few_shot: bool = True,
        spanner_min_confidence: float = 0.0,
        arbitrate_by_uncertainty: bool = None,
        use_new_labels: bool = None,
        doc_id: str = None,
        precomputed_candidates: Any = None
    ) -> list:
        """
        Runs the LinkNER RDC workflow on a single abstract (wrapper around predict_batch).
        """
        cands_input = None
        if precomputed_candidates is not None:
            if isinstance(precomputed_candidates, dict):
                cands_input = precomputed_candidates
            else:
                cands_input = [precomputed_candidates]

        results = self.predict_batch(
            texts=[text],
            batch_size=1,
            uncertainty_method=uncertainty_method,
            uncertainty_threshold=uncertainty_threshold,
            novelty_threshold=novelty_threshold,
            gating_mode=gating_mode,
            w_novelty=w_novelty,
            w_uncertainty=w_uncertainty,
            mcd_passes=mcd_passes,
            use_nms=use_nms,
            prompt_version=prompt_version,
            margin_threshold=margin_threshold,
            use_margin_safeguard=use_margin_safeguard,
            enable_few_shot=enable_few_shot,
            arbitrate_by_uncertainty=arbitrate_by_uncertainty,
            use_new_labels=use_new_labels,
            verbose=False,
            doc_ids=[doc_id] if doc_id else None,
            precomputed_candidates=cands_input
        )
        return results[0] if results else []


