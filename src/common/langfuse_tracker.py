"""
Langfuse Observability Tracker for Post-BERT ACE Multi-Agent System.

Provides thread-safe hierarchical tracing:
  - Root Trace / Span: Scoped per processed biomedical abstract / document
  - Child Generations: Scoped per Agent (ACE-Generator, ACE-Reflector, ACE-Curator)
    capturing prompt input, response output, model details, token usage, and latency.

Includes safe-fail fallback (DummyObservation) when Langfuse is disabled or unreachable.
"""

import os
import sys
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

logger = logging.getLogger("langfuse_tracker")


class DummyObservation:
    """Safe no-op observation returned when Langfuse is disabled or unconfigured."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


class LangfuseTracker:
    """
    Singleton manager for Langfuse client and hierarchical agent tracing.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LangfuseTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes Langfuse client if enabled via environment variables."""
        enabled_val = os.getenv("LANGFUSE_ENABLED", "").strip().lower()
        self.enabled = enabled_val in ("true", "1", "yes", "on")

        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        self.host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "http://localhost:3000"
        ).strip().rstrip("/")

        if not self.enabled:
            logger.debug("[LangfuseTracker] Tracing disabled via LANGFUSE_ENABLED.")
            return

        if not self.public_key or not self.secret_key:
            logger.warning("[LangfuseTracker] LANGFUSE_ENABLED is true, but public or secret key is missing. Tracing disabled.")
            self.enabled = False
            return

        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            logger.info(f"[LangfuseTracker] Connected to Langfuse at {self.host}")
        except Exception as exc:
            logger.error(f"[LangfuseTracker] Failed to initialize Langfuse client: {exc}")
            self.enabled = False
            self._client = None

    def reload_config(self):
        """Reloads configuration from current environment."""
        self._init_client()

    def auth_check(self) -> bool:
        """Verifies authentication against the configured Langfuse instance."""
        if not self.enabled or not self._client:
            return False
        try:
            return bool(self._client.auth_check())
        except Exception as exc:
            logger.error(f"[LangfuseTracker] auth_check failed: {exc}")
            return False

    @contextmanager
    def trace_abstract(
        self,
        doc_id: str,
        abstract: str,
        candidate_spans: Optional[List[Dict[str, Any]]] = None,
        mode: str = "train",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ):
        """
        Creates a root trace / span for an entire abstract processing cycle.
        """
        if not self.enabled or not self._client:
            yield DummyObservation()
            return

        cand_count = len(candidate_spans) if candidate_spans else 0
        trace_name = f"Abstract-{doc_id}"
        meta = {
            "doc_id": doc_id,
            "mode": mode,
            "candidate_spans_count": cand_count,
            **(metadata or {})
        }
        trace_input = {
            "doc_id": doc_id,
            "abstract": abstract,
            "candidate_spans_count": cand_count
        }

        try:
            with self._client.start_as_current_observation(
                name=trace_name,
                as_type="span",
                input=trace_input,
                metadata=meta
            ) as span:
                yield span
        except Exception as exc:
            logger.warning(f"[LangfuseTracker] Error in trace_abstract for {doc_id}: {exc}")
            yield DummyObservation()

    @contextmanager
    def trace_agent(
        self,
        agent_name: str,
        model_name: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Creates a child generation observation for an individual agent call.
        Yields the generation object so the caller can update output, metadata, and token usage.
        """
        if not self.enabled or not self._client:
            yield DummyObservation()
            return

        try:
            with self._client.start_as_current_observation(
                name=agent_name,
                as_type="generation",
                model=model_name,
                input=prompt,
                metadata=metadata or {},
                model_parameters=model_parameters or {}
            ) as gen:
                yield gen
        except Exception as exc:
            logger.warning(f"[LangfuseTracker] Error in trace_agent ({agent_name}): {exc}")
            yield DummyObservation()

    def flush(self):
        """Flushes any buffered observations to Langfuse."""
        if self.enabled and self._client:
            try:
                self._client.flush()
            except Exception as exc:
                logger.warning(f"[LangfuseTracker] Failed to flush events: {exc}")


_tracker_instance = None

def get_langfuse_tracker() -> LangfuseTracker:
    """Returns the singleton LangfuseTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = LangfuseTracker()
    return _tracker_instance
