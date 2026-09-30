"""
TypeSafe System One (Jev) Lightweight Client for JeBert Architecture.
Provides a clean, dependency-minimal Python client for evaluating System One models.
Supports Noul (binary), Choice (categorical), and Score (ordinal) primitives.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.error
    import urllib.request
    HAS_REQUESTS = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from src.common.config import PROJECT_ROOT


def _load_env_file(env_path: Path) -> None:
    """Load simple KEY=VAL pairs from a .env file into os.environ if not already set."""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val


# Load credentials from project root .env
_load_env_file(PROJECT_ROOT / ".env")


@dataclass
class NoulAnswer:
    noul: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreAnswer:
    score: float
    legend: Dict[str, str]
    probabilities: Dict[str, float]
    confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemOneResponse:
    model: str
    answers: Dict[str, Any]
    usage: Dict[str, int]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def nouls(self) -> Dict[str, NoulAnswer]:
        out = {}
        for k, v in self.answers.items():
            if isinstance(v, dict) and v.get("type") == "noul":
                out[k] = NoulAnswer(noul=float(v.get("noul", 0.0)), raw=v)
        return out

    @property
    def choices(self) -> Dict[str, ChoiceAnswer]:
        out = {}
        for k, v in self.answers.items():
            if isinstance(v, dict) and v.get("type") == "choice":
                out[k] = ChoiceAnswer(
                    choice=v.get("choice", ""),
                    probabilities=v.get("probabilities", {}),
                    confidence=float(v.get("confidence", 0.0)),
                    raw=v,
                )
        return out

    @property
    def scores(self) -> Dict[str, ScoreAnswer]:
        out = {}
        for k, v in self.answers.items():
            if isinstance(v, dict) and v.get("type") == "score":
                out[k] = ScoreAnswer(
                    score=float(v.get("score", 0.0)),
                    legend=v.get("legend", {}),
                    probabilities=v.get("probabilities", {}),
                    confidence=float(v.get("confidence", 0.0)),
                    raw=v,
                )
        return out


class TypeSafeClient:
    """
    Lightweight client for TypeSafe System One API (Jev).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ):
        self.api_key = (
            api_key
            or os.environ.get("TYPESAFE_API_KEY")
            or "apikey_21758b9c8afe51ba433e8eab8af8cc98440b_9e68101bce7858111ca8dc674572c97c22d7753598755e06b474b547de6151af"
        )
        self.endpoint = (
            endpoint
            or os.environ.get("TYPESAFE_ENDPOINT")
            or "https://api.typesafe.ai/v1/systemone"
        )
        self.default_model = (
            model
            or os.environ.get("TYPESAFE_MODEL")
            or "jev-latest"
        )
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "TypeSafe API key not found. Set TYPESAFE_API_KEY environment variable "
                "or pass api_key to TypeSafeClient()."
            )

    def system_one(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> SystemOneResponse:
        """
        Evaluate a state against a map of typed questions.
        """
        payload = {
            "state": state,
            "model": model or self.default_model,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Unified-LSFNER-JeBert/1.0",
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        delay = 1.0

        for attempt in range(self.max_retries + 1):
            try:
                if HAS_REQUESTS:
                    resp = requests.post(
                        self.endpoint,
                        headers=headers,
                        data=data_bytes,
                        timeout=self.timeout,
                    )
                    status_code = resp.status_code
                    if status_code in (429, 529) and attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    if status_code != 200:
                        raise RuntimeError(
                            f"TypeSafe API HTTP {status_code}: {resp.text}"
                        )
                    res_json = resp.json()
                else:
                    req = urllib.request.Request(
                        self.endpoint,
                        data=data_bytes,
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        res_bytes = response.read()
                        res_json = json.loads(res_bytes.decode("utf-8"))

                return SystemOneResponse(
                    model=res_json.get("model", ""),
                    answers=res_json.get("answers", {}),
                    usage=res_json.get("usage", {}),
                    raw=res_json,
                )

            except Exception as exc:
                if attempt < self.max_retries and ("429" in str(exc) or "529" in str(exc)):
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise RuntimeError("Exceeded maximum retries for TypeSafe System One API.")

    async def system_one_async(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
        http_client: Optional[Any] = None,
    ) -> SystemOneResponse:
        """
        Evaluate state against typed questions asynchronously.
        Uses httpx.AsyncClient if available or asyncio.to_thread fallback.
        """
        if not HAS_HTTPX:
            return await asyncio.to_thread(self.system_one, state, questions, model)

        payload = {
            "state": state,
            "model": model or self.default_model,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Unified-LSFNER-JeBert/1.0",
        }

        delay = 1.0
        should_close = False
        client = http_client
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close = True

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                    )
                    status_code = resp.status_code
                    if status_code in (429, 529) and attempt < self.max_retries:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    if status_code != 200:
                        raise RuntimeError(
                            f"TypeSafe API HTTP {status_code}: {resp.text}"
                        )
                    res_json = resp.json()
                    return SystemOneResponse(
                        model=res_json.get("model", ""),
                        answers=res_json.get("answers", {}),
                        usage=res_json.get("usage", {}),
                        raw=res_json,
                    )
                except Exception as exc:
                    if attempt < self.max_retries and ("429" in str(exc) or "529" in str(exc)):
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    raise
            raise RuntimeError("Exceeded maximum retries for TypeSafe System One async API.")
        finally:
            if should_close:
                await client.aclose()

    @staticmethod
    def noul_question(
        instructions: Union[str, Dict[str, Any]],
        criteria: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Helper to create a Noul (yes/no) question."""
        q = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = criteria
        return q

    @staticmethod
    def choice_question(
        instructions: Union[str, Dict[str, Any]],
        criteria: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Helper to create a Choice question."""
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }

    @staticmethod
    def score_question(
        instructions: Union[str, Dict[str, Any]],
        criteria: List[str],
    ) -> Dict[str, Any]:
        """Helper to create a Score question."""
        return {
            "type": "score",
            "instructions": instructions,
            "criteria": criteria,
        }
