"""
TypeSafe System One (Jev) Evaluator for JeBert Architecture.

Provides fast, independent semantic classification for candidate spans.
In JeBert, Jev acts as an independent semantic classifier:
  - Can operate in unbiased mode (SpanNER's predicted label is withheld), producing a genuine independent prior.
  - Supports all Jev modes:
      * 'two_stage': Two-stage dynamic rule assignment + scoped classification.
      * 'mode1': Zero guidelines, direct canonical LSF classification.
      * 'mode2': Helpful guidebook rules.
      * 'mode3': Full guidebook rules.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None

from src.architectures.JeBert.client import TypeSafeClient
from src.architectures.JeBert.jev_generator import (
    extract_sentence_for_span,
    resolve_multi_rule_assignments,
    LSF_CHOICE_CRITERIA,
    normalize_label,
    load_guidebook_rules,
    find_latest_guidebook,
    JevGenerator,
)
from src.common.config import PROJECT_ROOT, OUTPUT_DIR


class JeBertJevEvaluator:
    """
    Independent System One (Jev) evaluator for candidate spans proposed by SpanNER.
    """

    def __init__(
        self,
        mode: str = "two_stage",
        unbiased: bool = True,
        guidebook_path: Optional[str] = None,
        model_name: str = "jev-latest",
        batch_size: int = 4,
        max_assigned_rules: int = 4,
        client: Optional[TypeSafeClient] = None,
    ):
        """
        Args:
            mode: 'two_stage' (default), 'mode1', 'mode2', or 'mode3'.
            unbiased: If True, SpanNER's predicted label is withheld from Jev's prompt.
            guidebook_path: Path to dynamicGuidebook.json. Auto-resolved if None.
            model_name: Jev model identifier.
            batch_size: Max spans evaluated per API call chunk.
            max_assigned_rules: Max rules assigned per span in two-stage mode.
            client: Optional pre-configured TypeSafeClient.
        """
        self.mode = mode
        self.unbiased = unbiased
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_assigned_rules = max_assigned_rules
        self.client = client or TypeSafeClient(model=model_name)

        runs_dir = PROJECT_ROOT / "output" / "bertized_ace" / "runs"
        try:
            gb_file = Path(guidebook_path) if guidebook_path else find_latest_guidebook(runs_dir)
        except Exception:
            gb_file = None

        self.guidebook_path = gb_file
        if gb_file and gb_file.exists():
            helpful, full = load_guidebook_rules(gb_file)
            self.helpful_rules = helpful
            self.full_rules = full
        else:
            self.helpful_rules = []
            self.full_rules = []

        target_rules = self.helpful_rules if self.mode in ("two_stage", "mode2") else self.full_rules
        self.active_rules = target_rules

        self.rules_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in self.active_rules:
            cat = r.get("target_category", "O")
            self.rules_by_category[cat].append(r)

        # Pre-initialize calibrated JevGenerator once for high-throughput reuse
        self.two_stage_generator = JevGenerator(
            mode=self.mode,
            max_assigned_rules=self.max_assigned_rules,
            guidebook_path=str(self.guidebook_path) if self.guidebook_path else None,
            client=self.client,
        )

    def get_scoped_rules_for_span(
        self,
        span_text: str,
        hypothesis_label: Optional[str] = None,
        max_scoped: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Selects candidate rules for a span.
        If hypothesis_label is provided, scopes based on it; otherwise, matches keywords
        and universal boundary rules.
        """
        scoped: List[Dict[str, Any]] = []
        if hypothesis_label and hypothesis_label in self.rules_by_category and hypothesis_label != "O":
            scoped.extend(self.rules_by_category[hypothesis_label])

        # Keyword matching against span_text
        s_lower = span_text.lower()
        for cat, r_list in self.rules_by_category.items():
            if cat != "O":
                cat_words = cat.lower().replace("_", " ").split()
                if any(w in s_lower for w in cat_words if len(w) > 3):
                    scoped.extend(r_list)

        # Include universal boundary/exclusion rules from 'O'
        o_rules = self.rules_by_category.get("O", [])
        for r in o_rules:
            if r.get("id") in ("DG-0026", "DG-0038", "DG-0029", "DG-0020", "DG-0013", "DG-0016", "DG-0024"):
                scoped.append(r)

        # Deduplicate preserving order
        seen = set()
        final: List[Dict[str, Any]] = []
        for r in scoped:
            if r["id"] not in seen:
                seen.add(r["id"])
                final.append(r)
            if len(final) >= max_scoped:
                break

        if len(final) < 3:
            for r in o_rules:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    final.append(r)
                if len(final) >= max_scoped:
                    break

        return final

    def evaluate_spans(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Synchronously evaluates candidate spans with Jev.
        """
        if not candidate_spans:
            return []

        if self.mode == "two_stage":
            return self._evaluate_two_stage(abstract, candidate_spans)
        else:
            return self._evaluate_direct(abstract, candidate_spans)

    async def evaluate_spans_async(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously evaluates candidate spans with Jev.
        """
        if not candidate_spans:
            return []

        if self.mode == "two_stage":
            return await self._evaluate_two_stage_async(abstract, candidate_spans, http_client)
        else:
            return await self._evaluate_direct_async(abstract, candidate_spans, http_client)

    def _evaluate_direct(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Direct 1-pass classification against LSF_CHOICE_CRITERIA."""
        results: List[Dict[str, Any]] = []
        chunk_size = self.batch_size

        for start_idx in range(0, len(candidate_spans), chunk_size):
            batch = candidate_spans[start_idx : start_idx + chunk_size]
            state_spans = {}
            questions = {}

            for idx, c in enumerate(batch):
                sid = f"s{idx}"
                s_text = c.get("span_text", c.get("entity", "")).strip()
                sent_text = extract_sentence_for_span(
                    abstract, s_text, c.get("start_char", -1), c.get("end_char", -1)
                )

                state_spans[sid] = {
                    "span_text": s_text,
                    "sentence_context": sent_text,
                }
                questions[sid] = {
                    "type": "choice",
                    "instructions": (
                        f"You are the JeBert Semantic Classifier. Evaluate candidate span `candidate_spans.{sid}.span_text` "
                        f"within sentence `candidate_spans.{sid}.sentence_context` in `abstract`. "
                        f"Classify this entity into one of the 9 Lifestyle Factor categories or Non-LSF (O)."
                    ),
                    "criteria": LSF_CHOICE_CRITERIA,
                }

            try:
                resp = self.client.system_one(
                    state={"abstract": abstract, "candidate_spans": state_spans},
                    questions=questions,
                )
                for idx, c in enumerate(batch):
                    sid = f"s{idx}"
                    cid = c.get("id", c.get("span_id", start_idx + idx))
                    ans = resp.choices.get(sid)
                    lbl = normalize_label(ans.choice) if ans else "O"
                    conf = ans.confidence if ans else 0.0
                    probs = ans.probabilities if ans else {}
                    results.append({
                        "id": cid,
                        "span_text": c.get("span_text", ""),
                        "start_char": c.get("start_char", 0),
                        "end_char": c.get("end_char", 0),
                        "label": lbl,
                        "final_label": lbl,
                        "confidence": conf,
                        "probabilities": probs,
                        "assigned_guidelines": [],
                        "source": f"JeBert-Jev ({self.mode})",
                    })
            except Exception as exc:
                for idx, c in enumerate(batch):
                    cid = c.get("id", c.get("span_id", start_idx + idx))
                    results.append({
                        "id": cid,
                        "span_text": c.get("span_text", ""),
                        "start_char": c.get("start_char", 0),
                        "end_char": c.get("end_char", 0),
                        "label": "O",
                        "final_label": "O",
                        "confidence": 0.0,
                        "probabilities": {},
                        "assigned_guidelines": [],
                        "source": f"JeBert-Jev-Error: {exc}",
                    })

        return results

    async def _evaluate_direct_async(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronous direct 1-pass classification."""
        results: List[Dict[str, Any]] = []
        chunk_size = self.batch_size
        tasks = []
        sub_batches = []

        for start_idx in range(0, len(candidate_spans), chunk_size):
            batch = candidate_spans[start_idx : start_idx + chunk_size]
            sub_batches.append((start_idx, batch))
            state_spans = {}
            questions = {}

            for idx, c in enumerate(batch):
                sid = f"s{idx}"
                s_text = c.get("span_text", c.get("entity", "")).strip()
                sent_text = extract_sentence_for_span(
                    abstract, s_text, c.get("start_char", -1), c.get("end_char", -1)
                )

                state_spans[sid] = {
                    "span_text": s_text,
                    "sentence_context": sent_text,
                }
                questions[sid] = {
                    "type": "choice",
                    "instructions": (
                        f"You are the JeBert Semantic Classifier. Evaluate candidate span `candidate_spans.{sid}.span_text` "
                        f"within sentence `candidate_spans.{sid}.sentence_context` in `abstract`. "
                        f"Classify this entity into one of the 9 Lifestyle Factor categories or Non-LSF (O)."
                    ),
                    "criteria": LSF_CHOICE_CRITERIA,
                }

            tasks.append(
                self.client.system_one_async(
                    state={"abstract": abstract, "candidate_spans": state_spans},
                    questions=questions,
                    http_client=http_client,
                )
            )

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for (start_idx, batch), resp in zip(sub_batches, responses):
                if isinstance(resp, Exception):
                    for idx, c in enumerate(batch):
                        cid = c.get("id", c.get("span_id", start_idx + idx))
                        results.append({
                            "id": cid,
                            "span_text": c.get("span_text", ""),
                            "start_char": c.get("start_char", 0),
                            "end_char": c.get("end_char", 0),
                            "label": "O",
                            "final_label": "O",
                            "confidence": 0.0,
                            "probabilities": {},
                            "assigned_guidelines": [],
                            "source": f"JeBert-Jev-Error: {resp}",
                        })
                    continue

                for idx, c in enumerate(batch):
                    sid = f"s{idx}"
                    cid = c.get("id", c.get("span_id", start_idx + idx))
                    ans = resp.choices.get(sid)
                    lbl = normalize_label(ans.choice) if ans else "O"
                    conf = ans.confidence if ans else 0.0
                    probs = ans.probabilities if ans else {}
                    results.append({
                        "id": cid,
                        "span_text": c.get("span_text", ""),
                        "start_char": c.get("start_char", 0),
                        "end_char": c.get("end_char", 0),
                        "label": lbl,
                        "final_label": lbl,
                        "confidence": conf,
                        "probabilities": probs,
                        "assigned_guidelines": [],
                        "source": f"JeBert-Jev ({self.mode})",
                    })

        return results

    def _evaluate_two_stage(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Synchronous two-stage evaluation."""
        res = self.two_stage_generator.run(abstract=abstract, candidate_spans=candidate_spans)
        return res.get("llm_answer", [])

    async def _evaluate_two_stage_async(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronous two-stage evaluation."""
        res = await self.two_stage_generator.run_async(
            abstract=abstract,
            candidate_spans=candidate_spans,
            http_client=http_client,
        )
        return res.get("llm_answer", [])
