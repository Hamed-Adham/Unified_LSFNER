"""
JeBert End-to-End Pipeline Orchestrator.

Integrates:
  1. Dense Neural Span Proposal (SpanNER)
  2. Fast Semantic Judgment (TypeSafe Jev Evaluator)
  3. Dual-Model Gating & Escalation Engine (JeBertGating)
  4. Heavy LLM Escalation Arbitrator (JeBertArbitrator)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None

from src.common.config import JEBERT_DIR, PROJECT_ROOT
from src.common.label_mapping import canonicalize_label
from src.architectures.JeBert.gating import JeBertGating
from src.architectures.JeBert.jev_evaluator import JeBertJevEvaluator
from src.architectures.JeBert.arbitrator import JeBertArbitrator


class JeBertPipeline:
    """
    JeBert Unified Pipeline:
    SpanNER + TypeSafe System One (Jev) + Dual-Model Gating + Heavy LLM Escalation.
    """

    def __init__(
        self,
        jev_mode: str = "two_stage",
        gating_mode: str = "discrepancy",
        arbitrator_mode: str = "ace_generator",
        unbiased_jev: bool = True,
        guidebook_path: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_backend: Optional[str] = None,
        enable_gating: bool = True,
        enable_llm_arbitration: bool = False,
        min_consensus_conf: float = 0.35,
        max_uncertainty_threshold: float = 0.75,
        novelty_threshold: float = 0.55,
        composite_threshold: float = 0.45,
        enable_margin_safeguard_veto: bool = True,
        margin_threshold: float = 0.50,
        p_o_threshold: float = 0.20,
    ):
        self.jev_mode = jev_mode
        self.gating_mode = gating_mode
        self.arbitrator_mode = arbitrator_mode
        self.enable_gating = enable_gating
        self.enable_llm_arbitration = enable_llm_arbitration

        # Initialize sub-modules
        self.jev_evaluator = JeBertJevEvaluator(
            mode=jev_mode,
            unbiased=unbiased_jev,
            guidebook_path=guidebook_path,
        )

        self.gating = JeBertGating(
            gating_mode=gating_mode,
            min_consensus_conf=min_consensus_conf,
            max_uncertainty_threshold=max_uncertainty_threshold,
            novelty_threshold=novelty_threshold,
            composite_threshold=composite_threshold,
            enable_margin_safeguard_veto=enable_margin_safeguard_veto,
            margin_threshold=margin_threshold,
            p_o_threshold=p_o_threshold,
        )

        self.arbitrator = JeBertArbitrator(
            mode=arbitrator_mode,
            model_name=llm_model_name,
            backend=llm_backend,
        )

        # Operational metrics tracking
        self.metrics = {
            "total_spans": 0,
            "consensus_accepted": 0,
            "escalated_to_llm": 0,
            "label_discrepancies": 0,
            "margin_vetoes_triggered": 0,
            "discovered_entities": 0,
        }

    def reset_metrics(self) -> None:
        for k in self.metrics:
            self.metrics[k] = 0

    async def evaluate_jev_async(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluates candidate spans with Jev (without SpanNER labels if unbiased)."""
        return await self.jev_evaluator.evaluate_spans_async(
            abstract, candidate_spans, http_client=http_client
        )

    def apply_gating(
        self,
        candidate_spans: List[Dict[str, Any]],
        jev_predictions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits candidate spans into (escalated_cands, consensus_cands) using dual-model gating.
        Updates internal workload metrics.
        """
        escalated_cands, consensus_cands = self.gating.filter_candidates(
            candidate_spans, jev_predictions
        )
        self.metrics["total_spans"] += len(candidate_spans)
        self.metrics["consensus_accepted"] += len(consensus_cands)
        self.metrics["escalated_to_llm"] += len(escalated_cands)
        for c in escalated_cands:
            if c.get("is_discrepancy", False):
                self.metrics["label_discrepancies"] += 1
        return escalated_cands, consensus_cands

    def arbitrate_escalated(
        self,
        abstract: str,
        escalated_cands: List[Dict[str, Any]],
        guidelines: str = "",
        use_llm: bool = True,
        use_fast_referee_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Arbitrates escalated candidate spans using LLM or calibrated dual-model referee.
        Applies Margin Safeguard Veto.
        """
        if not escalated_cands:
            return []

        arb_spans: List[Dict[str, Any]] = []
        disc_spans: List[Dict[str, Any]] = []

        if use_llm and self.arbitrator_mode != "none":
            try:
                arb_spans, disc_spans = self.arbitrator.arbitrate(
                    abstract=abstract,
                    escalated_spans=escalated_cands,
                    guidelines=guidelines,
                )
            except Exception as exc:
                if use_fast_referee_fallback:
                    print(f"[WARN] LLM arbitration failed: {exc}. Falling back to Fast Referee.")
                    arb_spans, disc_spans = self._resolve_with_referee(escalated_cands), []
                else:
                    raise exc
        else:
            arb_spans, disc_spans = self._resolve_with_referee(escalated_cands), []

        resolved: List[Dict[str, Any]] = []
        for s in arb_spans:
            init_lbl = s.get("final_label", s.get("label", "O"))
            v_lbl, triggered, reason = self.gating.apply_margin_safeguard_veto(s, init_lbl)
            if triggered:
                self.metrics["margin_vetoes_triggered"] += 1
                s["final_label"] = v_lbl
                s["label"] = v_lbl
                s["source"] = "JeBert-Margin-Veto"
                s["arbitration_rationale"] = reason
            resolved.append(s)

        for d in disc_spans:
            self.metrics["discovered_entities"] += 1
            resolved.append(d)

        return resolved

    def predict_abstract(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        file_name: str = "doc",
        guidelines: str = "",
        save_outputs_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes synchronous JeBert inference on a single abstract.
        """
        if not candidate_spans:
            return []

        # Step 1: Jev Independent Semantic Evaluation
        jev_predictions = self.jev_evaluator.evaluate_spans(abstract, candidate_spans)

        # Step 2: Dual-Model Gating
        if self.enable_gating:
            escalated_cands, consensus_cands = self.gating.filter_candidates(
                candidate_spans, jev_predictions
            )
        else:
            escalated_cands = candidate_spans
            consensus_cands = []

        # Update metrics
        self.metrics["total_spans"] += len(candidate_spans)
        self.metrics["consensus_accepted"] += len(consensus_cands)
        self.metrics["escalated_to_llm"] += len(escalated_cands)
        for c in escalated_cands:
            if c.get("is_discrepancy", False):
                self.metrics["label_discrepancies"] += 1

        final_entities: List[Dict[str, Any]] = []

        # Add consensus candidates directly (0 LLM cost)
        for c in consensus_cands:
            lbl = canonicalize_label(c.get("jev_label", c.get("spanner_label", "O")), use_new_labels=True)
            c_dict = dict(c)
            c_dict["label"] = lbl
            c_dict["final_label"] = lbl
            c_dict["source"] = "JeBert-Consensus-FastPath"
            c_dict["escalated_to_llm"] = False
            final_entities.append(c_dict)

        # Step 3: Arbitrate Escalated Candidates
        if escalated_cands:
            if self.enable_llm_arbitration and self.arbitrator_mode != "none":
                try:
                    arb_spans, disc_spans = self.arbitrator.arbitrate(
                        abstract=abstract,
                        escalated_spans=escalated_cands,
                        guidelines=guidelines,
                    )
                except Exception as exc:
                    print(f"[WARN] JeBert LLM arbitration fallback: {exc}")
                    arb_spans, disc_spans = self._resolve_with_referee(escalated_cands), []
            else:
                # Fast dual-model referee resolution (0 LLM overhead)
                arb_spans, disc_spans = self._resolve_with_referee(escalated_cands), []

            # Step 4: Apply Margin Safeguard Veto to arbitrated results
            for s in arb_spans:
                init_lbl = s.get("final_label", s.get("label", "O"))
                v_lbl, triggered, reason = self.gating.apply_margin_safeguard_veto(s, init_lbl)
                if triggered:
                    self.metrics["margin_vetoes_triggered"] += 1
                    s["final_label"] = v_lbl
                    s["label"] = v_lbl
                    s["source"] = "JeBert-Margin-Veto"
                    s["arbitration_rationale"] = reason
                final_entities.append(s)

            for d in disc_spans:
                self.metrics["discovered_entities"] += 1
                final_entities.append(d)

        final_entities.sort(key=lambda x: (x.get("start_char", 0), x.get("end_char", 0)))

        if save_outputs_dir:
            os.makedirs(save_outputs_dir, exist_ok=True)
            out_file = Path(save_outputs_dir) / f"{file_name}_jebert.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_entities, f, indent=2)

        return final_entities

    def _resolve_with_referee(
        self, escalated_cands: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fast dual-model resolution for escalated/discrepant candidates without invoking a heavy LLM.
        Applies calibrated referee logic combining Jev's semantic prior and SpanNER's confidence.
        """
        resolved: List[Dict[str, Any]] = []
        for c in escalated_cands:
            s_lbl = canonicalize_label(
                c.get("spanner_label", c.get("bert_predicted_label", "O")),
                use_new_labels=True,
            )
            j_lbl = canonicalize_label(
                c.get("jev_label", "O"),
                use_new_labels=True,
            )
            j_conf = float(c.get("jev_confidence", 0.0))
            margin = float(c.get("margin", 0.0))
            u_score = float(c.get("uncertainty", 0.0))
            p_o = float(c.get("p_o", c.get("p_background_o", 0.0)))

            if s_lbl == j_lbl:
                final_lbl = j_lbl
                source = "JeBert-DualModel-Agreement"
            elif j_lbl != "O" and (j_conf >= 0.65 or u_score >= 0.35):
                # Jev found a verified LSF category and SpanNER is uncertain or Jev is confident
                final_lbl = j_lbl
                source = "JeBert-Jev-Semantic-Referee"
            elif s_lbl != "O" and margin >= 0.70 and p_o < 0.25:
                # SpanNER has decisive boundary/token margin
                final_lbl = s_lbl
                source = "JeBert-SpanNER-Margin-Referee"
            elif j_lbl != "O":
                # Jev detected semantic entity
                final_lbl = j_lbl
                source = "JeBert-Jev-Referee"
            else:
                # Both indicate low confidence / O
                final_lbl = "O"
                source = "JeBert-DualModel-Background"

            c_copy = dict(c)
            c_copy["label"] = final_lbl
            c_copy["final_label"] = final_lbl
            c_copy["source"] = source
            c_copy["escalated_to_llm"] = False
            c_copy["arbitration_mode"] = "fast_dual_model_referee"
            resolved.append(c_copy)
        return resolved

    async def predict_abstract_async(
        self,
        abstract: str,
        candidate_spans: List[Dict[str, Any]],
        http_client: Optional[httpx.AsyncClient] = None,
        guidelines: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Executes asynchronous JeBert inference for high-throughput batch evaluation.
        """
        if not candidate_spans:
            return []

        # Step 1: Jev Independent Semantic Evaluation (Async)
        jev_predictions = await self.jev_evaluator.evaluate_spans_async(
            abstract, candidate_spans, http_client=http_client
        )

        # Step 2: Dual-Model Gating
        if self.enable_gating:
            escalated_cands, consensus_cands = self.gating.filter_candidates(
                candidate_spans, jev_predictions
            )
        else:
            escalated_cands = candidate_spans
            consensus_cands = []

        self.metrics["total_spans"] += len(candidate_spans)
        self.metrics["consensus_accepted"] += len(consensus_cands)
        self.metrics["escalated_to_llm"] += len(escalated_cands)
        for c in escalated_cands:
            if c.get("is_discrepancy", False):
                self.metrics["label_discrepancies"] += 1

        final_entities: List[Dict[str, Any]] = []

        # Add consensus candidates directly (0 LLM cost)
        for c in consensus_cands:
            lbl = canonicalize_label(c.get("jev_label", c.get("spanner_label", "O")), use_new_labels=True)
            c_dict = dict(c)
            c_dict["label"] = lbl
            c_dict["final_label"] = lbl
            c_dict["source"] = "JeBert-Consensus-FastPath"
            c_dict["escalated_to_llm"] = False
            final_entities.append(c_dict)

        # Step 3: Arbitrate Escalated Candidates
        if escalated_cands:
            if self.enable_llm_arbitration and self.arbitrator_mode != "none":
                try:
                    arb_spans, disc_spans = self.arbitrator.arbitrate(
                        abstract=abstract,
                        escalated_spans=escalated_cands,
                        guidelines=guidelines,
                    )
                except Exception as exc:
                    print(f"[WARN] JeBert LLM arbitration fallback: {exc}")
                    arb_spans, disc_spans = self._resolve_with_referee(escalated_cands), []
            else:
                # Fast dual-model referee resolution (0 LLM overhead)
                arb_spans, disc_spans = self._resolve_with_referee(escalated_cands), []

            for s in arb_spans:
                init_lbl = s.get("final_label", s.get("label", "O"))
                v_lbl, triggered, reason = self.gating.apply_margin_safeguard_veto(s, init_lbl)
                if triggered:
                    self.metrics["margin_vetoes_triggered"] += 1
                    s["final_label"] = v_lbl
                    s["label"] = v_lbl
                    s["source"] = "JeBert-Margin-Veto"
                    s["arbitration_rationale"] = reason
                final_entities.append(s)

            for d in disc_spans:
                self.metrics["discovered_entities"] += 1
                final_entities.append(d)

        final_entities.sort(key=lambda x: (x.get("start_char", 0), x.get("end_char", 0)))
        return final_entities

    def print_gating_summary(self) -> None:
        """Prints a human-readable diagnostic report of the gating and arbitration workload."""
        total = max(1, self.metrics["total_spans"])
        consensus = self.metrics["consensus_accepted"]
        escalated = self.metrics["escalated_to_llm"]
        discrepancies = self.metrics["label_discrepancies"]
        vetoes = self.metrics["margin_vetoes_triggered"]
        discovered = self.metrics["discovered_entities"]

        print("=" * 80)
        print("📊 JeBert Dual-Model Gating & Workload Summary:")
        print(f"   • Total Candidate Spans Evaluated: {total}")
        print(f"   • Consensus Auto-Accepted (0 LLM Cost): {consensus} ({consensus / total * 100:.1f}%)")
        print(f"   • Escalated to Heavy LLM:            {escalated} ({escalated / total * 100:.1f}%)")
        print(f"   • Model Label Discrepancies:          {discrepancies} ({discrepancies / total * 100:.1f}%)")
        print(f"   • Margin Safeguard Vetoes Triggered:  {vetoes}")
        if discovered > 0:
            print(f"   • Missed Entities Discovered by LLM:  {discovered}")
        print("=" * 80)
