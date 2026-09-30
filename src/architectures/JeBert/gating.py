"""
Dual-Model Gating & Escalation Engine for JeBert Architecture.

Evaluates upstream candidate spans using coupled signals from:
  1. Dense Neural Extraction (SpanNER): uncertainty (u_score), novelty score (LOF), and prediction margin.
  2. Fast Semantic Judgment (TypeSafe Jev): independent categorical classification, confidence, and probability distribution.

Provides:
  - Discrepancy & Dual-Confidence Gating (Default): Fast consensus auto-acceptance (0 LLM cost) vs. targeted escalation.
  - Continuous Composite Unreliability Score: Tunable weighted scalar metric for ROC/Pareto sweeps.
  - Margin Safeguard Veto: Protects high-margin neural spans against erroneous LLM drops to 'O'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.common.label_mapping import canonicalize_label


class JeBertGating:
    """
    Arbitrates candidate spans between SpanNER and Jev to selectively escalate
    only genuinely ambiguous or conflicting entities to a heavy LLM.
    """

    def __init__(
        self,
        gating_mode: str = "discrepancy",
        min_consensus_conf: float = 0.35,
        max_uncertainty_threshold: float = 0.75,
        novelty_threshold: float = 0.55,
        composite_threshold: float = 0.45,
        w_discrepancy: float = 0.45,
        w_spanner: float = 0.30,
        w_jev: float = 0.25,
        enable_margin_safeguard_veto: bool = True,
        margin_threshold: float = 0.50,
        p_o_threshold: float = 0.20,
    ):
        """
        Args:
            gating_mode: 'discrepancy' (default, consensus vs conflict), 'composite' (scalar score), or 'referee'.
            min_consensus_conf: Minimum Jev confidence needed to auto-accept a matching label.
            max_uncertainty_threshold: SpanNER uncertainty threshold that triggers escalation even on agreement.
            novelty_threshold: Out-of-distribution LOF threshold triggering escalation.
            composite_threshold: Scalar unreliability threshold for 'composite' mode.
            w_discrepancy: Weight of label discrepancy in composite mode.
            w_spanner: Weight of SpanNER uncertainty/novelty in composite mode.
            w_jev: Weight of Jev uncertainty (1 - confidence) in composite mode.
            enable_margin_safeguard_veto: Whether to protect high-margin SpanNER spans from LLM drops to 'O'.
            margin_threshold: SpanNER logit margin required for safeguard veto.
            p_o_threshold: Maximum SpanNER background probability allowed for safeguard veto.
        """
        self.gating_mode = gating_mode.lower()
        self.min_consensus_conf = float(min_consensus_conf)
        self.max_uncertainty_threshold = float(max_uncertainty_threshold)
        self.novelty_threshold = float(novelty_threshold)
        self.composite_threshold = float(composite_threshold)
        self.w_discrepancy = float(w_discrepancy)
        self.w_spanner = float(w_spanner)
        self.w_jev = float(w_jev)
        self.enable_margin_safeguard_veto = bool(enable_margin_safeguard_veto)
        self.margin_threshold = float(margin_threshold)
        self.p_o_threshold = float(p_o_threshold)

    def compute_composite_unreliability(
        self,
        is_discrepancy: bool,
        u_score: float,
        novelty: float,
        jev_conf: float,
    ) -> float:
        """
        Computes a continuous scalar unreliability score in [0.0, 1.0].
        """
        disc_val = 1.0 if is_discrepancy else 0.0
        spanner_unrel = max(0.0, min(1.0, float(1.0 - (1.0 - novelty) * (1.0 - u_score))))
        jev_unrel = max(0.0, min(1.0, float(1.0 - jev_conf)))

        score = (
            self.w_discrepancy * disc_val
            + self.w_spanner * spanner_unrel
            + self.w_jev * jev_unrel
        )
        return float(max(0.0, min(1.0, score)))

    def evaluate_candidate(
        self,
        cand: Dict[str, Any],
        jev_pred: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates a single span against SpanNER and Jev signals.
        Returns: (escalate: bool, reason: str, metadata: Dict[str, Any])
        """
        s_lbl = canonicalize_label(
            cand.get("bert_predicted_label", cand.get("predicted_label", cand.get("spanner_label", "O"))),
            use_new_labels=True,
        )
        j_lbl = canonicalize_label(
            jev_pred.get("label", jev_pred.get("final_label", jev_pred.get("choice", "O"))),
            use_new_labels=True,
        )

        u_score = float(cand.get("uncertainty", cand.get("u_score", 0.0)))
        novelty = float(cand.get("novelty_score", cand.get("novelty", 0.0)))
        margin = float(cand.get("margin", 0.0))
        p_o = float(cand.get("p_o", cand.get("p_background_o", 0.0)))

        jev_conf = float(jev_pred.get("confidence", 0.0))
        jev_probs = jev_pred.get("probabilities", {})
        assigned_rules = jev_pred.get("assigned_guidelines", [])

        is_discrepancy = (s_lbl != j_lbl)
        comp_unrel = self.compute_composite_unreliability(is_discrepancy, u_score, novelty, jev_conf)

        escalate = False
        reason = "Consensus accepted"

        if self.gating_mode == "composite":
            escalate = (comp_unrel >= self.composite_threshold)
            reason = f"Composite unreliability ({comp_unrel:.3f} >= {self.composite_threshold:.3f})"
        elif self.gating_mode == "referee":
            # Hierarchical Referee: if discrepancy, check if Jev is overwhelmingly confident
            if is_discrepancy:
                if jev_conf >= 0.85 and u_score >= 0.40:
                    escalate = False
                    reason = f"Jev decisive override ({j_lbl} conf={jev_conf:.2f} vs SpanNER {s_lbl})"
                elif margin >= 0.85 and jev_conf < 0.30:
                    escalate = False
                    reason = f"SpanNER decisive override ({s_lbl} margin={margin:.2f} vs Jev {j_lbl})"
                else:
                    escalate = True
                    reason = f"Label discrepancy without decisive referee (SpanNER={s_lbl} vs Jev={j_lbl})"
            else:
                if u_score >= self.max_uncertainty_threshold and jev_conf < self.min_consensus_conf:
                    escalate = True
                    reason = f"Dual low-confidence consensus (u_score={u_score:.2f}, jev_conf={jev_conf:.2f})"
                else:
                    escalate = False
                    reason = f"High-confidence consensus ({s_lbl})"
        else:  # Default: "discrepancy"
            if is_discrepancy:
                escalate = True
                reason = f"Label discrepancy: SpanNER='{s_lbl}' vs Jev='{j_lbl}'"
            elif novelty >= self.novelty_threshold:
                escalate = True
                reason = f"Out-of-distribution novelty detected ({novelty:.2f} >= {self.novelty_threshold:.2f})"
            elif u_score >= self.max_uncertainty_threshold and jev_conf < self.min_consensus_conf:
                escalate = True
                reason = f"Dual low-confidence consensus (u_score={u_score:.2f} >= {self.max_uncertainty_threshold:.2f}, jev_conf={jev_conf:.2f} < {self.min_consensus_conf:.2f})"
            else:
                escalate = False
                reason = f"Consensus accepted: '{s_lbl}' (u_score={u_score:.2f}, jev_conf={jev_conf:.2f})"

        metadata = {
            "spanner_label": s_lbl,
            "jev_label": j_lbl,
            "is_discrepancy": is_discrepancy,
            "uncertainty": round(u_score, 4),
            "novelty_score": round(novelty, 4),
            "margin": round(margin, 4),
            "p_o": round(p_o, 4),
            "jev_confidence": round(jev_conf, 4),
            "jev_probabilities": jev_probs,
            "assigned_guidelines": assigned_rules,
            "composite_unreliability": round(comp_unrel, 4),
            "escalated_to_llm": escalate,
            "gating_reason": reason,
        }
        return escalate, reason, metadata

    def filter_candidates(
        self,
        candidate_spans: List[Dict[str, Any]],
        jev_predictions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits candidates into:
          1. escalated_cands: ambiguous / conflicting spans to arbitrate via heavy LLM
          2. consensus_cands: high-confidence agreement spans directly accepted (0 LLM cost)
        """
        escalated_cands: List[Dict[str, Any]] = []
        consensus_cands: List[Dict[str, Any]] = []

        # Map Jev predictions by span id or entity text
        jev_map: Dict[Any, Dict[str, Any]] = {}
        for idx, jp in enumerate(jev_predictions):
            jid = jp.get("id", idx)
            jev_map[jid] = jp

        for idx, c in enumerate(candidate_spans):
            cid = c.get("id", c.get("span_id", idx))
            jp = jev_map.get(cid)
            if not jp and idx < len(jev_predictions):
                jp = jev_predictions[idx]
            if not jp:
                jp = {}

            escalate, reason, meta = self.evaluate_candidate(c, jp)
            merged = dict(c)
            merged.update(meta)
            merged["id"] = cid

            if escalate:
                escalated_cands.append(merged)
            else:
                consensus_cands.append(merged)

        return escalated_cands, consensus_cands

    def apply_margin_safeguard_veto(
        self,
        cand: Dict[str, Any],
        arbitrated_label: str,
    ) -> Tuple[str, bool, str]:
        """
        If the heavy LLM drops a candidate to Non-LSF ('O'), but SpanNER had high margin
        (margin >= margin_threshold) and low P(O) (< p_o_threshold), veto the drop
        and preserve SpanNER's predicted category.
        """
        if not self.enable_margin_safeguard_veto:
            return arbitrated_label, False, "Veto disabled"

        s_lbl = cand.get("spanner_label", cand.get("bert_predicted_label", "O"))
        margin = float(cand.get("margin", 0.0))
        p_o = float(cand.get("p_o", cand.get("p_background_o", 0.0)))

        is_llm_drop = (arbitrated_label == "O" and s_lbl != "O")
        is_high_margin = (margin >= self.margin_threshold and p_o < self.p_o_threshold)

        if is_llm_drop and is_high_margin:
            veto_reason = (
                f"VETO: SpanNER high-confidence margin={margin:.2f} >= {self.margin_threshold:.2f} "
                f"with P(O)={p_o:.2f} < {self.p_o_threshold:.2f}. Restored '{s_lbl}' over LLM drop to 'O'."
            )
            return s_lbl, True, veto_reason

        return arbitrated_label, False, "No veto applied"
