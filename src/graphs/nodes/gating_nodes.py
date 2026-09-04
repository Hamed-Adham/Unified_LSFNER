"""
src/graphs/nodes/gating_nodes.py
Composite Unreliability Gating Node for LangGraph.
Implements PROB_OR, Weighted, Disjunctive (OR), Conjunctive (AND),
and Copula gating operators to route spans to local acceptance vs LLM escalation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableConfig
from src.graphs.state import UnifiedNERState, CandidateSpan


def compute_unreliability_score(
    u_score: float,
    novelty: float,
    mode: str = "prob_or",
    w_novelty: float = 0.30,
    w_uncertainty: float = 0.70
) -> float:
    """
    Combines uncertainty and novelty scores into a single composite unreliability value in [0, 1].
    """
    mode = (mode or "prob_or").lower()
    if mode in ("prob_or", "union"):
        # Probabilistic OR: 1 - (1 - Novelty) * (1 - Uncertainty)
        return float(1.0 - (1.0 - novelty) * (1.0 - u_score))
    elif mode in ("or", "max", "disjunctive"):
        return float(max(u_score, novelty))
    elif mode in ("and", "min", "conjunctive"):
        return float(min(u_score, novelty))
    elif mode in ("copula", "correlated"):
        return float(u_score + novelty - 1.5 * (u_score * novelty))
    else:  # "weighted" / default convex combination
        return float(w_novelty * novelty + w_uncertainty * u_score)


def composite_gating_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    LangGraph Node: Arbitrates each candidate span into either:
    1. Confident Spans: Unreliability < Threshold -> Direct local acceptance (0 API cost).
    2. Escalated Spans: Unreliability >= Threshold -> Flagged for Generative LLM arbitration.
    """
    candidates = state.get("candidate_spans", [])
    if not candidates:
        return {
            "confident_spans": [],
            "escalated_spans": [],
            "gating_summary": {"total": 0, "confident": 0, "escalated": 0}
        }

    cfg = (config or {}).get("configurable", {})
    threshold = cfg.get("threshold", 0.35)
    gating_mode = cfg.get("gating_mode", "prob_or")
    w_novelty = cfg.get("w_novelty", 0.30)
    w_uncertainty = cfg.get("w_uncertainty", 0.70)
    use_margin_safeguard = cfg.get("use_margin_safeguard", True)
    margin_threshold = cfg.get("margin_threshold", 0.50)

    confident: List[CandidateSpan] = []
    escalated: List[CandidateSpan] = []

    for c in candidates:
        c_item = dict(c)
        u_score = c_item.get("u_score", 0.0)
        novelty = c_item.get("novelty", 0.0)
        margin = c_item.get("margin", 0.0)

        unreliability = compute_unreliability_score(
            u_score=u_score,
            novelty=novelty,
            mode=gating_mode,
            w_novelty=w_novelty,
            w_uncertainty=w_uncertainty
        )
        c_item["unreliability"] = unreliability

        # Margin Safeguard: High neural confidence margin overrides escalation
        is_safeguarded = use_margin_safeguard and (margin >= margin_threshold)

        if (unreliability >= threshold) and not is_safeguarded:
            c_item["source"] = "escalated"
            escalated.append(c_item)
        else:
            c_item["source"] = "spanner_accepted"
            c_item["final_label"] = c_item.get("spanner_label", c_item.get("predicted_tag", "O"))
            confident.append(c_item)

    return {
        "confident_spans": confident,
        "escalated_spans": escalated,
        "gating_summary": {
            "total": len(candidates),
            "confident": len(confident),
            "escalated": len(escalated),
            "threshold": threshold,
            "operator": gating_mode,
        }
    }
