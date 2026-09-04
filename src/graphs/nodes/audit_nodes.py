"""
src/graphs/nodes/audit_nodes.py
LangGraph Node for 8-Outcome Escalation Diagnostics & PRF Evaluation.
Merges local and arbitrated predictions, computes exact/partial match metrics,
and records escalation audit categories.
"""

from __future__ import annotations
import os
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.runnables import RunnableConfig

from src.graphs.state import UnifiedNERState, CandidateSpan
from src.common.audit_logger import classify_8_outcome_str, match_gt_label, canonicalize_label


def safe_run_nms(spans: List[CandidateSpan]) -> List[CandidateSpan]:
    """Greedy Longest-Span Non-Maximum Suppression with safe confidence fallbacks."""
    sorted_spans = sorted(
        spans,
        key=lambda s: (
            s.get("end_char", 0) - s.get("start_char", 0),
            float(s.get("confidence", s.get("prob", 1.0)))
        ),
        reverse=True
    )
    kept = []
    for s in sorted_spans:
        overlap = False
        s_start = s.get("start_char", 0)
        s_end = s.get("end_char", 0)
        for k in kept:
            k_start = k.get("start_char", 0)
            k_end = k.get("end_char", 0)
            if not (s_end <= k_start or s_start >= k_end):
                overlap = True
                break
        if not overlap:
            kept.append(s)
    return sorted(kept, key=lambda s: s.get("start_char", 0))


def compute_prf_metrics(
    predictions: List[CandidateSpan],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes exact and partial match Precision, Recall, and F1.
    """
    if not ground_truth and not predictions:
        return {"exact_precision": 1.0, "exact_recall": 1.0, "exact_f1": 1.0, "partial_f1": 1.0}
    if not ground_truth:
        return {"exact_precision": 0.0, "exact_recall": 1.0, "exact_f1": 0.0, "partial_f1": 0.0}
    if not predictions:
        return {"exact_precision": 1.0, "exact_recall": 0.0, "exact_f1": 0.0, "partial_f1": 0.0}

    # Filter out 'O' predictions
    valid_preds = [p for p in predictions if canonicalize_label(p.get("final_label", "O")) not in ("O", "NON_LSF", "NONE")]
    valid_gts = [g for g in ground_truth if canonicalize_label(g.get("label", "O")) not in ("O", "NON_LSF", "NONE")]

    # Exact Matching
    exact_tp = 0
    matched_gt_indices = set()

    for p in valid_preds:
        p_lbl = canonicalize_label(p.get("final_label", ""))
        p_s = p.get("start_char", -1)
        p_e = p.get("end_char", -1)

        for g_idx, g in enumerate(valid_gts):
            if g_idx in matched_gt_indices:
                continue
            g_lbl = canonicalize_label(g.get("label", ""))
            g_s = g.get("start_char", -2)
            g_e = g.get("end_char", -2)

            if p_s == g_s and p_e == g_e and p_lbl == g_lbl:
                exact_tp += 1
                matched_gt_indices.add(g_idx)
                break

    exact_p = exact_tp / max(len(valid_preds), 1)
    exact_r = exact_tp / max(len(valid_gts), 1)
    exact_f1 = (2 * exact_p * exact_r) / max(exact_p + exact_r, 1e-9)

    # Partial / Overlap Matching
    partial_tp = 0
    matched_partial_gt = set()

    for p in valid_preds:
        p_lbl = canonicalize_label(p.get("final_label", ""))
        p_s = p.get("start_char", -1)
        p_e = p.get("end_char", -1)

        for g_idx, g in enumerate(valid_gts):
            if g_idx in matched_partial_gt:
                continue
            g_lbl = canonicalize_label(g.get("label", ""))
            g_s = g.get("start_char", 0)
            g_e = g.get("end_char", 0)

            # Overlap condition and matching label
            if not (p_e <= g_s or p_s >= g_e) and p_lbl == g_lbl:
                partial_tp += 1
                matched_partial_gt.add(g_idx)
                break

    part_p = partial_tp / max(len(valid_preds), 1)
    part_r = partial_tp / max(len(valid_gts), 1)
    part_f1 = (2 * part_p * part_r) / max(part_p + part_r, 1e-9)

    return {
        "exact_precision": round(float(exact_p), 4),
        "exact_recall": round(float(exact_r), 4),
        "exact_f1": round(float(exact_f1), 4),
        "partial_precision": round(float(part_p), 4),
        "partial_recall": round(float(part_r), 4),
        "partial_f1": round(float(part_f1), 4),
        "tp_exact": exact_tp,
        "pred_count": len(valid_preds),
        "gt_count": len(valid_gts),
    }


def audit_eval_node(
    state: UnifiedNERState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    LangGraph Node: Consolidates predictions from confident and arbitrated spans,
    classifies each escalated item into 1 of 8 audit outcome categories,
    and computes precision, recall, and F1 against ground truth.
    """
    cfg = (config or {}).get("configurable", {})
    confident = state.get("confident_spans", [])
    arbitrated = state.get("arbitrated_spans", [])
    ground_truth = state.get("ground_truth", [])

    # Ensure every span has a final_label and backward-compatible fields
    merged: List[CandidateSpan] = []
    for c in confident:
        item = dict(c)
        f_lbl = item.get("final_label", item.get("spanner_label", item.get("predicted_tag", "O")))
        item["final_label"] = f_lbl
        item["label"] = f_lbl
        item["spanner_label"] = item.get("spanner_label", item.get("predicted_tag", "O"))
        item["llm_label"] = item.get("llm_label", item.get("arbitrated_label", "O"))
        item["escalated_to_llm"] = False
        item["uncertainty"] = item.get("uncertainty", item.get("u_score", 0.0))
        item["unreliability"] = item.get("unreliability", 0.0)
        item["novelty"] = item.get("novelty", 0.0)
        merged.append(item)

    for a in arbitrated:
        item = dict(a)
        f_lbl = item.get("final_label", item.get("arbitrated_label", item.get("spanner_label", "O")))
        item["final_label"] = f_lbl
        item["label"] = f_lbl
        item["spanner_label"] = item.get("spanner_label", item.get("predicted_tag", "O"))
        item["llm_label"] = item.get("llm_label", item.get("arbitrated_label", f_lbl))
        item["escalated_to_llm"] = True
        item["uncertainty"] = item.get("uncertainty", item.get("u_score", 0.0))
        item["unreliability"] = item.get("unreliability", 0.0)
        item["novelty"] = item.get("novelty", 0.0)
        merged.append(item)

    # Sort spans by position (and apply NMS only if enabled)
    sorted_merged = sorted(merged, key=lambda x: (x.get("start_char", 0), -(x.get("end_char", 0) - x.get("start_char", 0))))
    apply_nms = cfg.get("apply_nms", os.getenv("APPLY_NMS", "False").strip().lower() in ("true", "1", "yes"))
    final_preds = safe_run_nms(sorted_merged) if (apply_nms and sorted_merged) else (sorted_merged or [])

    audit_records: List[Dict[str, Any]] = []
    outcome_counter = Counter()

    if ground_truth:
        for span in final_preds:
            st = span.get("span_text", "")
            s_char = span.get("start_char", 0)
            e_char = span.get("end_char", 0)
            spanner_lbl = span.get("spanner_label", span.get("predicted_tag", "O"))
            final_lbl = span.get("final_label", "O")
            is_escalated = (span.get("source") in ("escalated", "arbitrated", "dinasor", "hybrid_linkner"))

            gt_lbl = match_gt_label(st, s_char, e_char, ground_truth)
            span["gold_label"] = gt_lbl

            outcome = classify_8_outcome_str(
                gt_label=gt_lbl,
                spanner_label=spanner_lbl,
                final_label=final_lbl,
                escalated=is_escalated
            )
            span["audit_category"] = outcome
            outcome_counter[outcome] += 1

            audit_records.append({
                "span_text": st,
                "start_char": s_char,
                "end_char": e_char,
                "spanner_label": spanner_lbl,
                "final_label": final_lbl,
                "gold_label": gt_lbl,
                "outcome": outcome,
                "escalated": is_escalated,
                "unreliability": span.get("unreliability", 0.0),
            })

    metrics = compute_prf_metrics(final_preds, ground_truth) if ground_truth is not None else {}

    return {
        "final_predictions": final_preds,
        "audit_records": audit_records,
        "audit_breakdown": dict(outcome_counter),
        "metrics": metrics,
    }
