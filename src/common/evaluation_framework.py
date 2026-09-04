"""
Unified 4-Formulation NER Evaluation Framework for LSF-NER and SpanNER-DinGenPipe.

Provides standardized implementations and report formatters for all 4 NER evaluation formulations:
1. Strict Exact Match (CoNLL Standard)
2. Nested-Aware Exact Match (ACE / Nested NER Benchmark)
3. Partial / Relaxed Boundary Match (MUC-7 / BioCreative Standard)
4. Candidate Mention Match (Ablation / Gating Formulation)
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from src.common.label_mapping import canonicalize_label


def normalize_text(text: str) -> str:
    return text.strip().lower()


def calc_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_doc_4_formulations(
    preds: List[Dict[str, Any]],
    gt_ents: List[Dict[str, Any]],
    use_new_labels: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates predictions on a single abstract across all 4 evaluation formulations.
    """
    active_gts = [
        g for g in gt_ents
        if canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels) not in {"O", "Lifestyle_factor", "lifestyle_factor", "LSF_out_of_context", "Non_LSF"}
    ]
    active_preds = [
        p for p in preds
        if canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels) not in {"O", "Lifestyle_factor", "lifestyle_factor", "LSF_out_of_context", "Non_LSF"}
    ]

    # --------------------------------------------------------------------------
    # 1. Formulation 1: Strict Exact Match (CoNLL Exact Boundary: 1-to-1)
    # --------------------------------------------------------------------------
    matched_gt_strict = set()
    for p in active_preds:
        p_s = p.get("start_char", p.get("start", -1))
        p_e = p.get("end_char", p.get("end", -1))
        p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels).lower()
        p_norm = normalize_text(p.get("span_text", p.get("text", "")))
        
        for g_idx, g in enumerate(active_gts):
            if g_idx in matched_gt_strict:
                continue
            g_s = g.get("start_char", g.get("start", -1))
            g_e = g.get("end_char", g.get("end", -1))
            g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels).lower()
            g_norm = normalize_text(g.get("text", ""))

            exact_pos = (p_s == g_s and p_e == g_e)
            exact_text = (p_norm == g_norm and p_s == g_s)
            if (exact_pos or exact_text) and (p_lbl == g_lbl):
                matched_gt_strict.add(g_idx)
                break
    s_tp = len(matched_gt_strict)
    s_fp = len(active_preds) - s_tp
    s_fn = len(active_gts) - s_tp
    s_p, s_r, s_f1 = calc_prf(s_tp, s_fp, s_fn)

    # --------------------------------------------------------------------------
    # 2. Formulation 2: Nested-Aware Exact Match (ACE / Nested NER Benchmark)
    # --------------------------------------------------------------------------
    # Allows multiple exact matches if ground truth itself has nested annotations
    matched_gt_nested = set()
    matched_pred_nested = set()
    for p_idx, p in enumerate(active_preds):
        p_s = p.get("start_char", p.get("start", -1))
        p_e = p.get("end_char", p.get("end", -1))
        p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels).lower()
        p_norm = normalize_text(p.get("span_text", p.get("text", "")))
        
        for g_idx, g in enumerate(active_gts):
            if g_idx in matched_gt_nested:
                continue
            g_s = g.get("start_char", g.get("start", -1))
            g_e = g.get("end_char", g.get("end", -1))
            g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels).lower()
            g_norm = normalize_text(g.get("text", ""))

            exact_pos = (p_s == g_s and p_e == g_e)
            exact_text = (p_norm == g_norm and (p_s == g_s or abs(p_s - g_s) <= 1))
            if (exact_pos or exact_text) and (p_lbl == g_lbl):
                matched_gt_nested.add(g_idx)
                matched_pred_nested.add(p_idx)
                break
    n_tp = len(matched_gt_nested)
    n_fp = len(active_preds) - len(matched_pred_nested)
    n_fn = len(active_gts) - n_tp
    n_p, n_r, n_f1 = calc_prf(n_tp, n_fp, n_fn)

    # --------------------------------------------------------------------------
    # 3. Formulation 3: Partial / Relaxed Match (MUC-7 / BioCreative: Overlap)
    # --------------------------------------------------------------------------
    matched_gt_partial = set()
    for p in active_preds:
        p_s = p.get("start_char", p.get("start", -1))
        p_e = p.get("end_char", p.get("end", -1))
        p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels).lower()
        p_norm = normalize_text(p.get("span_text", p.get("text", "")))
        
        for g_idx, g in enumerate(active_gts):
            if g_idx in matched_gt_partial:
                continue
            g_s = g.get("start_char", g.get("start", -1))
            g_e = g.get("end_char", g.get("end", -1))
            g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels).lower()
            g_norm = normalize_text(g.get("text", ""))

            overlap = not (p_e <= g_s or p_s >= g_e)
            text_containment = (p_norm in g_norm or g_norm in p_norm) and (overlap or abs(p_s - g_s) <= 5)
            if (overlap or text_containment) and (p_lbl == g_lbl):
                matched_gt_partial.add(g_idx)
                break
    part_tp = len(matched_gt_partial)
    part_fp = len(active_preds) - part_tp
    part_fn = len(active_gts) - part_tp
    part_p, part_r, part_f1 = calc_prf(part_tp, part_fp, part_fn)

    # --------------------------------------------------------------------------
    # 4. Formulation 4: Candidate Mention Match (Ablation Gating Formulation)
    # --------------------------------------------------------------------------
    # Evaluates candidate-level classification without 1-to-1 bipartite lockout.
    # Multiple candidate predictions covering a valid ground truth mention are evaluated
    # on their classification correctness rather than penalized as duplicates.
    cand_tp = 0
    cand_fp = 0
    covered_gt_cand = set()
    for p in active_preds:
        p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels)
        p_norm = normalize_text(p.get("span_text", p.get("text", "")))
        p_s = p.get("start_char", p.get("start", -1))
        p_e = p.get("end_char", p.get("end", -1))

        is_tp = False
        for g_idx, g in enumerate(active_gts):
            g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels)
            if p_lbl != g_lbl:
                continue
            g_s = g.get("start_char", g.get("start", -1))
            g_e = g.get("end_char", g.get("end", -1))
            g_norm = normalize_text(g.get("text", ""))

            exact_pos = (p_s == g_s and p_e == g_e)
            overlap = not (p_e <= g_s or p_s >= g_e)
            text_containment = (p_norm in g_norm or g_norm in p_norm) and (overlap or abs(p_s - g_s) <= 5)
            if exact_pos or overlap or text_containment:
                is_tp = True
                covered_gt_cand.add(g_idx)
                break

        if is_tp:
            cand_tp += 1
        else:
            cand_fp += 1

    cand_fn = len(active_gts) - len(covered_gt_cand)
    cand_p = cand_tp / len(active_preds) if active_preds else (1.0 if not active_gts else 0.0)
    cand_r = len(covered_gt_cand) / len(active_gts) if active_gts else (1.0 if not active_preds else 0.0)
    cand_f1 = (2 * cand_p * cand_r) / (cand_p + cand_r) if (cand_p + cand_r) > 0 else 0.0

    return {
        "strict": {"tp": s_tp, "fp": s_fp, "fn": s_fn, "p": s_p, "r": s_r, "f1": s_f1, "preds": len(active_preds), "gt": len(active_gts)},
        "nested_aware": {"tp": n_tp, "fp": n_fp, "fn": n_fn, "p": n_p, "r": n_r, "f1": n_f1, "preds": len(active_preds), "gt": len(active_gts)},
        "partial": {"tp": part_tp, "fp": part_fp, "fn": part_fn, "p": part_p, "r": part_r, "f1": part_f1, "preds": len(active_preds), "gt": len(active_gts)},
        "mention": {"tp": cand_tp, "fp": cand_fp, "fn": cand_fn, "covered": len(covered_gt_cand), "p": cand_p, "r": cand_r, "f1": cand_f1, "preds": len(active_preds), "gt": len(active_gts)}
    }


def aggregate_4_formulations(doc_metrics_list: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Aggregates per-document 4-formulation metrics across a whole dataset."""
    totals = {
        "strict": defaultdict(int),
        "nested_aware": defaultdict(int),
        "partial": defaultdict(int),
        "mention": defaultdict(int)
    }
    for doc_m in doc_metrics_list:
        for form_key in totals:
            m = doc_m[form_key]
            totals[form_key]["tp"] += m["tp"]
            totals[form_key]["fp"] += m["fp"]
            totals[form_key]["fn"] += m["fn"]
            totals[form_key]["preds"] += m.get("preds", m["tp"] + m["fp"])
            totals[form_key]["gt"] += m.get("gt", m["tp"] + m["fn"])
            if "covered" in m:
                totals[form_key]["covered"] += m["covered"]

    results = {}
    for form_key, t in totals.items():
        if form_key == "mention":
            total_preds = t["preds"]
            total_gt = t["gt"]
            tp = t["tp"]
            fp = t["fp"]
            fn = t["fn"]
            cov = t.get("covered", total_gt - fn)
            p = tp / total_preds if total_preds > 0 else 0.0
            r = cov / total_gt if total_gt > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        else:
            p, r, f1 = calc_prf(t["tp"], t["fp"], t["fn"])
        results[form_key] = {
            "tp": t["tp"], "fp": t["fp"], "fn": t["fn"],
            "p": p, "r": r, "f1": f1,
            "preds": t["preds"], "gt": t["gt"]
        }
    return results


def format_4_formulations_summary_box(
    spanner_results: Dict[str, Dict[str, Any]],
    hybrid_results: Dict[str, Dict[str, Any]] = None,
    active_metrics: List[str] = None,
    system_name: str = "SpanNER + Hybrid LinkNER"
) -> str:
    """
    Returns a clean, beautifully formatted ASCII/Markdown benchmark table comparing
    the chosen evaluation formulations with concise explanations underneath.
    Supports selecting standard 3 formulations (Strict CoNLL, Partial MUC-7, Candidate Mention)
    or all 4.
    """
    lines = []
    lines.append("=" * 118)
    lines.append("           🏆 STANDARDIZED MULTI-TIER NER BENCHMARK REPORT (MULTI-FORMULATION EVALUATION)")
    lines.append("=" * 118)

    all_rows = [
        ("Strict CoNLL (1)", "strict"),
        ("Nested-Aware Match (2)", "nested_aware"),
        ("Partial MUC-7 (3)", "partial"),
        ("Candidate Mention (4)", "mention"),
    ]

    # Filter rows based on active_metrics if specified
    if active_metrics:
        normalized_active = [m.lower().strip() for m in active_metrics]
        rows = [r for r in all_rows if r[1] in normalized_active or r[0].lower().startswith(tuple(normalized_active))]
    else:
        rows = all_rows

    if hybrid_results is not None:
        header = f"{'Evaluation Formulation':<30} | {'SpanNER Baseline (P / R / F1)':<32} | {f'{system_name} (P / R / F1)':<36} | {'Gain / Delta'}"
        lines.append(header)
        lines.append("-" * 118)
        
        for name, key in rows:
            if key not in spanner_results or key not in hybrid_results:
                continue
            s = spanner_results[key]
            h = hybrid_results[key]
            s_str = f"{s['p']*100:5.1f}% / {s['r']*100:5.1f}% / {s['f1']*100:5.1f}%"
            h_str = f"{h['p']*100:5.1f}% / {h['r']*100:5.1f}% / {h['f1']*100:5.1f}%"
            delta = (h['f1'] - s['f1']) * 100
            lines.append(f"{name:<30} | {s_str:<32} | {h_str:<36} | {delta:>+6.2f}% F1")
        lines.append("=" * 118)
    else:
        header = f"{'Evaluation Formulation':<30} | {'Total Preds':<12} | {'TP / FP / FN':<18} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11}"
        lines.append(header)
        lines.append("-" * 118)
        for name, key in rows:
            if key not in spanner_results:
                continue
            s = spanner_results[key]
            tp_fp_fn = f"{s['tp']}/{s['fp']}/{s['fn']}"
            lines.append(f"{name:<30} | {s['preds']:<12} | {tp_fp_fn:<18} | {s['p']*100:6.2f}%     | {s['r']*100:6.2f}%     | {s['f1']*100:6.2f}%")
        lines.append("=" * 118)

    lines.append("\n📖 UNDERSTANDING THE EVALUATION FORMULATIONS:")
    lines.append("  • [Strict CoNLL (1)]:")
    lines.append("    Requires exact character start, exact end, and exact label. Extra sub-spans count as False Positives.")
    if not active_metrics or "nested_aware" in [m.lower() for m in active_metrics]:
        lines.append("  • [Nested-Aware Match (2)]:")
        lines.append("    Exact start, end, and label. Gives TP credit to nested sub-spans only when Ground Truth has nested annotations.")
    lines.append("  • [Partial MUC-7 (3)]:")
    lines.append("    Requires character span overlap and correct label. Recognizes extracted concepts with boundary shifts (1-to-1 matching).")
    lines.append("  • [Candidate Mention (4)]:")
    lines.append("    Evaluates category classification accuracy on candidate mentions without boundary duplication penalties.")
    lines.append("=" * 118)
    
    return "\n".join(lines)


def evaluate_stage_predictions(
    all_stage_preds: List[List[Dict[str, Any]]],
    all_doc_gts: List[List[Dict[str, Any]]],
    use_new_labels: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Evaluates predictions of a single pipeline stage against ground truth across all 3 standard formulations
    (strict, partial, mention) computing micro PRF1, per-abstract macro F1, and per-class macro F1.
    """
    if not all_stage_preds:
        return None

    doc_metrics = [
        evaluate_doc_4_formulations(preds, gts, use_new_labels=use_new_labels)
        for preds, gts in zip(all_stage_preds, all_doc_gts)
    ]
    agg = aggregate_4_formulations(doc_metrics)

    abs_stats = {
        m: {
            "p": [d[m]["p"] for d in doc_metrics],
            "r": [d[m]["r"] for d in doc_metrics],
            "f1": [d[m]["f1"] for d in doc_metrics]
        } for m in ["strict", "partial", "mention"]
    }

    class_counts = defaultdict(lambda: {
        "strict": {"tp": 0, "pred": 0, "gt": 0},
        "partial": {"tp": 0, "pred": 0, "gt": 0},
        "mention": {"tp": 0, "pred": 0, "gt": 0}
    })

    for preds, gts in zip(all_stage_preds, all_doc_gts):
        active_gts = [
            g for g in gts
            if canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels) not in {"O", "Lifestyle_factor", "lifestyle_factor", "LSF_out_of_context", "Non_LSF"}
        ]
        active_preds = [
            p for p in preds
            if canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels) not in {"O", "Lifestyle_factor", "lifestyle_factor", "LSF_out_of_context", "Non_LSF"}
        ]

        for g in active_gts:
            lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels)
            class_counts[lbl]["strict"]["gt"] += 1
            class_counts[lbl]["partial"]["gt"] += 1
            class_counts[lbl]["mention"]["gt"] += 1

        for p in active_preds:
            lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels)
            class_counts[lbl]["strict"]["pred"] += 1
            class_counts[lbl]["partial"]["pred"] += 1
            class_counts[lbl]["mention"]["pred"] += 1

        # Strict matches
        matched_gt_s = set()
        for p in active_preds:
            p_s = p.get("start_char", p.get("start", -1))
            p_e = p.get("end_char", p.get("end", -1))
            p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels)
            for g_idx, g in enumerate(active_gts):
                if g_idx in matched_gt_s:
                    continue
                g_s = g.get("start_char", g.get("start", -1))
                g_e = g.get("end_char", g.get("end", -1))
                g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels)
                if p_s == g_s and p_e == g_e and p_lbl.lower() == g_lbl.lower():
                    matched_gt_s.add(g_idx)
                    class_counts[g_lbl]["strict"]["tp"] += 1
                    break

        # Partial matches
        matched_gt_p = set()
        for p in active_preds:
            p_s = p.get("start_char", p.get("start", -1))
            p_e = p.get("end_char", p.get("end", -1))
            p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels)
            for g_idx, g in enumerate(active_gts):
                if g_idx in matched_gt_p:
                    continue
                g_s = g.get("start_char", g.get("start", -1))
                g_e = g.get("end_char", g.get("end", -1))
                g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels)
                overlap = not (p_e <= g_s or p_s >= g_e)
                if overlap and p_lbl.lower() == g_lbl.lower():
                    matched_gt_p.add(g_idx)
                    class_counts[g_lbl]["partial"]["tp"] += 1
                    break

        # Mention matches
        for p in active_preds:
            p_s = p.get("start_char", p.get("start", -1))
            p_e = p.get("end_char", p.get("end", -1))
            p_lbl = canonicalize_label(p.get("label", p.get("spanner_label", "O")), use_new_labels=use_new_labels)
            for g in active_gts:
                g_s = g.get("start_char", g.get("start", -1))
                g_e = g.get("end_char", g.get("end", -1))
                g_lbl = canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels)
                if not (p_e <= g_s or p_s >= g_e):
                    if p_lbl.lower() == g_lbl.lower():
                        class_counts[g_lbl]["mention"]["tp"] += 1
                        break

    res = {}
    for m in ["strict", "partial", "mention"]:
        m_data = agg[m]
        tp, fp, fn = m_data["tp"], m_data["fp"], m_data["fn"]
        micro_p = m_data["p"]
        micro_r = m_data["r"]
        micro_f1 = m_data["f1"]

        macro_abs_p = sum(abs_stats[m]["p"]) / max(1, len(abs_stats[m]["p"]))
        macro_abs_r = sum(abs_stats[m]["r"]) / max(1, len(abs_stats[m]["r"]))
        macro_abs_f1 = sum(abs_stats[m]["f1"]) / max(1, len(abs_stats[m]["f1"]))

        cls_f1s, cls_ps, cls_rs = [], [], []
        for c, data in class_counts.items():
            ct = data[m]
            cp = ct["tp"] / ct["pred"] if ct["pred"] > 0 else 0.0
            cr = ct["tp"] / ct["gt"] if ct["gt"] > 0 else 0.0
            cf1 = (2 * cp * cr) / (cp + cr) if (cp + cr) > 0 else 0.0
            cls_ps.append(cp)
            cls_rs.append(cr)
            cls_f1s.append(cf1)

        macro_cls_p = sum(cls_ps) / max(1, len(cls_ps))
        macro_cls_r = sum(cls_rs) / max(1, len(cls_rs))
        macro_cls_f1 = sum(cls_f1s) / max(1, len(cls_f1s))

        res[m] = {
            "total_gt": tp + fn,
            "total_pred": tp + fp,
            "tp": tp, "fp": fp, "fn": fn,
            "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
            "macro_abs_p": macro_abs_p, "macro_abs_r": macro_abs_r, "macro_abs_f1": macro_abs_f1,
            "macro_cls_p": macro_cls_p, "macro_cls_r": macro_cls_r, "macro_cls_f1": macro_cls_f1
        }
    return res


def format_stage_table(title: str, res: Dict[str, Any]) -> str:
    """Formats a single stage's 3-formulation evaluation report table."""
    s = res["strict"]
    p = res["partial"]
    m = res["mention"]

    s_tp_str = f"{s['tp']} / {s['total_gt']}"
    p_tp_str = f"{p['tp']} / {p['total_gt']}"
    m_tp_str = f"{m['tp']} / {m['total_gt']}"

    s_mp_str = f"{s['micro_p']*100:.2f}% ({s['micro_p']:.4f})"
    p_mp_str = f"{p['micro_p']*100:.2f}% ({p['micro_p']:.4f})"
    m_mp_str = f"{m['micro_p']*100:.2f}% ({m['micro_p']:.4f})"

    s_mr_str = f"{s['micro_r']*100:.2f}% ({s['micro_r']:.4f})"
    p_mr_str = f"{p['micro_r']*100:.2f}% ({p['micro_r']:.4f})"
    m_mr_str = f"{m['micro_r']*100:.2f}% ({m['micro_r']:.4f})"

    s_mf_str = f"{s['micro_f1']*100:.2f}% ({s['micro_f1']:.4f})"
    p_mf_str = f"{p['micro_f1']*100:.2f}% ({p['micro_f1']:.4f})"
    m_mf_str = f"{m['micro_f1']*100:.2f}% ({m['micro_f1']:.4f})"

    s_af_str = f"{s['macro_abs_f1']*100:.2f}% ({s['macro_abs_f1']:.4f})"
    p_af_str = f"{p['macro_abs_f1']*100:.2f}% ({p['macro_abs_f1']:.4f})"
    m_af_str = f"{m['macro_abs_f1']*100:.2f}% ({m['macro_abs_f1']:.4f})"

    s_cf_str = f"{s['macro_cls_f1']*100:.2f}% ({s['macro_cls_f1']:.4f})"
    p_cf_str = f"{p['macro_cls_f1']*100:.2f}% ({p['macro_cls_f1']:.4f})"
    m_cf_str = f"{m['macro_cls_f1']*100:.2f}% ({m['macro_cls_f1']:.4f})"

    lines = [
        "=" * 105,
        f"  {title.upper()}",
        "=" * 105,
        f"{'Metric':<32} | {'Strict (Exact CoNLL)':<22} | {'Partial (MUC-7)':<22} | {'Candidate Mention':<22}",
        "-" * 105,
        f"{'Total Ground Truth Entities':<32} | {s['total_gt']:<22} | {p['total_gt']:<22} | {m['total_gt']:<22}",
        f"{'Total Predicted Entities':<32} | {s['total_pred']:<22} | {p['total_pred']:<22} | {m['total_pred']:<22}",
        f"{'True Positives (TP)':<32} | {s_tp_str:<22} | {p_tp_str:<22} | {m_tp_str:<22}",
        f"{'False Positives (FP)':<32} | {s['fp']:<22} | {p['fp']:<22} | {m['fp']:<22}",
        f"{'False Negatives (FN)':<32} | {s['fn']:<22} | {p['fn']:<22} | {m['fn']:<22}",
        "-" * 105,
        f"{'Micro Precision':<32} | {s_mp_str:<22} | {p_mp_str:<22} | {m_mp_str:<22}",
        f"{'Micro Recall':<32} | {s_mr_str:<22} | {p_mr_str:<22} | {m_mr_str:<22}",
        f"{'Micro F1-Score':<32} | {s_mf_str:<22} | {p_mf_str:<22} | {m_mf_str:<22}",
        "-" * 105,
        f"{'Macro F1 (Per-Abstract)':<32} | {s_af_str:<22} | {p_af_str:<22} | {m_af_str:<22}",
        f"{'Macro F1 (Per-Class)':<32} | {s_cf_str:<22} | {p_cf_str:<22} | {m_cf_str:<22}",
        "=" * 105
    ]
    return "\n".join(lines)


def format_stage_progression_comparison(stage_results: Dict[str, Optional[Dict[str, Any]]]) -> str:
    """Formats a side-by-side comparative progression table across multiple stages."""
    valid_stages = {k: v for k, v in stage_results.items() if v is not None}
    if not valid_stages:
        return "No valid stage results to compare."

    col_w = max(24, max(len(k) + 2 for k in valid_stages.keys()))
    header_cols = [f"{k:<{col_w}}" for k in valid_stages.keys()]
    total_w = 34 + (col_w + 3) * len(valid_stages)

    def fmt_v(r, mode, key, is_pct=False):
        if not r or mode not in r or key not in r[mode]:
            return "N/A"
        val = r[mode][key]
        return f"{val * 100:.2f}%" if is_pct else str(val)

    def fmt_c(r, mode):
        if not r or mode not in r:
            return "N/A"
        return f"{r[mode]['tp']}/{r[mode]['fp']}/{r[mode]['fn']}"

    lines = [
        "=" * total_w,
        f"{'PIPELINE PROGRESSION & ERROR ANALYSIS COMPARISON':^{total_w}}",
        "=" * total_w,
        f"{'Metric':<34} | " + " | ".join(header_cols),
        "-" * total_w,
        f"{'Total Predicted Entities':<34} | " + " | ".join(f"{fmt_v(r, 'strict', 'total_pred'):<{col_w}}" for r in valid_stages.values()),
        f"{'Strict TP / FP / FN':<34} | " + " | ".join(f"{fmt_c(r, 'strict'):<{col_w}}" for r in valid_stages.values()),
        f"{'Partial TP / FP / FN':<34} | " + " | ".join(f"{fmt_c(r, 'partial'):<{col_w}}" for r in valid_stages.values()),
        f"{'Candidate Mention TP / FP / FN':<34} | " + " | ".join(f"{fmt_c(r, 'mention'):<{col_w}}" for r in valid_stages.values()),
        "-" * total_w,
        # Strict
        f"{'Strict Micro Precision':<34} | " + " | ".join(f"{fmt_v(r, 'strict', 'micro_p', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Strict Micro Recall':<34} | " + " | ".join(f"{fmt_v(r, 'strict', 'micro_r', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Strict Micro F1-Score':<34} | " + " | ".join(f"{fmt_v(r, 'strict', 'micro_f1', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Strict Macro F1 (Per-Abstract)':<34} | " + " | ".join(f"{fmt_v(r, 'strict', 'macro_abs_f1', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Strict Macro F1 (Per-Class)':<34} | " + " | ".join(f"{fmt_v(r, 'strict', 'macro_cls_f1', True):<{col_w}}" for r in valid_stages.values()),
        "-" * total_w,
        # Partial
        f"{'Partial Micro Precision':<34} | " + " | ".join(f"{fmt_v(r, 'partial', 'micro_p', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Partial Micro Recall':<34} | " + " | ".join(f"{fmt_v(r, 'partial', 'micro_r', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Partial Micro F1-Score':<34} | " + " | ".join(f"{fmt_v(r, 'partial', 'micro_f1', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Partial Macro F1 (Per-Abstract)':<34} | " + " | ".join(f"{fmt_v(r, 'partial', 'macro_abs_f1', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Partial Macro F1 (Per-Class)':<34} | " + " | ".join(f"{fmt_v(r, 'partial', 'macro_cls_f1', True):<{col_w}}" for r in valid_stages.values()),
        "-" * total_w,
        # Candidate Mention
        f"{'Candidate Mention Micro Precision':<34} | " + " | ".join(f"{fmt_v(r, 'mention', 'micro_p', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Candidate Mention Micro Recall':<34} | " + " | ".join(f"{fmt_v(r, 'mention', 'micro_r', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Candidate Mention Micro F1':<34} | " + " | ".join(f"{fmt_v(r, 'mention', 'micro_f1', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Candidate Mention Macro F1 (Abs)':<34} | " + " | ".join(f"{fmt_v(r, 'mention', 'macro_abs_f1', True):<{col_w}}" for r in valid_stages.values()),
        f"{'Candidate Mention Macro F1 (Cls)':<34} | " + " | ".join(f"{fmt_v(r, 'mention', 'macro_cls_f1', True):<{col_w}}" for r in valid_stages.values()),
        "=" * total_w
    ]
    return "\n".join(lines)


def run_stage_progression_analysis(
    stages: Dict[str, List[List[Dict[str, Any]]]],
    ground_truths: List[List[Dict[str, Any]]],
    use_new_labels: bool = True,
    print_individual_tables: bool = True
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Executes full multi-tier error analysis across all named pipeline stages, prints individual
    stage tables and the side-by-side comparative table, and returns the structured results dict.
    """
    results = {}
    for stage_name, stage_preds in stages.items():
        if stage_preds:
            r = evaluate_stage_predictions(stage_preds, ground_truths, use_new_labels=use_new_labels)
            results[stage_name] = r
            if print_individual_tables and r:
                print(format_stage_table(stage_name, r))
                print()
        else:
            results[stage_name] = None

    print(format_stage_progression_comparison(results))
    return results


# ------------------------------------------------------------------------------
# RoBERTa Token Classification Baseline (Stage 1 helper)
# ------------------------------------------------------------------------------
BIO21_DEFAULT_LABELS = [
    "O", "I-Socioeconomic_factors", "I-Sleep", "I-Physical_activity", "I-Nutrition",
    "I-Non_physical_leisure_time_activities", "I-Mental_health_practices", "I-Lifestyle_factor",
    "I-Environmental_exposures", "I-Drugs", "I-Beauty_and_Cleaning",
    "B-Socioeconomic_factors", "B-Sleep", "B-Physical_activity", "B-Nutrition",
    "B-Non_physical_leisure_time_activities", "B-Mental_health_practices", "B-Lifestyle_factor",
    "B-Environmental_exposures", "B-Drugs", "B-Beauty_and_Cleaning",
]


def bio_tokens_to_entities(text: str, bio_tags: List[str], offsets: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Decodes BIO tags + token offsets into character-level entity dicts."""
    ents = []
    cur_start, cur_end, cur_label = None, None, None

    def flush():
        nonlocal cur_start, cur_end, cur_label
        if cur_start is not None and cur_label:
            span_text = text[cur_start:cur_end].strip()
            if span_text:
                ents.append({"span_text": span_text, "start_char": cur_start, "end_char": cur_end, "label": cur_label})
        cur_start, cur_end, cur_label = None, None, None

    for tag, (cs, ce) in zip(bio_tags, offsets):
        if cs == ce or tag == "O":
            flush()
            continue
        lab = tag[2:] if (tag.startswith("B-") or tag.startswith("I-")) else tag
        if tag.startswith("B-"):
            flush()
            cur_start, cur_end, cur_label = cs, ce, lab
        elif tag.startswith("I-"):
            if cur_label == lab and cur_end == cs:
                cur_end = ce
            else:
                flush()
                cur_start, cur_end, cur_label = cs, ce, lab
    flush()
    return ents


def run_roberta_token_baseline(
    model_dir: str,
    selected_files: List[Any],
    device: Optional[Any] = None,
    id2label: Optional[Dict[int, str]] = None
) -> List[List[Dict[str, Any]]]:
    """
    Runs the RoBERTa token-classification baseline over the selected files and returns
    per-document entity predictions. Gracefully returns [] when the checkpoint is
    missing or inference fails, so the rest of the evaluation keeps running.
    selected_files items are (txt_path, ann_path, gt_ents) tuples.
    """
    doc_preds: List[List[Dict[str, Any]]] = []

    if not (model_dir and os.path.exists(model_dir)):
        print("⚠️ [Step 1] RoBERTa checkpoint directory not found. Skipping Step 1.")
        return doc_preds
    if not selected_files:
        return doc_preds

    try:
        import torch
        from transformers import RobertaForTokenClassification, AutoTokenizer

        if id2label is None:
            id2label = {i: l for i, l in enumerate(BIO21_DEFAULT_LABELS)}
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        print("⚡ [Step 1] Loading RoBERTa token classification baseline model...")
        tok_model = RobertaForTokenClassification.from_pretrained(model_dir).to(device).eval()
        tok_tokenizer = AutoTokenizer.from_pretrained(model_dir)

        with torch.no_grad():
            for item in selected_files:
                txt_path = item[0]
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                enc = tok_tokenizer(text, return_offsets_mapping=True, return_tensors="pt", max_length=512, truncation=True)
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                offsets = enc["offset_mapping"][0].tolist()
                logits = tok_model(input_ids=ids, attention_mask=mask).logits[0]
                pred_ids = logits.argmax(dim=-1).tolist()
                tags = [id2label[i] for i in pred_ids]
                doc_preds.append(bio_tokens_to_entities(text, tags, offsets))
        print(f"✅ [Step 1] RoBERTa token baseline evaluated on {len(doc_preds)} abstracts.")
    except Exception as e:
        print(f"⚠️ [Step 1] Skipped RoBERTa token model evaluation: {e}")
        doc_preds = []
    return doc_preds


def derive_pipeline_stage_predictions(
    all_doc_preds: List[List[Dict[str, Any]]]
) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Derives the intermediate pipeline stages from hybrid prediction dicts:
    1. SpanNER-only (local neural labels, pre-escalation)
    2. LinkNER single-pass (SpanNER labels for unescalated, LLM labels for escalated)
    3. DinGenPipe final (final labels)
    """
    spanner_only, linkner, dingen = [], [], []
    for preds in all_doc_preds:
        doc_spanner, doc_linkner, doc_dingen = [], [], []
        for p in preds:
            s_label = p.get("spanner_label", p.get("label", "O"))
            l_label = p.get("spanner_label", p.get("label", "O")) if not p.get("escalated_to_llm", False) else p.get("llm_label", p.get("label", "O"))
            f_label = p.get("label", "O")

            if s_label and s_label.lower() != "o":
                doc_spanner.append({"span_text": p["span_text"], "start_char": p["start_char"], "end_char": p["end_char"], "label": s_label})
            if l_label and l_label.lower() != "o":
                doc_linkner.append({"span_text": p["span_text"], "start_char": p["start_char"], "end_char": p["end_char"], "label": l_label})
            if f_label and f_label.lower() != "o":
                doc_dingen.append({"span_text": p["span_text"], "start_char": p["start_char"], "end_char": p["end_char"], "label": f_label})
        spanner_only.append(doc_spanner)
        linkner.append(doc_linkner)
        dingen.append(doc_dingen)
    return spanner_only, linkner, dingen


