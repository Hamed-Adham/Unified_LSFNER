import os
import time
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

from src.common.evaluation_framework import (
    evaluate_doc_4_formulations,
    aggregate_4_formulations,
    format_4_formulations_summary_box
)

def get_timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def to_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "N/A" or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def create_run_directory(
    base_output_dir: str,
    run_name: Optional[str] = None,
    test_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    use_timestamp: bool = True
) -> Tuple[str, str, str]:
    """
    Creates a dedicated run folder.
    Supports hierarchical naming:
      <base_output_dir>/<test_name>/<dataset_name>/<run_name>/
    or standard timestamped runs:
      <base_output_dir>/runs/run_<timestamp>_<run_name>/
    """
    if test_name and dataset_name:
        run_id = run_name or (f"run_{get_timestamp_str()}" if use_timestamp else "run")
        run_dir = os.path.join(base_output_dir, test_name, dataset_name, run_id)
    else:
        ts = get_timestamp_str()
        if not run_name:
            run_id = f"run_{ts}"
        else:
            run_id = f"run_{ts}_{run_name}" if use_timestamp else run_name
        run_dir = os.path.join(base_output_dir, "runs", run_id)

    abstracts_dir = os.path.join(run_dir, "abstracts")
    os.makedirs(abstracts_dir, exist_ok=True)
    return run_dir, abstracts_dir, run_id

def normalize_text(text: str) -> str:
    return text.strip().lower()

def match_gt_label(span_text: str, start_char: int, end_char: int, gt_ents: List[Dict[str, Any]], pred_label: str = None) -> str:
    p_norm = normalize_text(span_text)
    # 1. Exact character boundaries first
    for g in gt_ents:
        if start_char == g.get("start_char", -1) and end_char == g.get("end_char", -1):
            return g.get("label", "O")
    # 2. Nested entity aware match with pred_label
    if pred_label:
        from src.common.label_mapping import canonicalize_label as c_lbl
        p_c = c_lbl(pred_label)
        for g in gt_ents:
            g_c = c_lbl(g.get("label", "O"))
            overlap = not (end_char <= g.get("start_char", 0) or start_char >= g.get("end_char", 0))
            if overlap and p_c == g_c:
                return g.get("label", "O")
    # 3. Position proximity overlap fallback
    for g in gt_ents:
        g_norm = normalize_text(g.get("text", ""))
        overlap = not (end_char <= g.get("start_char", 0) or start_char >= g.get("end_char", 0))
        if overlap and (p_norm in g_norm or g_norm in p_norm):
            return g.get("label", "O")
    return "O"

def canonicalize_label(label: str, use_new_labels: bool = True) -> str:
    from src.common.label_mapping import canonicalize_label as c_lbl
    return c_lbl(label, use_new_labels=use_new_labels)

def classify_10_outcome_str(gt_label: str, spanner_label: str, final_label: str, escalated: bool = True) -> str:
    """
    Classify pipeline span decision into the canonical 10-outcome diagnostic taxonomy:
    1. GT_X_BERT_Y_LLM_X: BERT proposed an incorrect LSF category, but the Generator successfully corrected it to the Ground Truth LSF category.
    2. GT_O_BERT_X_LLM_O: BERT made a mistake and labeled a Non-LSF span as an LSF span, but the Generator correctly dropped it.
    3. GT_X_BERT_X_LLM_X: BERT proposed the correct LSF category, and the Generator preserved it.
    4. GT_X_BERT_X_LLM_Y: BERT proposed the correct LSF category, but the Generator mislabeled it as a different LSF category.
    5. GT_O_BERT_X_LLM_X: Both BERT and the LLM agreed and labeled the span as the same LSF category, but the span is Non-LSF.
    6. GT_O_BERT_X_LLM_Y: Both BERT and the LLM labeled the span as different incorrect LSF categories, whereas the Ground Truth is Non-LSF.
    7. GT_X_BERT_X_LLM_O: BERT proposed the correct LSF category, but the LLM made a mistake and labeled it as Non-LSF.
    8. GT_X_BERT_Y_LLM_O: Both BERT and the LLM made mistakes. BERT proposed an incorrect LSF category, and the LLM labeled the span as Non-LSF.
    9. GT_X_BERT_Y_LLM_Z: Both BERT and the LLM labeled the span as different incorrect LSF categories, and neither matched the Ground Truth LSF category.
    10. GT_X_BERT_Y_LLM_Y: Both BERT and the LLM agreed and labeled the span as the same incorrect LSF categories, and neither matched the Ground Truth LSF category.
    """
    g = canonicalize_label(gt_label)
    s = canonicalize_label(spanner_label)
    f = canonicalize_label(final_label)

    is_g_pos = (g != "O" and g != "Non_LSF" and g != "Non-LSF" and g != "LSF_out_of_context")
    is_s_pos = (s != "O" and s != "Non_LSF" and s != "Non-LSF" and s != "LSF_out_of_context")
    is_f_pos = (f != "O" and f != "Non_LSF" and f != "Non-LSF" and f != "LSF_out_of_context")

    if not escalated:
        if is_s_pos and s == g:
            return "GT_X_BERT_X"
        elif not is_s_pos and not is_g_pos:
            return "GT_O_BERT_O"
        elif is_s_pos and not is_g_pos:
            return "GT_O_BERT_X"
        else:
            return "GT_X_BERT_Y"

    if is_g_pos:
        if is_s_pos:
            if s == g:
                if is_f_pos and f == g:
                    return "GT_X_BERT_X_LLM_X"
                elif is_f_pos and f != g:
                    return "GT_X_BERT_X_LLM_Y"
                else:  # not is_f_pos
                    return "GT_X_BERT_X_LLM_O"
            else:  # s != g
                if is_f_pos and f == g:
                    return "GT_X_BERT_Y_LLM_X"
                elif is_f_pos and f != g:
                    if f == s:
                        return "GT_X_BERT_Y_LLM_Y"
                    else:
                        return "GT_X_BERT_Y_LLM_Z"
                else:  # not is_f_pos
                    return "GT_X_BERT_Y_LLM_O"
        else:  # s was O
            if is_f_pos and f == g:
                return "GT_X_BERT_O_LLM_X"
            elif is_f_pos and f != g:
                return "GT_X_BERT_O_LLM_Y"
            else:
                return "GT_X_BERT_O_LLM_O"
    else:  # Ground Truth is Non-LSF (O)
        if is_s_pos:
            if not is_f_pos:
                return "GT_O_BERT_X_LLM_O"
            else:  # is_f_pos
                if f == s:
                    return "GT_O_BERT_X_LLM_X"
                else:
                    return "GT_O_BERT_X_LLM_Y"
        else:  # s was O
            if not is_f_pos:
                return "GT_O_BERT_O_LLM_O"
            else:
                return "GT_O_BERT_O_LLM_X"

# Backward-compatibility aliases
classify_8_outcome_str = classify_10_outcome_str
classify_outcome = classify_10_outcome_str



def log_abstract_audit(
    doc_id: str,
    text: str,
    gt_ents: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    dinasor_hints: List[Dict[str, Any]],
    dingen_results: Dict[str, Any],
    anchored_abstract: str,
    config: Dict[str, Any],
    abstracts_dir: str,
    dinasor_prompt: str = "",
    dingenerator_prompt: str = "",
    dinasor_raw_response: str = "",
    dingenerator_raw_response: str = ""
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generates and saves a detailed, human-readable .txt audit file for a single abstract.
    Includes the exact, final filled prompts and raw LLM responses passed to and returned by Dinasor and DinGenerator.
    """
    abstract_log_path = os.path.join(abstracts_dir, f"{doc_id}.txt")

    # Auto-extract prompts and raw responses from preds if not passed directly
    if not dinasor_prompt:
        for p in preds:
            if p.get("dinasor_prompt"):
                dinasor_prompt = p["dinasor_prompt"]
                break
    if not dinasor_raw_response:
        for p in preds:
            if p.get("dinasor_raw_response"):
                dinasor_raw_response = p["dinasor_raw_response"]
                break
    if not dingenerator_prompt:
        for p in preds:
            if p.get("dingenerator_prompt"):
                dingenerator_prompt = p["dingenerator_prompt"]
                break
        if not dingenerator_prompt and isinstance(dingen_results, dict):
            dingenerator_prompt = dingen_results.get("prompt", "")
    if not dingenerator_raw_response:
        for p in preds:
            if p.get("dingenerator_raw_response"):
                dingenerator_raw_response = p["dingenerator_raw_response"]
                break
        if not dingenerator_raw_response and isinstance(dingen_results, dict):
            dingenerator_raw_response = dingen_results.get("raw_response", "")
    
    # 1. Compute Ground Truth Spans
    gt_lines = []
    for i, g in enumerate(gt_ents, start=1):
        gt_lines.append(f"  • [T{i}] ({g.get('start_char', 0)}, {g.get('end_char', 0)}): \"{g.get('text', '')}\" -> {canonicalize_label(g.get('label', 'O'))}")

    # 2. Process Predictions and Spans
    span_blocks = []
    audit_rows = []
    spanner_tp = 0; spanner_fp = 0; spanner_fn = 0
    hybrid_tp = 0; hybrid_fp = 0; hybrid_fn = 0

    covered_gts_spanner = set()
    covered_gts_hybrid = set()

    for idx, p in enumerate(preds, start=1):
        span_text = p.get("span_text", "")
        start_c = p.get("start_char", 0)
        end_c = p.get("end_char", 0)
        s_lbl = canonicalize_label(p.get("spanner_label", "O"))
        h_lbl = canonicalize_label(p.get("label", "O"))
        is_esc = p.get("escalated_to_llm", False)
        
        gt_lbl = match_gt_label(span_text, start_c, end_c, gt_ents, pred_label=h_lbl)
        gt_canonical = canonicalize_label(gt_lbl)
        l_lbl = canonicalize_label(p.get("llm_label", h_lbl))

        outcome, outcome_badge, outcome_desc, outcome_expl = classify_audit_outcome(s_lbl, l_lbl, h_lbl, gt_canonical)
        
        # Track metrics
        if s_lbl != "O":
            if s_lbl == gt_canonical:
                spanner_tp += 1
            else:
                spanner_fp += 1
        if h_lbl != "O":
            if h_lbl == gt_canonical:
                hybrid_tp += 1
            else:
                hybrid_fp += 1

        for g_idx, g in enumerate(gt_ents):
            if canonicalize_label(g.get("label", "O")) != "O":
                if s_lbl == canonicalize_label(g.get("label", "O")) and (start_c == g.get("start_char") or normalize_text(span_text) == normalize_text(g.get("text", ""))):
                    covered_gts_spanner.add(g_idx)
                if h_lbl == canonicalize_label(g.get("label", "O")) and (start_c == g.get("start_char") or normalize_text(span_text) == normalize_text(g.get("text", ""))):
                    covered_gts_hybrid.add(g_idx)

        # Span detail block
        esc_str = "⚠️ ESCALATE TO LLM" if is_esc else "⚡ KEEP LOCAL"
        thresh = to_float(config.get("UNRELIABILITY_THRESHOLD", 0.38))
        u_val = to_float(p.get("unreliability", 0.0))
        m_val = to_float(p.get("margin", 0.0))
        po_val = to_float(p.get("p_background_o", p.get("p_o", 0.0)))
        prob_val = to_float(p.get("prob", 0.0))
        top2_prob_val = to_float(p.get("top2_prob", 0.0))
        u_score_val = to_float(p.get("u_score", 0.0))
        nov_val = to_float(p.get("novelty", 0.0))
        ch_type = p.get("challenge_type", "NONE")
        
        blk = f"[Span {idx}]: '{span_text}' (Chars: {start_c}-{end_c})\n"
        blk += f"  • Top-1 Class   : {p.get('top1_label', s_lbl)} (Prob: {prob_val:.4f})\n"
        blk += f"  • Top-2 Class   : {p.get('top2_label', 'N/A')} (Prob: {top2_prob_val:.4f}) | Margin: {m_val:.4f} | P(O): {po_val:.4f}\n"
        blk += f"  • Uncertainty   : {u_score_val:.4f} ({config.get('METHOD', 'MCD').upper()}) | Novelty: {nov_val:.4f} (LOF) | Unreliability: {u_val:.4f}\n"
        blk += f"  • Gating Status : {esc_str} (Unreliability: {u_val:.4f}, Threshold: {thresh:.4f})\n"
        if is_esc:
            blk += f"  • Challenge Type: {ch_type}\n"
            if p.get("contrastive_exemplars"):
                blk += f"  • Contrastive Exemplars: {json.dumps(p.get('contrastive_exemplars'), ensure_ascii=False)}\n"
        span_blocks.append(blk)

        audit_rows.append({
            "span_text": span_text,
            "char_range": f"{start_c}-{end_c}",
            "ground_truth": gt_canonical,
            "spanner_label": s_lbl,
            "hybrid_label": h_lbl,
            "outcome": outcome,
            "outcome_explanation": outcome_expl,
            "escalated": is_esc
        })

    # Count FN
    for g_idx, g in enumerate(gt_ents):
        if canonicalize_label(g.get("label", "O")) != "O":
            if g_idx not in covered_gts_spanner:
                spanner_fn += 1
            if g_idx not in covered_gts_hybrid:
                hybrid_fn += 1

    # Precision, Recall, F1 for this single abstract
    def calc_single_prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 and tp == 0 else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 and tp == 0 else 0.0)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    s_p, s_r, s_f1 = calc_single_prf(spanner_tp, spanner_fp, spanner_fn)
    h_p, h_r, h_f1 = calc_single_prf(hybrid_tp, hybrid_fp, hybrid_fn)

    # 3. Assemble Full Abstract TXT Report
    lines = []
    lines.append("=" * 100)
    lines.append(f"📄 ABSTRACT AUDIT REPORT: [DOC_ID: {doc_id}]")
    lines.append(f"Run Timestamp: {config.get('RUN_TIMESTAMP', get_timestamp_str())} | Dataset: {config.get('DATA_DIR', '')}")
    lines.append(f"Model        : {config.get('MODEL_NAME', 'gemini-cli')} (Provider: {config.get('LLM_PROVIDER', '9router')})")
    lines.append(f"Gating Config: {config.get('GATING_MODE', 'WEIGHTED').upper()} (Unc: {config.get('W_UNCERTAINTY', 0.5)} [{config.get('METHOD', 'mcd').upper()}], Nov: {config.get('W_NOVELTY', 0.5)} [LOF], Thresh: {config.get('UNRELIABILITY_THRESHOLD', 0.38)}, Margin Veto: {config.get('ARBITRATION_MARGIN_THRESHOLD', 0.50)})")
    lines.append("=" * 100)
    lines.append("")

    lines.append("-" * 100)
    lines.append("1. RAW ABSTRACT TEXT & GROUND TRUTH ENTITIES")
    lines.append("-" * 100)
    lines.append("Text:")
    lines.append(f'"{text.strip()}"')
    lines.append("")
    lines.append(f"Ground Truth Entities ({len(gt_ents)} entities):")
    if gt_lines:
        lines.extend(gt_lines)
    else:
        lines.append("  (No entities in ground truth)")
    lines.append("")

    lines.append("-" * 100)
    lines.append(f"2. SPANNER LOCAL NEURAL INFERENCE (Total Candidates: {len(preds)})")
    lines.append("-" * 100)
    if span_blocks:
        lines.append("\n".join(span_blocks))
    else:
        lines.append("  (No candidate spans extracted)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("3. CONTEXT ANCHORING")
    lines.append("-" * 100)
    lines.append("Document-Level Anchored Representation:")
    lines.append(f'"{anchored_abstract.strip()}"')
    lines.append("")

    lines.append("-" * 100)
    lines.append("4. STAGE 4A: DINASOR DIAGNOSTIC REFLECTION")
    lines.append("-" * 100)
    if dinasor_hints or dinasor_prompt or dinasor_raw_response:
        if dinasor_prompt:
            lines.append("--- [DINASOR INPUT PROMPT (FINAL FILLED)] ---")
            lines.append(dinasor_prompt.strip())
            lines.append("----------------------------------------------------------------------------------------------------")
            lines.append("")
        if dinasor_raw_response:
            lines.append("--- [DINASOR RAW LLM RESPONSE] ---")
            lines.append(dinasor_raw_response.strip())
            lines.append("----------------------------------------------------------------------------------------------------")
            lines.append("")
        if dinasor_hints:
            lines.append("Extracted Diagnostic Hints:")
            if isinstance(dinasor_hints, list):
                for idx, h in enumerate(dinasor_hints, start=1):
                    if isinstance(h, dict):
                        lines.append(f"• Item {idx}: \"{h.get('target_entity', '')}\"")
                        lines.append(f"  - Predicted Tag         : {h.get('predicted_tag', '')}")
                        comp_cats = h.get('competing_categories', [])
                        comp_str = ', '.join(comp_cats) if isinstance(comp_cats, list) else str(comp_cats)
                        lines.append(f"  - Competing Categories  : {comp_str}")
                        lines.append(f"  - Dinasor Diagnostic Hint: \"{h.get('resolution_hint', '')}\"")
                    else:
                        lines.append(f"• Item {idx}: \"{str(h)}\"")
            else:
                lines.append(f"• Hints: {str(dinasor_hints)}")
    else:
        lines.append("  (Zero mentions escalated to Dinasor - 100% resolved locally)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("5. STAGE 4B: DINGENERATOR RESOLUTION & MARGIN SAFEGUARD")
    lines.append("-" * 100)
    labeled_ents = dingen_results.get("labeled_entities", []) if isinstance(dingen_results, dict) else []
    if labeled_ents or dingenerator_prompt or dingenerator_raw_response:
        if dingenerator_prompt:
            lines.append("--- [DINGENERATOR INPUT PROMPT (FINAL FILLED)] ---")
            lines.append(dingenerator_prompt.strip())
            lines.append("----------------------------------------------------------------------------------------------------")
            lines.append("")
        if dingenerator_raw_response:
            lines.append("--- [DINGENERATOR RAW LLM RESPONSE] ---")
            lines.append(dingenerator_raw_response.strip())
            lines.append("----------------------------------------------------------------------------------------------------")
            lines.append("")
        if labeled_ents:
            lines.append("Relabeled Entities & Rationale:")
            if isinstance(labeled_ents, list):
                for idx, item in enumerate(labeled_ents, start=1):
                    if isinstance(item, dict):
                        lines.append(f"• \"{item.get('entity', '')}\"")
                        lines.append(f"  - DinGenerator Label   : {item.get('label', '')}")
                        lines.append(f"  - DinGenerator Rationale: \"{item.get('rationale', '')}\"")
                        lines.append(f"  - Final Label           : {item.get('label', '')}")
                    else:
                        lines.append(f"• Item {idx}: {str(item)}")
            else:
                lines.append(f"• Entities: {str(labeled_ents)}")
    else:
        lines.append("  (Zero mentions escalated to DinGenerator)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("6. FINAL AUDIT MATRIX & HUMAN ERROR ANALYSIS")
    lines.append("-" * 100)
    header = f"{'Span Text':<30} | {'Char Range':<12} | {'Ground Truth':<25} | {'SpanNER Label':<25} | {'Final Hybrid':<25} | {'Audit Outcome':<42}"
    lines.append(header)
    lines.append("-" * 175)
    for r in audit_rows:
        lines.append(f"{r['span_text']:<30} | {r['char_range']:<12} | {r['ground_truth']:<25} | {r['spanner_label']:<25} | {r['hybrid_label']:<25} | {r['outcome']:<42}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("7. ABSTRACT-LEVEL PERFORMANCE METRICS")
    lines.append("-" * 100)
    lines.append(f"{'Metric':<25} | {'SpanNER Baseline':<18} | {'SpanNER + DinGenPipe':<20} | {'Gain / Delta':<12}")
    lines.append("-" * 80)
    lines.append(f"{'True Positives (TP)':<25} | {spanner_tp:<18} | {hybrid_tp:<20} | {hybrid_tp - spanner_tp:+d}")
    lines.append(f"{'False Positives (FP)':<25} | {spanner_fp:<18} | {hybrid_fp:<20} | {hybrid_fp - spanner_fp:+d}")
    lines.append(f"{'False Negatives (FN)':<25} | {spanner_fn:<18} | {hybrid_fn:<20} | {hybrid_fn - spanner_fn:+d}")
    lines.append(f"{'Precision':<25} | {s_p*100:>16.2f}% | {h_p*100:>18.2f}% | {(h_p - s_p)*100:>+10.2f}%")
    lines.append(f"{'Recall':<25} | {s_r*100:>16.2f}% | {h_r*100:>18.2f}% | {(h_r - s_r)*100:>+10.2f}%")
    lines.append(f"{'F1-Score':<25} | {s_f1*100:>16.2f}% | {h_f1*100:>18.2f}% | {(h_f1 - s_f1)*100:>+10.2f}%")
    lines.append("=" * 100)

    # Write file
    with open(abstract_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    abstract_metrics = {
        "doc_id": doc_id,
        "spanner": {"tp": spanner_tp, "fp": spanner_fp, "fn": spanner_fn, "p": s_p, "r": s_r, "f1": s_f1},
        "hybrid": {"tp": hybrid_tp, "fp": hybrid_fp, "fn": hybrid_fn, "p": h_p, "r": h_r, "f1": h_f1}
    }

    return abstract_log_path, abstract_metrics, audit_rows


def log_run_summary(
    all_audit_records: List[Dict[str, Any]],
    all_abstract_metrics: List[Dict[str, Any]],
    config: Dict[str, Any],
    execution_time: float,
    run_dir: str
) -> Tuple[str, str]:
    """
    Generates and saves run_summary_report.txt and run_summary_metrics.json.
    """
    summary_txt_path = os.path.join(run_dir, "run_summary_report.txt")
    summary_json_path = os.path.join(run_dir, "run_summary_metrics.json")

    total_abstracts = len(all_abstract_metrics)
    total_spans = len(all_audit_records)
    
    outcomes_counter = Counter(r["outcome"] for r in all_audit_records)
    escalated_count = sum(1 for r in all_audit_records if r.get("escalated", False))
    local_count = total_spans - escalated_count

    # Global Metric Accumulator
    spanner_tp = Counter(); spanner_fp = Counter(); spanner_fn = Counter()
    hybrid_tp = Counter(); hybrid_fp = Counter(); hybrid_fn = Counter()
    classes_set = set()

    for r in all_audit_records:
        gt = r["ground_truth"]
        s = r["spanner_label"]
        h = r["hybrid_label"]

        if s != "O":
            classes_set.add(s)
            if s == gt:
                spanner_tp[s] += 1
            else:
                spanner_fp[s] += 1
        if h != "O":
            classes_set.add(h)
            if h == gt:
                hybrid_tp[h] += 1
            else:
                hybrid_fp[h] += 1

    # Sum total FN across all abstracts
    tot_s_fn = sum(m["spanner"]["fn"] for m in all_abstract_metrics)
    tot_h_fn = sum(m["hybrid"]["fn"] for m in all_abstract_metrics)

    def calc_prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    tot_s_tp = sum(spanner_tp.values()); tot_s_fp = sum(spanner_fp.values())
    tot_h_tp = sum(hybrid_tp.values()); tot_h_fp = sum(hybrid_fp.values())

    s_micro_p, s_micro_r, s_micro_f1 = calc_prf(tot_s_tp, tot_s_fp, tot_s_fn)
    h_micro_p, h_micro_r, h_micro_f1 = calc_prf(tot_h_tp, tot_h_fp, tot_h_fn)

    # Per-Class Breakdown
    class_rows = []
    s_f1_list = []; h_f1_list = []
    for cls in sorted(list(classes_set)):
        c_s_tp = spanner_tp[cls]; c_s_fp = spanner_fp[cls]; c_s_fn = 0
        c_h_tp = hybrid_tp[cls]; c_h_fp = hybrid_fp[cls]; c_h_fn = 0
        # Estimate class-level FN from records
        c_fn = sum(1 for r in all_audit_records if r["ground_truth"] == cls and r["hybrid_label"] != cls)
        c_s_fn_val = sum(1 for r in all_audit_records if r["ground_truth"] == cls and r["spanner_label"] != cls)

        _, _, c_s_f1 = calc_prf(c_s_tp, c_s_fp, c_s_fn_val)
        _, _, c_h_f1 = calc_prf(c_h_tp, c_h_fp, c_fn)
        s_f1_list.append(c_s_f1)
        h_f1_list.append(c_h_f1)
        delta_f = c_h_f1 - c_s_f1
        class_rows.append({
            "class": cls,
            "spanner_f1": c_s_f1,
            "hybrid_f1": c_h_f1,
            "delta": delta_f,
            "tp": c_h_tp, "fp": c_h_fp, "fn": c_fn
        })

    s_macro_f1 = sum(s_f1_list) / len(s_f1_list) if s_f1_list else 0.0
    h_macro_f1 = sum(h_f1_list) / len(h_f1_list) if h_f1_list else 0.0

    # Build TXT Report
    lines = []
    lines.append("=" * 100)
    lines.append("📊 LINKNER EXPERIMENT RUN SUMMARY REPORT")
    lines.append(f"Run Timestamp: {config.get('RUN_TIMESTAMP', get_timestamp_str())} | Total Abstracts Evaluated: {total_abstracts}")
    lines.append("=" * 100)
    lines.append("")

    lines.append("-" * 100)
    lines.append("1. EXPERIMENT & PIPELINE CONFIGURATION (Loaded from Cell 2 / .env)")
    lines.append("-" * 100)
    lines.append(f"  • Random Seed               : {config.get('SEED', 42)}")
    lines.append(f"  • Dataset Directory         : {config.get('DATA_DIR', '')} ({total_abstracts} Abstracts)")
    lines.append(f"  • Label Schema              : {'Relabeled' if config.get('USE_NEW_LABELS', True) else 'Legacy'}")
    lines.append(f"  • Active LLM Provider       : {config.get('LLM_PROVIDER', '9ROUTER').upper()} (Base URL: {config.get('API_Base_URL', 'http://localhost:20128/v1')})")
    lines.append(f"  • Active LLM Model          : {config.get('MODEL_NAME', 'gemini-cli')} (Temp: {config.get('LLM_TEMPERATURE', 0.0)}, Max Tokens: {config.get('LLM_MAX_TOKENS', 3064)})")
    lines.append(f"  • Uncertainty Estimator     : {config.get('METHOD', 'MCD').upper()}")
    lines.append(f"  • Novelty Estimator         : LOF (k=10, alpha=0.5)")
    lines.append(f"  • Vector Embedding Provider : {config.get('EMBEDDING_PROVIDER', '9router')} (Model: {config.get('EMBEDDING_MODEL', 'un/Qwen3-Embedding-8B')})")
    lines.append(f"  • Gating Strategy           : {config.get('GATING_MODE', 'WEIGHTED').upper()} (w_nov={config.get('W_NOVELTY', 0.5)}, w_unc={config.get('W_UNCERTAINTY', 0.5)})")
    lines.append(f"  • Unreliability Threshold   : {config.get('UNRELIABILITY_THRESHOLD', 0.38)}")
    lines.append(f"  • Novelty Threshold         : {config.get('NOVELTY_THRESHOLD', 0.55)}")
    lines.append(f"  • Margin Safeguard Veto     : {config.get('ARBITRATION_MARGIN_THRESHOLD', 0.50)}")
    lines.append(f"  • Batch Inference Size      : {config.get('BATCH_SIZE', 5)} abstracts per tensor batch")
    lines.append(f"  • Output Logs Directory     : {run_dir}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("2. GLOBAL BENCHMARK METRICS (SpanNER Baseline vs. Hybrid Pipeline)")
    lines.append("-" * 100)
    lines.append(f"{'Metric':<25} | {'SpanNER Baseline':<18} | {'SpanNER + DinGenPipe':<20} | {'Delta / Gain':<12}")
    lines.append("-" * 80)
    lines.append(f"{'Micro Precision':<25} | {s_micro_p*100:>16.2f}% | {h_micro_p*100:>18.2f}% | {(h_micro_p - s_micro_p)*100:>+10.2f}%")
    lines.append(f"{'Micro Recall':<25} | {s_micro_r*100:>16.2f}% | {h_micro_r*100:>18.2f}% | {(h_micro_r - s_micro_r)*100:>+10.2f}%")
    lines.append(f"{'Micro F1-Score':<25} | {s_micro_f1*100:>16.2f}% | {h_micro_f1*100:>18.2f}% | {(h_micro_f1 - s_micro_f1)*100:>+10.2f}%")
    lines.append(f"{'Macro F1-Score':<25} | {s_macro_f1*100:>16.2f}% | {h_macro_f1*100:>18.2f}% | {(h_macro_f1 - s_macro_f1)*100:>+10.2f}%")
    lines.append("")

    lines.append("-" * 100)
    lines.append("3. PER-CATEGORY PERFORMANCE BREAKDOWN")
    lines.append("-" * 100)
    lines.append(f"{'Category':<45} | {'SpanNER F1':<12} | {'Hybrid F1':<12} | {'Gain / Loss':<12} | {'TP / FP / FN':<15}")
    lines.append("-" * 105)
    for cr in class_rows:
        lines.append(f"{cr['class']:<45} | {cr['spanner_f1']*100:>10.2f}% | {cr['hybrid_f1']*100:>10.2f}% | {cr['delta']*100:>+10.2f}% | {cr['tp']:>3} / {cr['fp']:>3} / {cr['fn']:>3}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("4. REFLECTIVE AUDIT OUTCOMES BREAKDOWN (Human Error Analysis)")
    lines.append("-" * 100)
    lines.append(f"{'Outcome Classification':<42} | {'Total Spans':<12} | {'% of Extracted':<16} | {'Description'}")
    lines.append("-" * 115)
    for outcome, cnt in outcomes_counter.most_common():
        pct = (cnt / total_spans * 100) if total_spans > 0 else 0.0
        lines.append(f"• {outcome:<40} | {cnt:>10} | {pct:>14.2f}% |")
    lines.append("")

    lines.append("-" * 100)
    lines.append("5. COMPUTATIONAL EFFICIENCY & GATING STATISTICS")
    lines.append("-" * 100)
    local_pct = (local_count / total_spans * 100) if total_spans > 0 else 0.0
    esc_pct = (escalated_count / total_spans * 100) if total_spans > 0 else 0.0
    avg_esc = escalated_count / total_abstracts if total_abstracts > 0 else 0.0
    avg_sec = execution_time / total_abstracts if total_abstracts > 0 else 0.0

    lines.append(f"  • Total Spans Extracted     : {total_spans} spans across {total_abstracts} abstracts")
    lines.append(f"  • Kept Local (SpanNER Only) : {local_count} spans ({local_pct:.2f}% computation saved!)")
    lines.append(f"  • Escalated to Dinasor/DinG : {escalated_count} spans ({esc_pct:.2f}%)")
    lines.append(f"  • Average Escalations / Doc : {avg_esc:.2f} mentions per abstract")
    lines.append(f"  • Total Execution Time      : {execution_time:.2f} seconds (~{avg_sec:.2f}s per abstract)")
    lines.append("=" * 100)

    # Write summary TXT
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Write summary JSON
    summary_dict = {
        "config": config,
        "total_abstracts": total_abstracts,
        "total_spans": total_spans,
        "execution_time_seconds": execution_time,
        "metrics": {
            "micro": {"spanner": {"p": s_micro_p, "r": s_micro_r, "f1": s_micro_f1}, "hybrid": {"p": h_micro_p, "r": h_micro_r, "f1": h_micro_f1}},
            "macro": {"spanner_f1": s_macro_f1, "hybrid_f1": h_macro_f1, "delta": h_macro_f1 - s_macro_f1}
        },
        "per_class": class_rows,
        "outcomes": dict(outcomes_counter)
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)

    return summary_txt_path, summary_json_path


# ==============================================================================
# Generative Escalation Breakdown & Reporting Utilities
# ==============================================================================
"""
Audit and Diagnostic Utilities for LinkNER Generative Escalation Pipeline
Provides fine-grained classification of LLM escalation behaviors:
  1. 🚀 Category Correction     : Fixes SpanNER classification error
  2. 🚀 False Negative Recovery : Discovers entity missed by SpanNER
  3. 🧹 False Positive Filter   : Filters out SpanNER false alarm
  4. ❌ Class Override Error    : Swaps correct SpanNER to wrong class
  5. ❌ Dropped Entity          : Discards true entity into background
  6. 🛡️ Arbitration Shield      : Uncertainty check prevented LLM error
  7. ✅ Confirmed Entity        : Both agreed on correct entity
  8. ⚠️ Confused Category       : Both predicted wrong classes
"""

from collections import Counter
from typing import Dict, List, Tuple, Any

AUDIT_CATEGORIES = [
    {
        "key": cat[0],
        "name": cat[0],
        "description": cat[1],
    }
    for cat in [
        ("GT_X_BERT_Y_LLM_X", "BERT proposed an incorrect LSF category, but the Generator successfully corrected it to the Ground Truth LSF category"),
        ("GT_O_BERT_X_LLM_O", "BERT made a mistake and labeled a Non-LSF span as an LSF span, but the Generator correctly dropped it"),
        ("GT_X_BERT_X_LLM_X", "BERT proposed the correct LSF category, and the Generator preserved it"),
        ("GT_X_BERT_X_LLM_Y", "BERT proposed the correct LSF category, but the Generator mislabeled it as a different LSF category"),
        ("GT_O_BERT_X_LLM_X", "Both BERT and the LLM agreed and labeled the span as the same LSF category, but the span is Non-LSF"),
        ("GT_O_BERT_X_LLM_Y", "Both BERT and the LLM labeled the span as different incorrect LSF categories, whereas the Ground Truth is Non-LSF"),
        ("GT_X_BERT_X_LLM_O", "BERT proposed the correct LSF category, but the LLM made a mistake and labeled it as Non-LSF"),
        ("GT_X_BERT_Y_LLM_O", "Both BERT and the LLM made mistakes. BERT proposed an incorrect LSF category, and the LLM labeled the span as Non-LSF"),
        ("GT_X_BERT_Y_LLM_Z", "Both BERT and the LLM labeled the span as different incorrect LSF categories, and neither matched the Ground Truth LSF category"),
        ("GT_X_BERT_Y_LLM_Y", "Both BERT and the LLM agreed and labeled the span as the same incorrect LSF categories, and neither matched the Ground Truth LSF category"),
    ]
]


def classify_audit_outcome(
    spanner_label: str,
    llm_label: str,
    final_label: str,
    gt_label: str,
) -> Tuple[str, str, str, str]:
    """
    Classify an escalated span decision into one of 10 canonical audit categories,
    and produce a natural-language diagnostic explanation.

    Returns:
        (category_code, category_code, category_description, outcome_explanation)
    """
    cat_code = classify_10_outcome_str(
        gt_label=gt_label,
        spanner_label=spanner_label,
        final_label=final_label,
        escalated=True
    )
    desc = dict(CANONICAL_OUTCOME_CATEGORIES).get(cat_code, "Pipeline diagnostic outcome")
    explanation = generate_outcome_explanation(spanner_label, llm_label, final_label, gt_label, cat_code)
    return (cat_code, cat_code, desc, explanation)


def generate_outcome_explanation(
    spanner_label: str,
    llm_label: str,
    final_label: str,
    gt_label: str,
    category_key: str = None,
) -> str:
    """
    Generates a natural-language diagnostic explanation of how the Generative LLM
    interacted with the Primary SpanNER model prediction and Ground Truth.
    """
    s_raw = spanner_label or "O"
    l_raw = llm_label or "O"
    f_raw = final_label or "O"
    g_raw = gt_label or "O"

    if not category_key:
        category_key, _, _, _ = classify_audit_outcome(s_raw, l_raw, f_raw, g_raw)

    if category_key == "GT_X_BERT_Y_LLM_X":
        return f"BERT proposed incorrect LSF category '{s_raw}', but Generator successfully corrected it to Ground Truth '{g_raw}'."
    elif category_key == "GT_O_BERT_X_LLM_O":
        return f"BERT falsely labeled Non-LSF span as '{s_raw}', but Generator correctly dropped it to 'O'."
    elif category_key == "GT_X_BERT_X_LLM_X":
        return f"BERT proposed correct LSF category '{s_raw}', and Generator preserved it."
    elif category_key == "GT_X_BERT_X_LLM_Y":
        return f"BERT proposed correct LSF category '{s_raw}', but Generator mislabeled it as different LSF category '{l_raw}'."
    elif category_key == "GT_O_BERT_X_LLM_X":
        return f"Both BERT and Generator agreed on '{l_raw}', but Ground Truth is Non-LSF ('O')."
    elif category_key == "GT_O_BERT_X_LLM_Y":
        return f"Both models labeled different incorrect LSF categories (BERT: '{s_raw}', Generator: '{l_raw}'), whereas Ground Truth is Non-LSF."
    elif category_key == "GT_X_BERT_X_LLM_O":
        return f"BERT proposed correct LSF category '{s_raw}', but Generator made a mistake and labeled it as Non-LSF ('O')."
    elif category_key == "GT_X_BERT_Y_LLM_O":
        return f"Both made mistakes: BERT proposed incorrect LSF '{s_raw}', and Generator labeled it as Non-LSF ('O') instead of true class '{g_raw}'."
    elif category_key == "GT_X_BERT_Y_LLM_Z":
        return f"Both models labeled different incorrect LSF categories (BERT: '{s_raw}', Generator: '{l_raw}'), and neither matched Ground Truth '{g_raw}'."
    elif category_key == "GT_X_BERT_Y_LLM_Y":
        return f"Both BERT and Generator agreed on the same incorrect LSF category '{l_raw}', and neither matched Ground Truth '{g_raw}'."
    else:
        return f"BERT: '{s_raw}', LLM: '{l_raw}', Final: '{f_raw}', Ground Truth: '{g_raw}'."


def get_audit_status(category_key: str) -> str:
    """
    Maps fine-grained 10 audit categories to high-level 4 diagnostic statuses:
    - LLM_CORRECTED: GT_X_BERT_Y_LLM_X, GT_O_BERT_X_LLM_O
    - LLM_RUINED: GT_X_BERT_X_LLM_Y, GT_X_BERT_X_LLM_O
    - PRESERVED_CORRECT: GT_X_BERT_X_LLM_X
    - BOTH_FAILED: GT_O_BERT_X_LLM_X, GT_O_BERT_X_LLM_Y, GT_X_BERT_Y_LLM_O, GT_X_BERT_Y_LLM_Z, GT_X_BERT_Y_LLM_Y
    """
    norm_key = str(category_key).upper().strip()
    if norm_key in ("GT_X_BERT_Y_LLM_X", "GT_O_BERT_X_LLM_O", "CATEGORY_CORRECTION", "FALSE_NEGATIVE_RECOVERY", "FALSE_POSITIVE_FILTER", "FN_RECOVERY", "FP_FILTER"):
        return "FIXED"
    elif norm_key in ("GT_X_BERT_X_LLM_Y", "GT_X_BERT_X_LLM_O", "CLASS_OVERRIDE_ERROR", "DROPPED_ENTITY"):
        return "RUINED"
    elif norm_key in ("GT_X_BERT_X_LLM_X", "CONFIRMED_ENTITY", "ARBITRATION_SHIELD"):
        return "CONFIRMED_CORRECT"
    else:
        return "NOT_FIXED"


def compute_audit_breakdown(
    audit_items_or_counts: Any,
    total_escalated: int = None,
) -> Dict[str, Any]:
    """
    Computes aggregated counts, percentages, and metrics for a list of audited escalation items
    or an existing category counts mapping.
    """
    KEY_MAP = {
        "category_correction": "CATEGORY_CORRECTION",
        "fn_recovery": "FALSE_NEGATIVE_RECOVERY",
        "false_negative_recovery": "FALSE_NEGATIVE_RECOVERY",
        "fp_filter": "FALSE_POSITIVE_FILTER",
        "false_positive_filter": "FALSE_POSITIVE_FILTER",
        "class_override_error": "CLASS_OVERRIDE_ERROR",
        "dropped_entity": "DROPPED_ENTITY",
        "arbitration_shield": "ARBITRATION_SHIELD",
        "confirmed_entity": "CONFIRMED_ENTITY",
        "confused_category": "CONFUSED_CATEGORY",
    }

    counts = Counter()

    if isinstance(audit_items_or_counts, (list, tuple)):
        audit_items = audit_items_or_counts
        for item in audit_items:
            key = item.get("category_key")
            if not key:
                key, _, _, _ = classify_audit_outcome(
                    item.get("spanner_label", "O"),
                    item.get("llm_label", "O"),
                    item.get("final_label", "O"),
                    item.get("ground_truth_label", "O"),
                )
            norm_key = KEY_MAP.get(str(key).lower().strip(), str(key).upper().strip())
            counts[norm_key] += 1
        total = len(audit_items)
    elif isinstance(audit_items_or_counts, dict):
        raw_counts = audit_items_or_counts
        for k, v in raw_counts.items():
            norm_key = KEY_MAP.get(str(k).lower().strip(), str(k).upper().strip())
            counts[norm_key] += v
        total = total_escalated if total_escalated is not None else sum(counts.values())
    else:
        total = 0

    category_summary = []
    for cat in AUDIT_CATEGORIES:
        k = cat["key"]
        c = counts[k]
        pct = (c / total * 100) if total > 0 else 0.0
        category_summary.append({
            "key": k,
            "name": cat["name"],
            "description": cat["description"],
            "count": c,
            "percentage": pct,
        })

    fixed_count = (
        counts["CATEGORY_CORRECTION"]
        + counts["FALSE_NEGATIVE_RECOVERY"]
        + counts["FALSE_POSITIVE_FILTER"]
    )
    ruined_count = counts["CLASS_OVERRIDE_ERROR"] + counts["DROPPED_ENTITY"]
    arbitration_shield = counts["ARBITRATION_SHIELD"]
    confirmed_count = counts["CONFIRMED_ENTITY"] + arbitration_shield
    confused_count = counts["CONFUSED_CATEGORY"]

    return {
        "total_escalated": total,
        "counts": dict(counts),
        "categories": category_summary,
        "fixed_count": fixed_count,
        "ruined_count": ruined_count,
        "net_improvement": fixed_count - ruined_count,
        "arbitration_shield_count": arbitration_shield,
        "confirmed_count": confirmed_count,
        "confused_count": confused_count,
    }


def format_audit_summary_box(breakdown_or_items: Any, method: str = "MCD") -> str:
    """
    Formats the top-level AUDIT SUMMARY box into a clean ASCII table.
    """
    if isinstance(breakdown_or_items, dict) and "fixed_count" in breakdown_or_items:
        b = breakdown_or_items
    else:
        b = compute_audit_breakdown(breakdown_or_items)

    tot = b["total_escalated"]
    fixed = b["fixed_count"]
    ruined = b["ruined_count"]
    confused = b["confused_count"]
    confirmed = b["confirmed_count"]
    net = b["net_improvement"]

    fixed_pct = (fixed / tot * 100) if tot > 0 else 0.0
    ruined_pct = (ruined / tot * 100) if tot > 0 else 0.0
    confused_pct = (confused / tot * 100) if tot > 0 else 0.0
    confirmed_pct = (confirmed / tot * 100) if tot > 0 else 0.0

    lines = [
        "=" * 96,
        f"                      AUDIT SUMMARY ({method.upper()})",
        "=" * 96,
        f"Total Spans Escalated to Generative LLM : {tot}",
        f"  🚀 Spans FIXED by Generative LLM       : {fixed} ({fixed_pct:.1f}%)",
        f"  ❌ Spans RUINED by Generative LLM      : {ruined} ({ruined_pct:.1f}%)",
        f"  ⚠️ Spans NOT FIXED (Wrong)             : {confused} ({confused_pct:.1f}%)",
        f"  ✅ Both Correct (Confirmed)            : {confirmed} ({confirmed_pct:.1f}%)",
        f"NET GENERATIVE LLM IMPROVEMENT           : {'+' if net >= 0 else ''}{net} Spans",
    ]
    return "\n".join(lines)


def format_audit_breakdown_table(
    category_counts_or_breakdown: Any,
    total_escalated: int = None,
) -> str:
    """
    Formats the 8-category breakdown into a clean ASCII table matching the academic evaluation style.
    Robustly handles uppercase, lowercase, and shorthand alias keys.
    """
    if isinstance(category_counts_or_breakdown, dict) and "categories" in category_counts_or_breakdown:
        breakdown = category_counts_or_breakdown
        categories = breakdown["categories"]
    else:
        breakdown = compute_audit_breakdown(category_counts_or_breakdown, total_escalated=total_escalated)
        categories = breakdown["categories"]

    lines = []
    lines.append("------------------------------------------------------------------------------------------------")
    lines.append(f"  {'Category Breakdown':<32} | {'Count':<6} | {'Percentage':<10} | Description")
    lines.append("------------------------------------------------------------------------------------------------")
    for cat in categories:
        name = cat["name"]
        count = cat["count"]
        pct = cat["percentage"]
        desc = cat["description"]
        lines.append(f"  {name:<32} | {count:<6} | {pct:>6.1f}%    | {desc}")
    lines.append("================================================================================================")
    return "\n".join(lines)


CANONICAL_OUTCOME_CATEGORIES = [
    ("GT_X_BERT_Y_LLM_X", "BERT proposed an incorrect LSF category, but the Generator successfully corrected it to the Ground Truth LSF category"),
    ("GT_O_BERT_X_LLM_O", "BERT made a mistake and labeled a Non-LSF span as an LSF span, but the Generator correctly dropped it"),
    ("GT_X_BERT_X_LLM_X", "BERT proposed the correct LSF category, and the Generator preserved it"),
    ("GT_X_BERT_X_LLM_Y", "BERT proposed the correct LSF category, but the Generator mislabeled it as a different LSF category"),
    ("GT_O_BERT_X_LLM_X", "Both BERT and the LLM agreed and labeled the span as the same LSF category, but the span is Non-LSF"),
    ("GT_O_BERT_X_LLM_Y", "Both BERT and the LLM labeled the span as different incorrect LSF categories, whereas the Ground Truth is Non-LSF"),
    ("GT_X_BERT_X_LLM_O", "BERT proposed the correct LSF category, but the LLM made a mistake and labeled it as Non-LSF"),
    ("GT_X_BERT_Y_LLM_O", "Both BERT and the LLM made mistakes. BERT proposed an incorrect LSF category, and the LLM labeled the span as Non-LSF"),
    ("GT_X_BERT_Y_LLM_Z", "Both BERT and the LLM labeled the span as different incorrect LSF categories, and neither matched the Ground Truth LSF category"),
    ("GT_X_BERT_Y_LLM_Y", "Both BERT and the LLM agreed and labeled the span as the same incorrect LSF categories, and neither matched the Ground Truth LSF category"),
]

def compute_canonical_audit_breakdown(
    all_doc_preds: List[List[Dict[str, Any]]],
    all_doc_gts: List[List[Dict[str, Any]]],
    use_new_labels: bool = True
) -> Dict[str, Any]:
    """
    Computes canonical 10-outcome diagnostic breakdowns across all candidate spans (escalated and unescalated).
    Works universally across all architectures (Single-Call, Dinasor/DinGenerator, Sweep Engine, etc.).
    """
    all_outcomes = []
    escalated_outcomes = []
    unescalated_outcomes = []

    for doc_idx, (preds, gts) in enumerate(zip(all_doc_preds, all_doc_gts)):
        for p in preds:
            gt_lbl = match_gt_label(p.get("span_text", ""), p.get("start_char", 0), p.get("end_char", 0), gts)
            is_esc = p.get("escalated_to_llm", False)
            s_lbl = p.get("spanner_label", p.get("label", "O"))
            f_lbl = p.get("label", "O")

            outcome_str = classify_10_outcome_str(
                gt_label=gt_lbl,
                spanner_label=s_lbl,
                final_label=f_lbl,
                escalated=is_esc
            )
            all_outcomes.append(outcome_str)
            if is_esc:
                escalated_outcomes.append(outcome_str)
            else:
                unescalated_outcomes.append(outcome_str)

    counts = Counter(all_outcomes)
    total = len(all_outcomes)

    category_summary = []
    for cat_name, desc in CANONICAL_OUTCOME_CATEGORIES:
        c = counts[cat_name]
        pct = (c / total * 100.0) if total > 0 else 0.0
        category_summary.append({
            "category": cat_name,
            "count": c,
            "percentage": pct,
            "description": desc
        })

    return {
        "total_spans": total,
        "escalated_spans": len(escalated_outcomes),
        "unescalated_spans": len(unescalated_outcomes),
        "categories": category_summary,
        "counts": dict(counts)
    }

def format_canonical_audit_table(
    breakdown_or_preds: Any,
    all_doc_gts: Optional[List[List[Dict[str, Any]]]] = None,
    method: str = "MCD",
    use_new_labels: bool = True
) -> str:
    """
    Formats the canonical 10-outcome diagnostic table matching the academic publication taxonomy.
    """
    if isinstance(breakdown_or_preds, dict) and "categories" in breakdown_or_preds:
        b = breakdown_or_preds
    elif all_doc_gts is not None:
        b = compute_canonical_audit_breakdown(breakdown_or_preds, all_doc_gts, use_new_labels=use_new_labels)
    else:
        raise ValueError("Must provide either a precomputed breakdown dict or (all_doc_preds, all_doc_gts).")

    total = b["total_spans"]
    esc = b["escalated_spans"]
    unesc = b["unescalated_spans"]
    esc_pct = (esc / total * 100.0) if total > 0 else 0.0
    unesc_pct = (unesc / total * 100.0) if total > 0 else 0.0

    lines = []
    lines.append("=" * 110)
    lines.append(f"               AUDIT SUMMARY & OUTCOME CLASSIFICATION ({method.upper()})")
    lines.append("=" * 110)
    lines.append(f"Total Span Predictions Evaluated        : {total}")
    lines.append(f"  • Locally Accepted Spans (Unescalated) : {unesc} ({unesc_pct:.1f}%)")
    lines.append(f"  • Escalated to Generative LLM         : {esc} ({esc_pct:.1f}%)")
    lines.append("-" * 110)
    lines.append(f"  {'Outcome Category':<25} | {'Count':<6} | {'Percentage':<10} | Description")
    lines.append("-" * 110)
    for row in b["categories"]:
        c_name = row["category"]
        c_cnt = row["count"]
        c_pct = row["percentage"]
        c_desc = row["description"]
        lines.append(f"  • {c_name:<23} | {c_cnt:<6} | {c_pct:>6.1f}%    | {c_desc}")
    lines.append("=" * 110)
    return "\n".join(lines)


def run_audit_benchmark_report(
    baseline_preds: List[List[Dict[str, Any]]],
    final_preds: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    active_metrics: Optional[List[str]] = None,
    system_name: str = "Hybrid LinkNER",
    method: str = "MCD",
    use_new_labels: bool = True
) -> Dict[str, Any]:
    """
    Executes and prints:
    1. Standardized Multi-Tier 3-Formulation NER Benchmark Report
    2. Canonical 8/9-Outcome Diagnostic Audit Summary Table

    Returns aggregated results and breakdown dictionary.
    """
    spanner_doc_m = [
        evaluate_doc_4_formulations(baseline_preds[i], ground_truths[i], use_new_labels=use_new_labels)
        for i in range(len(ground_truths))
    ]
    final_doc_m = [
        evaluate_doc_4_formulations(final_preds[i], ground_truths[i], use_new_labels=use_new_labels)
        for i in range(len(ground_truths))
    ]

    spanner_results = aggregate_4_formulations(spanner_doc_m)
    final_results = aggregate_4_formulations(final_doc_m)

    # 1. Print Summary Box
    summary_box = format_4_formulations_summary_box(
        spanner_results=spanner_results,
        hybrid_results=final_results,
        active_metrics=active_metrics,
        system_name=system_name
    )
    print(summary_box)

    # 2. Compute and Print Canonical Audit Breakdown
    breakdown = compute_canonical_audit_breakdown(final_preds, ground_truths, use_new_labels=use_new_labels)
    audit_table = format_canonical_audit_table(breakdown, method=method, use_new_labels=use_new_labels)
    print("\n" + audit_table)

    return {
        "spanner_results": spanner_results,
        "final_results": final_results,
        "audit_breakdown": breakdown
    }


def export_pipeline_run_artifacts(
    base_export_dir: str,
    master_reports_dir: str,
    run_folder_name: str,
    experiment_config: Dict[str, Any],
    selected_files: List[Any],
    baseline_preds: List[List[Dict[str, Any]]],
    final_preds: List[List[Dict[str, Any]]],
    all_doc_gts: List[List[Dict[str, Any]]],
    use_new_labels: bool = True,
    schema_name: str = "Relabeled",
    system_name: str = "Hybrid LinkNER",
    prompt_version: str = "v1"
) -> Dict[str, str]:
    """
    Universal exporter for full pipeline runs:
    1. Generates per-abstract detailed TXT audit logs (prompts, raw LLM outputs, span matrices)
    2. Computes master 3-formulation metrics & per-class strict breakdowns
    3. Saves run_summary_metrics.json
    4. Saves run_summary_report.txt
    5. Updates master_reports_dir index (ablation_summary_metrics.json)
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_export_dir, run_folder_name)
    abstracts_export_dir = os.path.join(run_dir, "abstracts")

    os.makedirs(abstracts_export_dir, exist_ok=True)
    os.makedirs(master_reports_dir, exist_ok=True)

    print("=" * 105)
    print(f"  EXPORTING {system_name.upper()} RUN REPORTS & ABSTRACT AUDIT LOGS")
    print("=" * 105)
    print(f"  • Output Directory : {run_dir}")
    print(f"  • Total Abstracts  : {len(selected_files)}")

    # 1. Generate Per-Abstract TXT Audit Files
    all_audit_outcomes = []
    total_spans_logged = 0
    total_escalated_logged = 0

    for doc_idx, file_info in enumerate(selected_files):
        if isinstance(file_info, (tuple, list)):
            txt_path, ann_path, gt_ents = file_info[0], file_info[1], file_info[2]
        else:
            txt_path = file_info.get("txt_path", "")
            gt_ents = file_info.get("ground_truth", all_doc_gts[doc_idx])

        doc_id = os.path.splitext(os.path.basename(txt_path))[0] if txt_path else f"doc_{doc_idx}"
        raw_text = ""
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

        h_preds = final_preds[doc_idx]
        dinasor_hints = []
        dingen_results = {}
        anchored_text = raw_text

        for p in h_preds:
            if p.get("dinasor_hints"):
                dinasor_hints = p["dinasor_hints"]
            if p.get("dingen_results"):
                dingen_results = p["dingen_results"]
            if p.get("anchored_abstract"):
                anchored_text = p["anchored_abstract"]

        # Check if log_abstract_audit can be called with custom hints or basic matrix
        log_abstract_audit(
            doc_id=doc_id,
            text=raw_text,
            gt_ents=gt_ents,
            preds=h_preds,
            dinasor_hints=dinasor_hints,
            dingen_results=dingen_results,
            anchored_abstract=anchored_text,
            config=experiment_config,
            abstracts_dir=abstracts_export_dir
        )

        for p in h_preds:
            total_spans_logged += 1
            if p.get("escalated_to_llm", False):
                total_escalated_logged += 1

    # 2. Master 3-Formulation Metrics
    spanner_m = [
        evaluate_doc_4_formulations(baseline_preds[i], all_doc_gts[i], use_new_labels=use_new_labels)
        for i in range(len(selected_files))
    ]
    hybrid_m = [
        evaluate_doc_4_formulations(final_preds[i], all_doc_gts[i], use_new_labels=use_new_labels)
        for i in range(len(selected_files))
    ]

    spanner_agg = aggregate_4_formulations(spanner_m)
    hybrid_agg = aggregate_4_formulations(hybrid_m)

    # 3. Per-Class Strict Stats
    class_names = sorted(list(set(
        g["label"] for gts in all_doc_gts for g in gts
        if canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels) not in {"O", "Lifestyle_factor", "lifestyle_factor", "LSF_out_of_context", "Non_LSF", "Non-LSF"}
    )))
    per_class_summary = []
    for c in class_names:
        c_canon = canonicalize_label(c, use_new_labels=use_new_labels)
        s_tp = sum(1 for p, gts in zip(baseline_preds, all_doc_gts) for item in p if canonicalize_label(item.get("spanner_label", item.get("label", "O")), use_new_labels=use_new_labels) == c_canon and match_gt_label(item.get("span_text", ""), item.get("start_char", 0), item.get("end_char", 0), gts) == c_canon)
        s_pred = sum(1 for p in baseline_preds for item in p if canonicalize_label(item.get("spanner_label", item.get("label", "O")), use_new_labels=use_new_labels) == c_canon)
        h_tp = sum(1 for p, gts in zip(final_preds, all_doc_gts) for item in p if canonicalize_label(item.get("label", "O"), use_new_labels=use_new_labels) == c_canon and match_gt_label(item.get("span_text", ""), item.get("start_char", 0), item.get("end_char", 0), gts) == c_canon)
        h_pred = sum(1 for p in final_preds for item in p if canonicalize_label(item.get("label", "O"), use_new_labels=use_new_labels) == c_canon)
        gt_cnt = sum(1 for gts in all_doc_gts for g in gts if canonicalize_label(g.get("label", "O"), use_new_labels=use_new_labels) == c_canon)

        s_p = s_tp / max(1, s_pred)
        s_r = s_tp / max(1, gt_cnt)
        s_f1 = (2 * s_p * s_r) / max(1e-9, s_p + s_r)
        h_p = h_tp / max(1, h_pred)
        h_r = h_tp / max(1, gt_cnt)
        h_f1 = (2 * h_p * h_r) / max(1e-9, h_p + h_r)
        per_class_summary.append({
            "class": c_canon,
            "spanner_f1": s_f1,
            "hybrid_f1": h_f1,
            "delta": h_f1 - s_f1,
            "tp": h_tp,
            "fp": h_pred - h_tp,
            "fn": gt_cnt - h_tp
        })

    # 4. Save JSON Metrics
    summary_metrics_json = {
        "config": experiment_config,
        "total_abstracts": len(selected_files),
        "total_spans": total_spans_logged,
        "escalated_spans": total_escalated_logged,
        "local_computation_saved_pct": (total_spans_logged - total_escalated_logged) / max(1, total_spans_logged) * 100.0,
        "metrics_3_formulations": {
            "strict_conll": {"spanner": spanner_agg["strict"], "hybrid": hybrid_agg["strict"]},
            "partial_muc7": {"spanner": spanner_agg["partial"], "hybrid": hybrid_agg["partial"]},
            "candidate_mention": {"spanner": spanner_agg["mention"], "hybrid": hybrid_agg["mention"]}
        },
        "per_class": per_class_summary
    }

    metrics_json_path = os.path.join(run_dir, "run_summary_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics_json, f, indent=4)

    # 5. Save TXT Report
    s_p_str = f"{spanner_agg['strict']['p']*100:.1f}%"
    s_r_str = f"{spanner_agg['strict']['r']*100:.1f}%"
    s_f_str = f"{spanner_agg['strict']['f1']*100:.1f}%"
    h_p_str = f"{hybrid_agg['strict']['p']*100:.1f}%"
    h_r_str = f"{hybrid_agg['strict']['r']*100:.1f}%"
    h_f_str = f"{hybrid_agg['strict']['f1']*100:.1f}%"
    d_f_str = f"{(hybrid_agg['strict']['f1'] - spanner_agg['strict']['f1'])*100:+.2f}% F1"

    sp_p_str = f"{spanner_agg['partial']['p']*100:.1f}%"
    sp_r_str = f"{spanner_agg['partial']['r']*100:.1f}%"
    sp_f_str = f"{spanner_agg['partial']['f1']*100:.1f}%"
    hp_p_str = f"{hybrid_agg['partial']['p']*100:.1f}%"
    hp_r_str = f"{hybrid_agg['partial']['r']*100:.1f}%"
    hp_f_str = f"{hybrid_agg['partial']['f1']*100:.1f}%"
    dp_f_str = f"{(hybrid_agg['partial']['f1'] - spanner_agg['partial']['f1'])*100:+.2f}% F1"

    sm_p_str = f"{spanner_agg['mention']['p']*100:.1f}%"
    sm_r_str = f"{spanner_agg['mention']['r']*100:.1f}%"
    sm_f_str = f"{spanner_agg['mention']['f1']*100:.1f}%"
    hm_p_str = f"{hybrid_agg['mention']['p']*100:.1f}%"
    hm_r_str = f"{hybrid_agg['mention']['r']*100:.1f}%"
    hm_f_str = f"{hybrid_agg['mention']['f1']*100:.1f}%"
    dm_f_str = f"{(hybrid_agg['mention']['f1'] - spanner_agg['mention']['f1'])*100:+.2f}% F1"

    report_lines = [
        "=" * 100,
        f"{system_name.upper()} EXPERIMENT RUN SUMMARY REPORT",
        f"Run Timestamp: {timestamp_str} | Total Abstracts Evaluated: {len(selected_files)}",
        "=" * 100,
        "",
        "----------------------------------------------------------------------------------------------------",
        "1. EXPERIMENT & PIPELINE CONFIGURATION",
        "----------------------------------------------------------------------------------------------------",
        f"  • Architecture              : {system_name} ({prompt_version})",
        f"  • Dataset Directory         : {experiment_config.get('DATA_DIR', '')} ({len(selected_files)} Abstracts)",
        f"  • Label Schema              : {schema_name} (USE_NEW_LABELS = {use_new_labels})",
        f"  • Uncertainty Estimator     : {experiment_config.get('METHOD', 'MCD').upper()}",
        f"  • Novelty Estimator         : LOF (k=10, alpha=0.5)",
        f"  • Gating Strategy           : {experiment_config.get('GATING_MODE', 'WEIGHTED').upper()} (w_nov={experiment_config.get('W_NOVELTY', 0.5):.2f}, w_unc={experiment_config.get('W_UNCERTAINTY', 0.5):.2f})",
        f"  • Unreliability Threshold   : {experiment_config.get('UNRELIABILITY_THRESHOLD', experiment_config.get('THRESHOLD', 0.38)):.2f}",
        f"  • Margin Safeguard Veto     : {experiment_config.get('ARBITRATION_MARGIN_THRESHOLD', experiment_config.get('MARGIN_THRESHOLD', 0.50)):.2f}",
        f"  • Output Directory          : {run_dir}",
        "",
        "----------------------------------------------------------------------------------------------------",
        "2. MULTI-TIER 3-FORMULATION BENCHMARK SUMMARY",
        "----------------------------------------------------------------------------------------------------",
        f"{'Formulation':<25} | {'SpanNER Baseline (P/R/F1)':<28} | {f'{system_name} (P/R/F1)':<30} | Gain / Delta",
        "-" * 105,
        f"{'Strict CoNLL (1)':<25} | {s_p_str:>5} / {s_r_str:>5} / {s_f_str:>5} | {h_p_str:>5} / {h_r_str:>5} / {h_f_str:>5} | {d_f_str:>10}",
        f"{'Partial MUC-7 (3)':<25} | {sp_p_str:>5} / {sp_r_str:>5} / {sp_f_str:>5} | {hp_p_str:>5} / {hp_r_str:>5} / {hp_f_str:>5} | {dp_f_str:>10}",
        f"{'Candidate Mention (4)':<25} | {sm_p_str:>5} / {sm_r_str:>5} / {sm_f_str:>5} | {hm_p_str:>5} / {hm_r_str:>5} / {hm_f_str:>5} | {dm_f_str:>10}",
        "=" * 105
    ]

    if per_class_summary:
        report_lines.extend([
            "",
            "----------------------------------------------------------------------------------------------------",
            "3. PER-CATEGORY PERFORMANCE BREAKDOWN (Strict Matching)",
            "----------------------------------------------------------------------------------------------------",
            f"{'Category':<48} | {'SpanNER F1':<12} | {'Hybrid F1':<12} | {'Gain / Loss':<12} | TP / FP / FN",
            "-" * 105,
        ])
        for cr in per_class_summary:
            s_f1_val = f"{cr['spanner_f1']*100:.2f}%"
            h_f1_val = f"{cr['hybrid_f1']*100:.2f}%"
            d_f1_val = f"{cr['delta']*100:+.2f}%"
            cnt_val = f"{cr['tp']} / {cr['fp']} / {cr['fn']}"
            report_lines.append(f"{cr['class']:<48} | {s_f1_val:>12} | {h_f1_val:>12} | {d_f1_val:>12} | {cnt_val}")

    report_txt_path = os.path.join(run_dir, "run_summary_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 6. Master Summary Update
    master_json_path = os.path.join(master_reports_dir, "ablation_summary_metrics.json")
    master_data = {}
    if os.path.exists(master_json_path):
        try:
            with open(master_json_path, "r", encoding="utf-8") as f:
                master_data = json.load(f)
        except Exception:
            master_data = {}

    master_data[run_folder_name] = summary_metrics_json
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=4)

    print(f"\nSuccessfully exported:")
    print(f"  1. Abstract Reports   : {len(selected_files)} files saved to {abstracts_export_dir}/")
    print(f"  2. Run Metrics JSON   : {metrics_json_path}")
    print(f"  3. Summary Report TXT : {report_txt_path}")
    print(f"  4. Master Metrics JSON: {master_json_path}")
    print("=" * 105)

    return {
        "run_dir": run_dir,
        "abstracts_dir": abstracts_export_dir,
        "metrics_json": metrics_json_path,
        "summary_report": report_txt_path,
        "master_json": master_json_path
    }


