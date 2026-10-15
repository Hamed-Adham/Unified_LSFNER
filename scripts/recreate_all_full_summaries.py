"""
Recreate All Full Summaries with Multi-Formulation Benchmarks
============================================================

Implements the Cell 11 Phase 5 export format across all existing reports folders:
1. Candidate Arbitration Benchmark (All Proposed Mentions: Micro-F1, Macro-F1)
2. Global End-to-End Dataset Benchmark (All Gold Entities) with:
   - Strict CoNLL (1)
   - Partial MUC-7 (3)
   - Candidate Mention (4)
3. Escalated Subpopulation Benchmark (Local Lift on Ambiguous Spans)
4. Arbitration Balance Sheet & Win Rate (Correction vs Degradation Dynamics)
5. Canonical 10-Outcome Diagnostic Breakdown
6. Per-Class Performance Breakdown
"""

import os
import re
import json
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.label_mapping import canonicalize_label, CANONICAL_10_LABELS
from src.common.audit_logger import classify_10_outcome_str, format_canonical_audit_table
from src.common.evaluation_framework import (
    evaluate_doc_4_formulations,
    aggregate_4_formulations,
    calc_prf
)

# 10 Canonical Triad Outcome Descriptions + Unescalated Classes
TRIAD_DESCRIPTIONS = {
    "GT_X_BERT_Y_LLM_X": "BERT proposed an incorrect LSF category, but the Generator successfully corrected it to the Ground Truth LSF category.",
    "GT_O_BERT_X_LLM_O": "BERT made a mistake and labeled a Non-LSF span as an LSF span, but the Generator correctly dropped it.",
    "GT_X_BERT_X_LLM_X": "BERT proposed the correct LSF category, and the Generator preserved it.",
    "GT_X_BERT_X_LLM_Y": "BERT proposed the correct LSF category, but the Generator mislabeled it as a different LSF category.",
    "GT_O_BERT_X_LLM_X": "Both BERT and the LLM agreed and labeled the span as the same LSF category, but the span is Non-LSF.",
    "GT_O_BERT_X_LLM_Y": "Both BERT and the LLM labeled the span as different incorrect LSF categories, whereas the Ground Truth is Non-LSF.",
    "GT_X_BERT_X_LLM_O": "BERT proposed the correct LSF category, but the LLM made a mistake and labeled it as Non-LSF.",
    "GT_X_BERT_Y_LLM_O": "Both BERT and the LLM made mistakes. BERT proposed an incorrect LSF category, and the LLM labeled the span as Non-LSF.",
    "GT_X_BERT_Y_LLM_Z": "Both BERT and the LLM labeled the span as different incorrect LSF categories, and neither matched the Ground Truth LSF category.",
    "GT_X_BERT_Y_LLM_Y": "BERT proposed an incorrect LSF category, and the Generator uncritically adopted it instead of correcting to Ground Truth.",
    "GT_X_BERT_X": "Correct BERT prediction accepted directly by gating filter (no LLM call).",
    "GT_O_BERT_X": "Spurious BERT candidate accepted directly by gating filter (no LLM call).",
    "GT_X_BERT_Y": "Incorrect BERT prediction accepted directly by gating filter (no LLM call).",
    "GT_O_BERT_O": "Non-LSF span correctly left unproposed (no LLM call)."
}

REPORT_FOLDERS = [
    {
        "name": "Weighted Gated ACE Generator v3",
        "dir": PROJECT_ROOT / "reports" / "ace_generator_v3",
        "system_name": "SpanNER + Weighted Gated ACE Gen v3",
    },
    {
        "name": "JEV-Gated ACE Generator v3",
        "dir": PROJECT_ROOT / "reports" / "jev_gated_ace_v3",
        "system_name": "SpanNER + JEV-Gated ACE Gen v3",
    },
    {
        "name": "JEV-Gated Dinasor-DinGenerator (2-Step)",
        "dir": PROJECT_ROOT / "reports" / "jev_gated_dinasor_dingen",
        "system_name": "SpanNER + JEV-Gated Dinasor-DinGen",
    },
    {
        "name": "JEV Uncertainty-Gated ACE Generator",
        "dir": PROJECT_ROOT / "reports" / "jev_ace_gen_unc",
        "system_name": "SpanNER + JEV Uncertainty-Gated ACE Gen",
    }
]



def load_gold_entities_map() -> Dict[str, List[Dict[str, Any]]]:
    """Finds and parses .ann files for 400Abstracts dataset (prioritizing no_disease split)."""
    gold_map = {}
    
    # Priority paths for validation and training ground truth
    search_dirs = [
        PROJECT_ROOT / "data" / "new" / "400Abstracts" / "no_disease" / "val",
        PROJECT_ROOT / "data" / "new" / "400Abstracts" / "no_disease" / "train",
        PROJECT_ROOT / "data" / "new" / "400Abstracts" / "val",
        PROJECT_ROOT / "data" / "new" / "400Abstracts" / "train",
        PROJECT_ROOT / "data" / "new" / "400Abstracts" / "no_disease" / "balanced",
    ]
    
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for af in s_dir.glob("*.ann"):
            doc_id = af.stem
            if doc_id in gold_map:
                continue
            ents = []
            with open(af, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("T"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        type_offsets = parts[1]
                        span_text = parts[2]
                        type_parts = type_offsets.split()
                        lbl = type_parts[0]
                        c_lbl = canonicalize_label(lbl)
                        # Filter out non-LSF and Disease labels
                        if c_lbl == "O" or c_lbl == "Disease":
                            continue
                        raw_offsets = type_offsets[len(lbl):].strip().replace(";", " ").split()
                        if len(raw_offsets) >= 2:
                            st_c = int(raw_offsets[0])
                            end_c = int(raw_offsets[-1])
                            ents.append({
                                "label": c_lbl,
                                "start_char": st_c,
                                "end_char": end_c,
                                "text": span_text
                            })
            if ents:
                gold_map[doc_id] = ents

    return gold_map


def parse_doc_report_txt(report_path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Parses a per-document report text file to extract span-level decisions and offsets."""
    content = report_path.read_text(encoding="utf-8")
    doc_id = report_path.stem
    
    # Extract abstract text
    abstract_text = ""
    abs_match = re.search(r"DOCUMENT ABSTRACT TEXT:\s*-+\s*(.*?)\s*={20,}", content, re.DOTALL)
    if abs_match:
        abstract_text = abs_match.group(1).strip()

    # Extract spans
    spans = []
    span_blocks = re.findall(r'\[SPAN \d+/\d+\] "(.*?)" \(Offsets: \[(\d+):(\d+)\]\)(.*?)(?=(?:\[SPAN \d+/\d+\]|={20,}|$))', content, re.DOTALL)
    for span_text, s_start, s_end, block in span_blocks:
        gt_match = re.search(r"Ground Truth \(GT\)\s*:\s*([^\s\(]+)", block)
        bert_match = re.search(r"BERT Prior\s*:\s*([^\s\[]+)", block)
        gating_match = re.search(r"Gating Decision\s*:\s*([^\n]+)", block)
        final_match = re.search(r"Final Assigned Label\s*:\s*([^\n]+)", block)

        gt_lbl = canonicalize_label(gt_match.group(1).strip()) if gt_match else "O"
        bert_lbl = canonicalize_label(bert_match.group(1).strip()) if bert_match else "O"
        final_lbl = canonicalize_label(final_match.group(1).strip()) if final_match else bert_lbl

        escalated = "ESCALATED" in (gating_match.group(1) if gating_match else "")

        spans.append({
            "span_text": span_text,
            "start_char": int(s_start),
            "end_char": int(s_end),
            "gt_label": gt_lbl,
            "bert_label": bert_lbl,
            "spanner_label": bert_lbl,
            "label": final_lbl,
            "final_label": final_lbl,
            "escalated_to_llm": escalated,
            "escalated": escalated
        })

    return doc_id, abstract_text, spans


def is_pos(lbl: str) -> bool:
    c = canonicalize_label(lbl)
    return c not in ("O", "Non-LSF", "Non_LSF", "LSF_out_of_context", "None", "")


def process_report_folder(info: Dict[str, Any], gold_map: Dict[str, List[Dict[str, Any]]]):
    report_dir = info["dir"]
    if not report_dir.exists():
        print(f"Skipping {report_dir} (not found)")
        return

    txt_files = sorted([f for f in report_dir.glob("*.txt") if f.stem.isdigit()])
    if not txt_files:
        print(f"No document reports found in {report_dir}")
        return

    print(f"\nProcessing {len(txt_files)} document reports in {report_dir.name}...")

    all_test_preds = []
    all_spanner_preds = []
    all_test_gts = []
    doc_spanner_metrics_list = []
    doc_hybrid_metrics_list = []

    tot_esc_spans = 0
    tot_unesc_spans = 0
    esc_s_tp = esc_s_fp = esc_s_fn = 0
    esc_h_tp = esc_h_fp = esc_h_fn = 0

    pos_corr_cat = 0   # GT_X_BERT_Y_LLM_X
    pos_corr_drop = 0  # GT_O_BERT_X_LLM_O
    harm_corrupt = 0   # GT_X_BERT_X_LLM_Y
    harm_drop = 0      # GT_X_BERT_X_LLM_O
    pres_correct = 0   # GT_X_BERT_X_LLM_X

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    all_gt_labels = []
    all_bert_labels = []
    all_final_labels = []
    all_triads = Counter()

    for rf in txt_files:
        doc_id, abstract_text, spans = parse_doc_report_txt(rf)
        if not spans:
            continue

        gt_ents = gold_map.get(doc_id, [])
        if not gt_ents:
            # Fallback to ground truth in cached JSON if available
            cached_json = PROJECT_ROOT / "data" / "cached" / "spanner_160_10_class" / "400" / "val" / f"{doc_id}.json"
            if cached_json.exists():
                with open(cached_json, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                gt_ents = [
                    {
                        "label": canonicalize_label(s.get("ground_truth", {}).get("gt_label", "O")),
                        "start_char": s.get("boundaries", {}).get("start_char", 0),
                        "end_char": s.get("boundaries", {}).get("end_char", len(s.get("span_text", ""))),
                        "text": s.get("span_text", "")
                    }
                    for s in c_data.get("spans", [])
                    if s.get("ground_truth", {}).get("is_entity", False) and canonicalize_label(s.get("ground_truth", {}).get("gt_label", "O")) != "O"
                ]

        all_test_gts.append(gt_ents)

        # Build prediction lists for multi-formulation evaluator
        doc_hybrid_preds = []
        doc_spanner_preds = []

        for p in spans:
            gt_lbl = canonicalize_label(p["gt_label"])
            s_lbl = canonicalize_label(p["bert_label"])
            f_lbl = canonicalize_label(p["final_label"])
            is_esc = p["escalated"]

            all_gt_labels.append(gt_lbl)
            all_bert_labels.append(s_lbl)
            all_final_labels.append(f_lbl)

            # Classify Triad outcome
            triad = classify_10_outcome_str(gt_lbl, s_lbl, f_lbl, escalated=is_esc)
            all_triads[triad] += 1

            # Per-class metrics accumulation
            gt_p = is_pos(gt_lbl)
            f_p = is_pos(f_lbl)
            s_p = is_pos(s_lbl)

            if gt_p and f_p:
                if gt_lbl == f_lbl:
                    per_class_tp[gt_lbl] += 1
                else:
                    per_class_fp[f_lbl] += 1
                    per_class_fn[gt_lbl] += 1
            elif not gt_p and f_p:
                per_class_fp[f_lbl] += 1
            elif gt_p and not f_p:
                per_class_fn[gt_lbl] += 1

            # Escalated subpopulation tracking
            if is_esc:
                tot_esc_spans += 1
                # SpanNER candidate stats
                if s_p:
                    if gt_p and s_lbl == gt_lbl: esc_s_tp += 1
                    else: esc_s_fp += 1
                if gt_p and (not s_p or s_lbl != gt_lbl):
                    esc_s_fn += 1

                # Hybrid model candidate stats
                if f_p:
                    if gt_p and f_lbl == gt_lbl: esc_h_tp += 1
                    else: esc_h_fp += 1
                if gt_p and (not f_p or f_lbl != gt_lbl):
                    esc_h_fn += 1

                # Arbitration dynamics breakdown
                if gt_p:
                    if s_p and s_lbl == gt_lbl:
                        if f_p and f_lbl == gt_lbl: pres_correct += 1
                        elif f_p and f_lbl != gt_lbl: harm_corrupt += 1
                        else: harm_drop += 1
                    elif s_p and s_lbl != gt_lbl:
                        if f_p and f_lbl == gt_lbl: pos_corr_cat += 1
                else:
                    if s_p and not f_p:
                        pos_corr_drop += 1
            else:
                tot_unesc_spans += 1

            doc_hybrid_preds.append({
                "span_text": p["span_text"],
                "start_char": p["start_char"],
                "end_char": p["end_char"],
                "label": f_lbl,
                "spanner_label": s_lbl,
                "escalated_to_llm": is_esc
            })

            doc_spanner_preds.append({
                "span_text": p["span_text"],
                "start_char": p["start_char"],
                "end_char": p["end_char"],
                "label": s_lbl,
                "spanner_label": s_lbl,
                "escalated_to_llm": False
            })

        all_test_preds.append(doc_hybrid_preds)
        all_spanner_preds.append(doc_spanner_preds)

        # Compute 4-formulation metrics for this document
        doc_spanner_metrics_list.append(evaluate_doc_4_formulations(doc_spanner_preds, gt_ents, use_new_labels=True))
        doc_hybrid_metrics_list.append(evaluate_doc_4_formulations(doc_hybrid_preds, gt_ents, use_new_labels=True))

    # --------------------------------------------------------------------------
    # Aggregate Global Multi-Formulation Metrics
    # --------------------------------------------------------------------------
    spanner_results = aggregate_4_formulations(doc_spanner_metrics_list)
    hybrid_results = aggregate_4_formulations(doc_hybrid_metrics_list)

    # --------------------------------------------------------------------------
    # Candidate Arbitration Benchmark (Micro / Macro)
    # --------------------------------------------------------------------------
    def calc_micro_macro(gt_l, pred_l):
        tp = fp = fn = 0
        c_tp = defaultdict(int)
        c_fp = defaultdict(int)
        c_fn = defaultdict(int)
        for g, p in zip(gt_l, pred_l):
            gp = is_pos(g)
            pp = is_pos(p)
            if gp and pp:
                if g == p:
                    tp += 1
                    c_tp[g] += 1
                else:
                    fp += 1
                    fn += 1
                    c_fp[p] += 1
                    c_fn[g] += 1
            elif not gp and pp:
                fp += 1
                c_fp[p] += 1
            elif gp and not pp:
                fn += 1
                c_fn[g] += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        all_cats = [c for c in CANONICAL_10_LABELS if c != "O"]
        cat_f1s = []
        for cat in all_cats:
            cat_p = c_tp[cat] / (c_tp[cat] + c_fp[cat]) if (c_tp[cat] + c_fp[cat]) > 0 else 0.0
            cat_r = c_tp[cat] / (c_tp[cat] + c_fn[cat]) if (c_tp[cat] + c_fn[cat]) > 0 else 0.0
            cat_f = (2 * cat_p * cat_r) / (cat_p + cat_r) if (cat_p + cat_r) > 0 else 0.0
            cat_f1s.append(cat_f)
        macro_f1 = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0.0
        return prec, rec, f1, macro_f1

    s_mic_p, s_mic_r, s_mic_f1, s_mac_f1 = calc_micro_macro(all_gt_labels, all_bert_labels)
    h_mic_p, h_mic_r, h_mic_f1, h_mac_f1 = calc_micro_macro(all_gt_labels, all_final_labels)

    # --------------------------------------------------------------------------
    # Escalated Subpopulation Benchmark
    # --------------------------------------------------------------------------
    tot_spans = tot_esc_spans + tot_unesc_spans
    esc_rate = (tot_esc_spans / max(1, tot_spans)) * 100.0

    esc_s_prec = esc_s_tp / max(1, esc_s_tp + esc_s_fp)
    esc_s_rec  = esc_s_tp / max(1, esc_s_tp + esc_s_fn)
    esc_s_f1   = (2 * esc_s_prec * esc_s_rec) / max(1e-9, esc_s_prec + esc_s_rec)

    esc_h_prec = esc_h_tp / max(1, esc_h_tp + esc_h_fp)
    esc_h_rec  = esc_h_tp / max(1, esc_h_tp + esc_h_fn)
    esc_h_f1   = (2 * esc_h_prec * esc_h_rec) / max(1e-9, esc_h_prec + esc_h_rec)

    tot_pos_corr = pos_corr_cat + pos_corr_drop
    tot_harm_inv = harm_corrupt + harm_drop
    net_corrections = tot_pos_corr - tot_harm_inv
    win_rate = (tot_pos_corr / max(1, tot_pos_corr + tot_harm_inv)) * 100.0
    acc_swing = (net_corrections / max(1, tot_esc_spans)) * 100.0

    # --------------------------------------------------------------------------
    # Format canonical audit table
    # --------------------------------------------------------------------------
    audit_table_lines = [
        f"{'Triad Outcome Class':<22} | {'Count':<6} | {'Pct':<7} | Description",
        "-" * 105,
    ]
    for k, v in all_triads.most_common():
        desc = TRIAD_DESCRIPTIONS.get(k, "Arbitration combination outcome")
        audit_table_lines.append(f"{k:<22} | {v:<6} | {v/max(1, tot_spans)*100:>5.2f}% | {desc}")
    audit_table_str = "\n".join(audit_table_lines)

    # --------------------------------------------------------------------------
    # Generate run_summary_report.txt / full_summary.txt
    # --------------------------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 105,
        f"📊 {info['name'].upper()} EXPERIMENT RUN SUMMARY REPORT",
        f"Timestamp: {timestamp_str} | Total Test Abstracts: {len(txt_files)}",
        f"System Name : {info['system_name']}",
        "=" * 105,
        "",
        "1. CANDIDATE ARBITRATION BENCHMARK (All Proposed Mentions)",
        "-" * 105,
        f"  • Candidate Micro F1-Score : {h_mic_p*100:.2f}% / {h_mic_r*100:.2f}% / {h_mic_f1*100:.2f}% (SpanNER: {s_mic_f1*100:.2f}%, Gain: +{(h_mic_f1-s_mic_f1)*100:.2f}%)",
        f"  • Macro F1-Score (Classes) : {h_mac_f1*100:.2f}% (SpanNER: {s_mac_f1*100:.2f}%, Gain: +{(h_mac_f1-s_mac_f1)*100:.2f}%)",
        "",
        "2. GLOBAL END-TO-END DATASET BENCHMARK (All Gold Entities)",
        "-" * 105,
        f"{'Formulation':<25} | {'SpanNER Baseline (P/R/F1)':<28} | {'Pipeline Model (P/R/F1)':<32} | Gain / Delta",
        "-" * 105,
        f"{'Strict CoNLL (1)':<25} | {spanner_results['strict']['p']*100:>5.1f}% / {spanner_results['strict']['r']*100:>5.1f}% / {spanner_results['strict']['f1']*100:>5.1f}% | {hybrid_results['strict']['p']*100:>5.1f}% / {hybrid_results['strict']['r']*100:>5.1f}% / {hybrid_results['strict']['f1']*100:>5.1f}% | {(hybrid_results['strict']['f1'] - spanner_results['strict']['f1'])*100:>+6.2f}% F1",
        f"{'Partial MUC-7 (3)':<25} | {spanner_results['partial']['p']*100:>5.1f}% / {spanner_results['partial']['r']*100:>5.1f}% / {spanner_results['partial']['f1']*100:>5.1f}% | {hybrid_results['partial']['p']*100:>5.1f}% / {hybrid_results['partial']['r']*100:>5.1f}% / {hybrid_results['partial']['f1']*100:>5.1f}% | {(hybrid_results['partial']['f1'] - spanner_results['partial']['f1'])*100:>+6.2f}% F1",
        f"{'Candidate Mention (4)':<25} | {spanner_results['mention']['p']*100:>5.1f}% / {spanner_results['mention']['r']*100:>5.1f}% / {spanner_results['mention']['f1']*100:>5.1f}% | {hybrid_results['mention']['p']*100:>5.1f}% / {hybrid_results['mention']['r']*100:>5.1f}% / {hybrid_results['mention']['f1']*100:>5.1f}% | {(hybrid_results['mention']['f1'] - spanner_results['mention']['f1'])*100:>+6.2f}% F1",
        "",
        "3. ESCALATED SUBPOPULATION BENCHMARK (Local Lift on Ambiguous Spans)",
        "-" * 105,
        f"  • Escalated Spans Volume    : {tot_esc_spans} / {tot_spans} spans ({esc_rate:.1f}% escalated, {100-esc_rate:.1f}% gated locally)",
        f"  • SpanNER on Escalated      : Precision = {esc_s_prec*100:.2f}%, Recall = {esc_s_rec*100:.2f}%, Micro F1 = {esc_s_f1*100:.2f}% (FP = {esc_s_fp})",
        f"  • System on Escalated       : Precision = {esc_h_prec*100:.2f}%, Recall = {esc_h_rec*100:.2f}%, Micro F1 = {esc_h_f1*100:.2f}% (FP = {esc_h_fp})",
        f"  • Isolated Escalated Gain   : Delta P = {(esc_h_prec-esc_s_prec)*100:+.2f}%, Delta R = {(esc_h_rec-esc_s_rec)*100:+.2f}%, Delta F1 = {(esc_h_f1-esc_s_f1)*100:+.2f}%",
        f"  • False Positive Reduction  : {esc_s_fp - esc_h_fp} spurious spans eliminated ({esc_s_fp} -> {esc_h_fp})",
        "",
        "4. ARBITRATION BALANCE SHEET & WIN RATE (Correction vs Degradation Dynamics)",
        "-" * 105,
        f"  • Beneficial Corrections (Positive) : {tot_pos_corr} spans (+{pos_corr_cat} category fixes, +{pos_corr_drop} false positives dropped to O)",
        f"  • Harmful Inversions (Degradations) : {tot_harm_inv} spans (-{harm_corrupt} category corruptions, -{harm_drop} false drops to O)",
        f"  • Preserved Correct Ground Truth    : {pres_correct} spans accurately validated and maintained",
        f"  • Net Corrected Spans               : {net_corrections:+d} spans ({acc_swing:+.2f}% net accuracy swing on escalated tranche)",
        f"  • Arbitration Win Rate              : {win_rate:.2f}% (Positive Corrections / Total Label Modifications)",
        "",
        "5. CANONICAL 10-OUTCOME DIAGNOSTIC BREAKDOWN",
        "-" * 105,
        audit_table_str,
        "",
        "6. PER-CLASS PERFORMANCE BREAKDOWN",
        "-" * 105,
        f"{'Category':<50} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}",
        "-" * 105,
    ]

    for cat in sorted(CANONICAL_10_LABELS):
        if cat == "O":
            continue
        tp = per_class_tp[cat]
        fp = per_class_fp[cat]
        fn = per_class_fn[cat]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        report_lines.append(f"{cat:<50} {tp:<6} {fp:<6} {fn:<6} {p:<10.4f} {r:<10.4f} {f1:<10.4f}")

    report_lines.append("=" * 105)

    report_text = "\n".join(report_lines)

    # Save to both full_summary.txt and run_summary_report.txt
    with open(report_dir / "full_summary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(report_dir / "run_summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save run_summary_metrics.json
    metrics_json = {
        "system_name": info["system_name"],
        "timestamp": timestamp_str,
        "total_test_abstracts": len(txt_files),
        "candidate_arbitration_all_spans": {
            "spanner_micro_f1": s_mic_f1,
            "system_micro_f1": h_mic_f1,
            "delta": h_mic_f1 - s_mic_f1,
            "spanner_macro_f1": s_mac_f1,
            "system_macro_f1": h_mac_f1
        },
        "metrics_3_formulations_global": {
            "strict_conll": {"spanner": spanner_results["strict"], "system": hybrid_results["strict"]},
            "partial_muc7": {"spanner": spanner_results["partial"], "system": hybrid_results["partial"]},
            "candidate_mention": {"spanner": spanner_results["mention"], "system": hybrid_results["mention"]}
        },
        "escalated_subpopulation_benchmark": {
            "total_spans": tot_spans,
            "escalated_spans": tot_esc_spans,
            "unescalated_spans": tot_unesc_spans,
            "escalation_rate_pct": esc_rate,
            "spanner_on_escalated": {"tp": esc_s_tp, "fp": esc_s_fp, "fn": esc_s_fn, "p": esc_s_prec, "r": esc_s_rec, "f1": esc_s_f1},
            "system_on_escalated": {"tp": esc_h_tp, "fp": esc_h_fp, "fn": esc_h_fn, "p": esc_h_prec, "r": esc_h_rec, "f1": esc_h_f1},
            "delta_on_escalated": {
                "delta_tp": esc_h_tp - esc_s_tp,
                "delta_fp": esc_h_fp - esc_s_fp,
                "delta_p": esc_h_prec - esc_s_prec,
                "delta_r": esc_h_rec - esc_s_rec,
                "delta_f1": esc_h_f1 - esc_s_f1
            }
        },
        "arbitration_dynamics": {
            "positive_corrections": tot_pos_corr,
            "category_fixes": pos_corr_cat,
            "spurious_fp_pruned": pos_corr_drop,
            "harmful_inversions": tot_harm_inv,
            "category_corruptions": harm_corrupt,
            "false_rejections": harm_drop,
            "preserved_correct": pres_correct,
            "net_corrected_spans": net_corrections,
            "arbitration_win_rate_pct": win_rate,
            "accuracy_swing_pct": acc_swing
        },
        "canonical_10_outcomes": dict(all_triads)
    }

    with open(report_dir / "run_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    print(f"✅ Recreated full summary and metrics in: {report_dir}")


def main():
    print("=" * 80)
    print("RECREATING ALL FULL SUMMARY REPORTS (PHASE 5 EXPORT FORMAT)")
    print("=" * 80)
    gold_map = load_gold_entities_map()
    print(f"Loaded gold entities for {len(gold_map)} abstracts from .ann files.")

    for info in REPORT_FOLDERS:
        process_report_folder(info, gold_map)


if __name__ == "__main__":
    main()
