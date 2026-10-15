"""
Unified Metrics Calculator for LSF-NER Experiments (Phase 5 / Benchmark Format)
==============================================================================

Computes both per-document and global multi-formulation benchmarks for any experiment
reports directory, matching the standard format used in jev_gated_ace_v3 and others:

1. Candidate Arbitration Benchmark (All Proposed Mentions: Micro-F1, Macro-F1, Gain)
2. Global End-to-End Dataset Benchmark (Strict CoNLL, Partial MUC-7, Candidate Mention)
3. Escalated Subpopulation Benchmark (Local Lift on Ambiguous Spans)
4. Arbitration Balance Sheet & Win Rate (Correction vs Degradation Dynamics)
5. Canonical 10-Outcome Diagnostic Breakdown
6. Per-Class Performance Breakdown

Usage:
  CLI:
    python scripts/calculate_metrics.py --dir reports/jev_ace_gen_unc
    python scripts/calculate_metrics.py --all

  Python Import:
    from scripts.calculate_metrics import calculate_and_write_summary
    calculate_and_write_summary(report_dir="reports/jev_ace_gen_unc", system_name="SpanNER + JEV Uncertainty-Gated ACE Gen")
"""

import os
import sys
import re
import json
import argparse
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
from src.common.audit_logger import classify_10_outcome_str
from src.common.evaluation_framework import (
    evaluate_doc_4_formulations,
    aggregate_4_formulations,
    calc_prf
)

# 10 Canonical Triad Outcome Descriptions
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


def load_gold_entities_map() -> Dict[str, List[Dict[str, Any]]]:
    """Finds and parses .ann files for 400Abstracts dataset (prioritizing no_disease split)."""
    gold_map = {}
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
                        if c_lbl in ("O", "Disease"):
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
    """Parses a per-document report text file to extract span-level decisions and character offsets."""
    content = report_path.read_text(encoding="utf-8")
    doc_id = report_path.stem

    abstract_text = ""
    abs_match = re.search(r"DOCUMENT ABSTRACT TEXT:\s*-+\s*(.*?)\s*={20,}", content, re.DOTALL)
    if abs_match:
        abstract_text = abs_match.group(1).strip()

    spans = []
    span_blocks = re.findall(
        r'\[SPAN \d+/\d+\] "(.*?)" \(Offsets: \[(\d+):(\d+)\]\)(.*?)(?=(?:\[SPAN \d+/\d+\]|={20,}|$))',
        content,
        re.DOTALL
    )
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


def calculate_and_write_summary(
    report_dir: Path | str,
    system_name: Optional[str] = None,
    gold_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Computes all standard benchmarks for a report directory and writes:
    - full_summary.txt
    - run_summary_report.txt
    - run_summary_metrics.json
    """
    report_dir = Path(report_dir)
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    if gold_map is None:
        gold_map = load_gold_entities_map()

    txt_files = sorted([f for f in report_dir.glob("*.txt") if f.stem.isdigit()])
    if not txt_files:
        raise FileNotFoundError(f"No document report .txt files found in {report_dir}")

    if not system_name:
        system_name = report_dir.name.replace("_", " ").title()

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

            triad = classify_10_outcome_str(gt_lbl, s_lbl, f_lbl, escalated=is_esc)
            all_triads[triad] += 1

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

            if is_esc:
                tot_esc_spans += 1
                if s_p:
                    if gt_p and s_lbl == gt_lbl:
                        esc_s_tp += 1
                    else:
                        esc_s_fp += 1
                if gt_p and (not s_p or s_lbl != gt_lbl):
                    esc_s_fn += 1

                if f_p:
                    if gt_p and f_lbl == gt_lbl:
                        esc_h_tp += 1
                    else:
                        esc_h_fp += 1
                if gt_p and (not f_p or f_lbl != gt_lbl):
                    esc_h_fn += 1

                if gt_p:
                    if s_p and s_lbl == gt_lbl:
                        if f_p and f_lbl == gt_lbl:
                            pres_correct += 1
                        elif f_p and f_lbl != gt_lbl:
                            harm_corrupt += 1
                        else:
                            harm_drop += 1
                    elif s_p and s_lbl != gt_lbl:
                        if f_p and f_lbl == gt_lbl:
                            pos_corr_cat += 1
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

        doc_spanner_metrics_list.append(evaluate_doc_4_formulations(doc_spanner_preds, gt_ents, use_new_labels=True))
        doc_hybrid_metrics_list.append(evaluate_doc_4_formulations(doc_hybrid_preds, gt_ents, use_new_labels=True))

    spanner_results = aggregate_4_formulations(doc_spanner_metrics_list)
    hybrid_results = aggregate_4_formulations(doc_hybrid_metrics_list)

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

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 105,
        f"📊 {system_name.upper()} EXPERIMENT RUN SUMMARY REPORT",
        f"Timestamp: {timestamp_str} | Total Test Abstracts: {len(txt_files)}",
        f"System Name : {system_name}",
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
        "A. ESCALATED TRANCHE (Arbitrated by LLM):",
        f"{'Triad Outcome Class':<22} | {'Count':<6} | {'Pct (Total)':<11} | {'Pct (Esc)':<10} | Description",
        "-" * 105,
    ]

    escalated_triad_keys = [k for k in all_triads.keys() if "_LLM_" in k]
    for k in sorted(escalated_triad_keys, key=lambda x: all_triads[x], reverse=True):
        v = all_triads[k]
        desc = TRIAD_DESCRIPTIONS.get(k, "Arbitration combination outcome")
        pct_tot = (v / max(1, tot_spans)) * 100
        pct_esc = (v / max(1, tot_esc_spans)) * 100
        report_lines.append(f"{k:<22} | {v:<6} | {pct_tot:>9.2f}% | {pct_esc:>8.2f}% | {desc}")

    report_lines.extend([
        "",
        "B. CONFIDENT LOCAL TRANCHE (BERT == JEV, Low Uncertainty — Kept Local):",
        f"{'Local Outcome Class':<22} | {'Count':<6} | {'Pct (Total)':<11} | {'Pct (Local)':<11} | Description",
        "-" * 105,
    ])

    local_triad_keys = [k for k in all_triads.keys() if "_LLM_" not in k]
    for k in sorted(local_triad_keys, key=lambda x: all_triads[x], reverse=True):
        v = all_triads[k]
        desc = TRIAD_DESCRIPTIONS.get(k, "Confident local span decision")
        pct_tot = (v / max(1, tot_spans)) * 100
        pct_loc = (v / max(1, tot_unesc_spans)) * 100
        report_lines.append(f"{k:<22} | {v:<6} | {pct_tot:>9.2f}% | {pct_loc:>9.2f}% | {desc}")


    report_lines.extend([
        "",
        "6. PER-CLASS PERFORMANCE BREAKDOWN",
        "-" * 105,
        f"{'Category':<50} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}",
        "-" * 105,
    ])

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
    full_report_str = "\n".join(report_lines)

    # Write output summary files
    out_summary_file = report_dir / "full_summary.txt"
    out_summary_file.write_text(full_report_str, encoding="utf-8")

    out_run_report = report_dir / "run_summary_report.txt"
    out_run_report.write_text(full_report_str, encoding="utf-8")

    metrics_dict = {
        "system_name": system_name,
        "timestamp": timestamp_str,
        "total_abstracts": len(txt_files),
        "total_spans": tot_spans,
        "escalated_spans": tot_esc_spans,
        "candidate_micro_p": round(h_mic_p, 4),
        "candidate_micro_r": round(h_mic_r, 4),
        "candidate_micro_f1": round(h_mic_f1, 4),
        "candidate_macro_f1": round(h_mac_f1, 4),
        "spanner_micro_f1": round(s_mic_f1, 4),
        "strict_conll_f1": round(hybrid_results['strict']['f1'], 4),
        "partial_muc7_f1": round(hybrid_results['partial']['f1'], 4),
        "candidate_mention_f1": round(hybrid_results['mention']['f1'], 4),
        "win_rate": round(win_rate, 2),
        "net_corrections": net_corrections
    }

    out_json = report_dir / "run_summary_metrics.json"
    out_json.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    return full_report_str, metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Calculate complete Phase 5 multi-formulation metrics for LSF-NER runs.")
    parser.add_argument("--dir", type=str, default=None, help="Directory of document reports (e.g. reports/jev_ace_gen_unc)")
    parser.add_argument("--system_name", type=str, default=None, help="System display name")
    parser.add_argument("--all", action="store_true", help="Process all report directories in reports/")

    args = parser.parse_args()

    gold_map = load_gold_entities_map()
    print(f"Loaded gold entities for {len(gold_map)} abstracts.")

    if args.all:
        reports_root = PROJECT_ROOT / "reports"
        for sub in sorted(reports_root.iterdir()):
            if sub.is_dir() and any(f.stem.isdigit() for f in sub.glob("*.txt")):
                print(f"\nProcessing {sub.name}...")
                report_str, _ = calculate_and_write_summary(sub, gold_map=gold_map)
                print(f"✅ Successfully wrote summary to {sub / 'full_summary.txt'}")
    elif args.dir:
        target_dir = Path(args.dir)
        report_str, _ = calculate_and_write_summary(target_dir, system_name=args.system_name, gold_map=gold_map)
        print(report_str)
        print(f"\n✅ Successfully wrote full summary to {target_dir / 'full_summary.txt'}")
    else:
        # Default to jev_ace_gen_unc
        target_dir = PROJECT_ROOT / "reports" / "jev_ace_gen_unc"
        report_str, _ = calculate_and_write_summary(target_dir, system_name="SpanNER + JEV Uncertainty-Gated ACE Gen", gold_map=gold_map)
        print(report_str)
        print(f"\n✅ Successfully wrote full summary to {target_dir / 'full_summary.txt'}")


if __name__ == "__main__":
    main()
