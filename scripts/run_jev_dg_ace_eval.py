"""
JEV Dynamic Guidebook-Gated ACE Generator Evaluation Runner
===========================================================

Evaluates ACE_Generator_v3_bob.txt on validation abstracts from
data/cached/spanner_160_10_class/400/val using 9Router JEV System One gating
and 9Router LLM Backend with the following decision policy:

  Gating Rule:
    - Escalate to LLM if:
        1. bert_label != jev_label
        2. bert_label == jev_label AND jev_uncertainty > 0.6
    - Otherwise:
        Keep local decision (BERT prior)

Generates:
1. Cached JEV decisions in ./reports/jev_dg_cache/{doc_id}.json
2. Detailed per-abstract reports in ./reports/jev_dg_ace_eval/{doc_id}.txt
3. Prompt files for first 3 escalated abstracts: ./reports/jev_dg_ace_eval/prompt_{doc_id}.txt
4. Comprehensive global summary in ./reports/jev_dg_ace_eval/full_summary.txt
   using scripts.calculate_metrics.
"""

import os
import sys
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.llm_client import get_llm_provider
from src.common.label_mapping import canonicalize_label, to_new_label, CANONICAL_10_LABELS
from src.common.audit_logger import classify_10_outcome_str
from src.architectures.jev_gating.jev_call import (
    call_two_stage_doc_jev,
    call_batch_jev_decisions,
    load_jev_state_template,
    load_jev_questions,
    format_guidelines_from_json,
    extract_guideline_ids_from_json,
    extract_guideline_ids_from_text,
    JevDecisionResult
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("jev_dg_ace_eval")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "data" / "cached" / "spanner_160_10_class" / "400" / "val",
    PROJECT_ROOT / "data" / "cached" / "spanner_160_10" / "400" / "val"
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "jev_2stage"
JEV_CACHE_DIR = PROJECT_ROOT / "reports" / "jev_2stage_cache"
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "ACE_Generator_v3_bob.txt"
GUIDEBOOK_PATH = PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidebook.json"
EXAMPLES_PATH = PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidebook_examples.json"
LLM_BACKEND = "9router"
UNCERTAINTY_THRESHOLD = 0.6  # Escalate when BERT == JEV and unc > 0.6
LIMIT = None  # None = all validation files

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
    "GT_X_BERT_Y_LLM_Y": "BERT proposed an incorrect LSF category, and the Generator uncritically adopted it instead of correcting to Ground Truth."
}


def normalize_label(label: str) -> str:
    """Normalizes label string to canonical categories or 'O'."""
    if not label:
        return "O"
    clean = label.strip().replace(" ", "_")
    if clean.lower() in ("o", "non_lsf", "non-lsf", "none", "background", "other"):
        return "O"
    return canonicalize_label(clean)


def load_or_fetch_jev_decisions_for_doc(
    doc_id: str,
    abstract_text: str,
    spans: List[Dict[str, Any]],
    cache_dir: Path,
    formatted_guidelines: str,
    guideline_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Loads JEV decisions from cache or calls Two-Stage Document-Level JEV 9Router API and saves cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{doc_id}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read JEV cache {cache_file}: {e}. Re-fetching...")

    # Prepare candidate spans for this document
    spans_to_call = []
    for s_idx, s in enumerate(spans, 1):
        span_text = s.get("span_text", "")
        boundaries = s.get("boundaries", {})
        start_char = boundaries.get("start_char", 0)
        end_char = boundaries.get("end_char", len(span_text))
        spanner_info = s.get("spanner_evidence", {})
        bert_label = normalize_label(spanner_info.get("predicted_label", "O"))

        spans_to_call.append({
            "id": s_idx,
            "span_id": s_idx,
            "span_text": span_text,
            "start_char": start_char,
            "end_char": end_char,
            "bert_predicted_label": bert_label
        })

    logger.info(f"Calling Two-Stage JEV API for doc {doc_id} ({len(spans_to_call)} candidate spans)...")
    jev_results = call_two_stage_doc_jev(
        doc_id=doc_id,
        abstract_text=abstract_text,
        spans=spans_to_call,
        guidelines=formatted_guidelines,
        guideline_ids=guideline_ids,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        timeout=60.0
    )

    doc_jev_cache = {}
    has_unrecovered_error = False
    for sid, dec in jev_results.items():
        if dec.gating_uncertainty == 2.0 and not dec.category_probabilities:
            has_unrecovered_error = True
        logger.info(f"  -> JEV [{sid}]: \"{dec.span_text}\" => Pred={dec.predicted_category} (P_lsf={dec.is_lsf_entity:.2f}, Unc={dec.gating_uncertainty:.2f}, Guideline={dec.selected_guideline})")
        doc_jev_cache[str(sid)] = {
            "span_id": dec.span_id,
            "span_text": dec.span_text,
            "is_lsf_entity": dec.is_lsf_entity,
            "predicted_category": dec.predicted_category,
            "category_probabilities": dec.category_probabilities,
            "gating_uncertainty": dec.gating_uncertainty,
            "should_escalate_to_llm": dec.should_escalate_to_llm,
            "selected_guideline": dec.selected_guideline,
            "guideline_scores": dec.guideline_scores,
            "selected_guidelines": dec.selected_guidelines
        }

    if not has_unrecovered_error and doc_jev_cache:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(doc_jev_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write JEV cache {cache_file}: {e}")

    return doc_jev_cache


def build_gated_anchored_abstract(
    abstract: str,
    uncertain_spans: List[Dict[str, Any]],
    confident_spans: List[Dict[str, Any]]
) -> str:
    """
    Inserts anchors into abstract:
    - [id: {id}, span: {text}] for uncertain candidate spans to be evaluated by Generator
    - [ANCHOR: {text} ({label})] for high-confidence local spans serving as semantic context
    """
    events = []
    for c in uncertain_spans:
        cid = c["id"]
        s_text = c["span_text"]
        start_char = c.get("start_char", 0)
        end_char = c.get("end_char", len(s_text))

        if start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
            found = abstract.find(s_text)
            if found != -1:
                start_char = found
                end_char = found + len(s_text)
            else:
                continue

        span_len = end_char - start_char
        events.append((end_char, 0, span_len, cid, "]"))
        events.append((start_char, 1, -span_len, cid, f"[id: {cid}, span: "))

    for c in confident_spans:
        s_text = c["span_text"]
        lbl = c["bert_predicted_label"]
        start_char = c.get("start_char", 0)
        end_char = c.get("end_char", len(s_text))

        if start_char < 0 or end_char > len(abstract) or abstract[start_char:end_char] != s_text:
            found = abstract.find(s_text)
            if found != -1:
                start_char = found
                end_char = found + len(s_text)
            else:
                continue

        span_len = end_char - start_char
        events.append((end_char, 0, span_len, -1, f" ({lbl})]"))
        events.append((start_char, 1, -span_len, -1, "[ANCHOR: "))

    events.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    res = abstract
    for pos, _, _, _, tag in events:
        res = res[:pos] + tag + res[pos:]
    return res


def format_candidate_spans_with_jev_json(
    candidate_spans: List[Dict[str, Any]],
    guidebook_map: Optional[Dict[str, Any]] = None
) -> str:
    """Formats candidate spans with BERT evidence, JEV Stage 1 decision, and JEV Stage 2 anchored guideline into JSON string."""
    evidence_list = []
    for c in candidate_spans:
        top_probs = dict(sorted(c.get("jev_category_probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:3])
        g_id = c.get("jev_selected_guideline", "None")
        g_info = "None"
        if guidebook_map and g_id in guidebook_map and g_id != "None":
            g_item = guidebook_map[g_id]
            g_info = {
                "id": g_id,
                "guideline": g_item.get("guideline", "")
            }

        evidence_list.append({
            "id": c["id"],
            "span_text": c["span_text"],
            "bert_prior": {
                "predicted_label": c.get("bert_predicted_label", "O"),
                "confidence": round(float(c.get("bert_top1_prob", 0.0)), 4),
                "margin": round(float(c.get("bert_margin", 0.0)), 4),
            },
            "jev_stage1_decision": {
                "predicted_category": c.get("jev_predicted_label", "O"),
                "is_lsf_entity_score": round(float(c.get("jev_is_lsf_entity", 0.0)), 4),
                "uncertainty_score": round(float(c.get("jev_uncertainty", 0.0)), 4),
                "top_probabilities": top_probs,
                "escalation_reason": c.get("escalation_reason", "N/A"),
            },
            "anchored_guideline_from_stage2": g_info
        })
    return json.dumps(evidence_list, indent=2)


def parse_ace_generator_response(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts reasoning block and JSON resolutions from ACE Generator response."""
    reasoning = ""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    cleaned_json_str = raw_text.strip()
    if "```json" in cleaned_json_str:
        cleaned_json_str = cleaned_json_str.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned_json_str:
        cleaned_json_str = cleaned_json_str.split("```", 1)[1].split("```", 1)[0].strip()

    parsed_items = []
    try:
        parsed = json.loads(cleaned_json_str)
        if isinstance(parsed, dict):
            if not reasoning and "reasoning" in parsed:
                reasoning = parsed.get("reasoning", "")
            ans = parsed.get("llm_answer", parsed.get("labeled_entities", parsed.get("resolutions", [])))
            if isinstance(ans, list):
                parsed_items = ans
        elif isinstance(parsed, list):
            parsed_items = parsed
    except Exception as e:
        logger.warning(f"json.loads failed ({e}); attempting regex fallback extraction...")
        pattern = r'\{\s*"id":\s*(\d+)[^}]*?"llm_predicted_label":\s*"([^"]+)"(?:[^}]*?"rationale":\s*"([^"]*)")?[^}]*?\}'
        matches = re.findall(pattern, cleaned_json_str)
        for m in matches:
            parsed_items.append({
                "id": int(m[0]),
                "llm_predicted_label": m[1],
                "bullet_ids": [],
                "rationale": m[2] if len(m) > 2 else "Extracted via fallback parser"
            })
        if not parsed_items:
            item_matches = re.findall(r"\{[^{}]*\"id\"[^{}]*\}", raw_text, re.DOTALL)
            for im in item_matches:
                try:
                    obj = json.loads(im)
                    parsed_items.append(obj)
                except Exception:
                    continue

    return reasoning, parsed_items


def calculate_metrics_dict(gt_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
    """Calculates Micro-Precision, Recall, and F1."""
    tp = 0
    fp = 0
    fn = 0

    for gt, pred in zip(gt_labels, pred_labels):
        gt_is_ent = (gt != "O")
        pred_is_ent = (pred != "O")

        if gt_is_ent and pred_is_ent:
            if gt == pred:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif not gt_is_ent and pred_is_ent:
            fp += 1
        elif gt_is_ent and not pred_is_ent:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1
    }


def generate_doc_report(
    doc_id: str,
    abstract_text: str,
    spans_data: List[Dict[str, Any]],
    reasoning: str,
    llm_prompt: Optional[str],
    llm_raw_response: Optional[str],
    out_file: Path
) -> Dict[str, Any]:
    """Writes detailed per-document report."""
    total_spans = len(spans_data)
    escalated_count = sum(1 for s in spans_data if s["escalated"])
    confident_count = total_spans - escalated_count
    esc_disagree = sum(1 for s in spans_data if s.get("escalation_reason") == "BERT!=JEV")
    esc_uncertain = sum(1 for s in spans_data if str(s.get("escalation_reason", "")).startswith("UNCERTAINTY>"))

    gt_labels = [s["gt_label"] for s in spans_data]
    bert_labels = [s["bert_label"] for s in spans_data]
    jev_labels = [s["jev_predicted_label"] for s in spans_data]
    final_labels = [s["final_label"] for s in spans_data]

    bert_metrics = calculate_metrics_dict(gt_labels, bert_labels)
    jev_metrics = calculate_metrics_dict(gt_labels, jev_labels)
    final_metrics = calculate_metrics_dict(gt_labels, final_labels)

    triad_counts = Counter(s["triad_class"] for s in spans_data)

    lines = [
        "=" * 80,
        f"JEV DYNAMIC GUIDEBOOK-GATED ACE GENERATOR REPORT: Document {doc_id}",
        "=" * 80,
        f"Generated At          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Gating Uncertainty Th : {UNCERTAINTY_THRESHOLD}",
        f"Total Spans           : {total_spans}",
        f"Agreed + Low Unc (Local)    : {confident_count} ({confident_count/max(1, total_spans)*100:.1f}%)",
        f"Escalated (Total)           : {escalated_count} ({escalated_count/max(1, total_spans)*100:.1f}%)",
        f"  -> BERT!=JEV              : {esc_disagree}",
        f"  -> BERT==JEV, Unc > {UNCERTAINTY_THRESHOLD}   : {esc_uncertain}",
        "",
        "-" * 80,
        "DOCUMENT PERFORMANCE SUMMARY:",
        "-" * 80,
        f"BERT Alone          : P={bert_metrics['precision']:.4f} | R={bert_metrics['recall']:.4f} | Micro-F1={bert_metrics['f1']:.4f} (TP={bert_metrics['tp']}, FP={bert_metrics['fp']}, FN={bert_metrics['fn']})",
        f"JEV Alone           : P={jev_metrics['precision']:.4f} | R={jev_metrics['recall']:.4f} | Micro-F1={jev_metrics['f1']:.4f} (TP={jev_metrics['tp']}, FP={jev_metrics['fp']}, FN={jev_metrics['fn']})",
        f"Unc-Gated ACE Gen   : P={final_metrics['precision']:.4f} | R={final_metrics['recall']:.4f} | Micro-F1={final_metrics['f1']:.4f} (TP={final_metrics['tp']}, FP={final_metrics['fp']}, FN={final_metrics['fn']})",
        f"F1 Delta vs BERT    : {final_metrics['f1'] - bert_metrics['f1']:+.4f}",
        "",
        "-" * 80,
        "TRIAD OUTCOME BREAKDOWN (THIS DOCUMENT):",
        "-" * 80,
    ]

    for triad_key, count in triad_counts.most_common():
        desc = TRIAD_DESCRIPTIONS.get(triad_key, "Other / Non-canonical combination")
        lines.append(f"  * {triad_key:<20} : {count} ({count/max(1, total_spans)*100:.1f}%) -> {desc}")

    if reasoning:
        lines.extend([
            "",
            "-" * 80,
            "GENERATOR REASONING TRACE:",
            "-" * 80,
            reasoning.strip(),
        ])

    lines.extend([
        "",
        "-" * 80,
        "DOCUMENT ABSTRACT TEXT:",
        "-" * 80,
        abstract_text.strip(),
        "",
        "=" * 80,
        "CANDIDATE SPAN BREAKDOWN & DECISION TRACE:",
        "=" * 80,
    ])

    for s in spans_data:
        top_probs_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(s['jev_category_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]])
        esc_reason_str = s.get("escalation_reason", "N/A")
        if s["escalated"]:
            gating_label = f"ESCALATED TO GENERATOR ({esc_reason_str})"
        else:
            gating_label = f"RESOLVED LOCALLY (BERT==JEV, Unc={s['jev_uncertainty']:.4f} <= {UNCERTAINTY_THRESHOLD})"
        lines.extend([
            f"\n[SPAN {s['global_id']}/{total_spans}] \"{s['span_text']}\" (Offsets: [{s['start_char']}:{s['end_char']}])",
            f"  * Ground Truth (GT)   : {s['gt_label']} (is_entity: {s['gt_is_entity']})",
            f"  * BERT Prior          : {s['bert_label']} [Top1 Prob: {s['bert_top1_prob']:.4f}, Margin: {s['bert_margin']:.4f}]",
            f"  * JEV System One      : {s['jev_predicted_label']} [Entity Prob: {s['jev_is_lsf_entity']:.4f}, Unc: {s['jev_uncertainty']:.4f}, Guideline: {s.get('jev_selected_guideline', 'None')}, Probs: {{{top_probs_str}}}]",
            f"  * Gating Decision     : {gating_label}",
            f"  * Generator Label     : {s['llm_label']}",
            f"  * Final Assigned Label: {s['final_label']}",
            f"  * Triad Outcome Class : {s['triad_class']}",
            f"  * DynamicGuidebook IDs: {s.get('bullet_ids', [])}",
            f"  * Generator Rationale : {s.get('rationale', 'N/A')}",
            f"  * Status / Correctness: BERT Correct: {s['gt_label'] == s['bert_label']} | Final Correct: {s['gt_label'] == s['final_label']}",
            "-" * 80
        ])

    if llm_prompt:
        lines.extend([
            "\n" + "=" * 80,
            "ACE GENERATOR RAW LLM RESPONSE:",
            "=" * 80,
            llm_raw_response or "N/A"
        ])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "doc_id": doc_id,
        "total_spans": total_spans,
        "confident_count": confident_count,
        "escalated_count": escalated_count,
        "esc_disagree": esc_disagree,
        "esc_uncertain": esc_uncertain,
        "bert_metrics": bert_metrics,
        "jev_metrics": jev_metrics,
        "final_metrics": final_metrics,
        "triad_counts": triad_counts,
        "spans_data": spans_data
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JEV Dynamic Guidebook-Gated ACE Generator Evaluation Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation documents to evaluate (e.g. 1)")
    parser.add_argument("--doc-id", type=str, default=None, help="Specific doc ID to evaluate")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    # Resolve input directory
    input_dir = None
    for d in DEFAULT_INPUT_DIRS:
        if d.exists():
            input_dir = d
            break
    if not input_dir:
        logger.error(f"None of the candidate input directories exist: {DEFAULT_INPUT_DIRS}")
        return

    output_dir = Path(args.output_dir)
    cache_dir = JEV_CACHE_DIR
    prompt_path = PROMPT_TEMPLATE_PATH
    guidebook_path = GUIDEBOOK_PATH
    examples_path = EXAMPLES_PATH

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING JEV DYNAMIC GUIDEBOOK-GATED ACE GENERATOR EVALUATION")
    logger.info(f"Input Directory  : {input_dir}")
    logger.info(f"Output Directory : {output_dir}")
    logger.info(f"JEV Cache Dir    : {cache_dir}")
    logger.info(f"Prompt Template  : {prompt_path}")
    logger.info(f"DynamicGuidebook : {guidebook_path}")
    logger.info(f"DG Examples Path : {examples_path}")
    logger.info(f"Gating Policy    : bert != jev OR (bert == jev AND unc > {UNCERTAINTY_THRESHOLD})")
    logger.info("=" * 80)

    # 1. Load Generator Prompt Template & Formatted Guidelines
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    with open(guidebook_path, "r", encoding="utf-8") as f:
        dynamic_guidebook_content = f.read().strip()
        try:
            gb_raw = json.loads(dynamic_guidebook_content)
            gb_list = gb_raw.get("guidebook", gb_raw) if isinstance(gb_raw, dict) else gb_raw
            guidebook_map = {item.get("id", f"DG-{i+1:04d}"): item for i, item in enumerate(gb_list)}
        except Exception as e:
            logger.warning(f"Failed to parse guidebook map: {e}")
            guidebook_map = {}

    formatted_dg_for_jev = format_guidelines_from_json(guidebook_path, examples_path, max_examples_per_rule=1)
    guideline_ids = extract_guideline_ids_from_json(guidebook_path)

    # 2. Initialize LLM Client
    llm = get_llm_provider(LLM_BACKEND)

    # 3. Find and sort JSON validation files
    json_files = sorted(list(input_dir.glob("*.json")))
    json_files = [f for f in json_files if f.stem != "all_candidates"]
    if args.doc_id:
        json_files = [f for f in json_files if f.stem == args.doc_id or f.name == f"{args.doc_id}.json"]
    elif args.limit:
        json_files = json_files[:args.limit]
    elif LIMIT:
        json_files = json_files[:LIMIT]

    logger.info(f"Found {len(json_files)} validation files to evaluate.")

    all_doc_results = []
    saved_prompts_count = 0
    t_start_all = time.time()

    for idx, f_path in enumerate(json_files, 1):
        with open(f_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_id = doc_data.get("doc_id", f_path.stem)
        abstract_text = doc_data.get("text", "")
        spans = doc_data.get("spans", [])

        # Load or fetch cached JEV decisions for this document
        jev_cache = load_or_fetch_jev_decisions_for_doc(
            doc_id=doc_id,
            abstract_text=abstract_text,
            spans=spans,
            cache_dir=cache_dir,
            formatted_guidelines=formatted_dg_for_jev,
            guideline_ids=guideline_ids
        )

        uncertain_spans = []
        confident_spans = []
        all_spans_list = []

        for s_idx, s in enumerate(spans, 1):
            span_text = s.get("span_text", "")
            boundaries = s.get("boundaries", {})
            start_char = boundaries.get("start_char", 0)
            end_char = boundaries.get("end_char", len(span_text))

            gt_info = s.get("ground_truth", {})
            gt_label = normalize_label(gt_info.get("gt_label", "O"))
            gt_is_entity = gt_info.get("is_entity", gt_label != "O")

            spanner_info = s.get("spanner_evidence", {})
            bert_label = normalize_label(spanner_info.get("predicted_label", "O"))
            bert_prob = spanner_info.get("top1_prob", 0.0)
            bert_margin = spanner_info.get("margin", 0.0)

            # Load JEV decision
            str_idx = str(s_idx)
            jev_entry = jev_cache.get(str_idx, {})
            jev_label = normalize_label(jev_entry.get("predicted_category", "O"))
            jev_is_lsf = float(jev_entry.get("is_lsf_entity", 0.0))
            jev_uncertainty = float(jev_entry.get("gating_uncertainty", 1.0))
            jev_selected_g = jev_entry.get("selected_guideline", "None")
            jev_probs = jev_entry.get("category_probabilities", {})

            # Gating Logic: bert != jev OR (bert == jev AND unc > 0.6)
            if bert_label != jev_label:
                escalate = True
                escalation_reason = "BERT!=JEV"
            elif jev_uncertainty > UNCERTAINTY_THRESHOLD:
                escalate = True
                escalation_reason = f"UNCERTAINTY>{UNCERTAINTY_THRESHOLD}"
            else:
                escalate = False
                escalation_reason = "LOCAL"

            span_item = {
                "id": s_idx,
                "global_id": s_idx,
                "span_text": span_text,
                "start_char": start_char,
                "end_char": end_char,
                "gt_label": gt_label,
                "gt_is_entity": gt_is_entity,
                "bert_label": bert_label,
                "bert_predicted_label": bert_label,
                "bert_top1_prob": bert_prob,
                "bert_margin": bert_margin,
                "jev_predicted_label": jev_label,
                "jev_is_lsf_entity": jev_is_lsf,
                "jev_uncertainty": jev_uncertainty,
                "jev_selected_guideline": jev_selected_g,
                "jev_category_probabilities": jev_probs,
                "escalated": escalate,
                "escalation_reason": escalation_reason,
                "final_label": bert_label,
                "llm_label": "N/A",
                "bullet_ids": [],
                "rationale": f"Resolved locally (BERT==JEV, Unc={jev_uncertainty:.4f} <= {UNCERTAINTY_THRESHOLD})." if not escalate else ""
            }

            all_spans_list.append(span_item)
            if escalate:
                uncertain_spans.append(span_item)
            else:
                confident_spans.append(span_item)

        escalated_count = len(uncertain_spans)
        reasoning_text = ""
        prompt_content = None
        raw_response = None

        if escalated_count > 0:
            esc_disagree = sum(1 for s in uncertain_spans if s["escalation_reason"] == "BERT!=JEV")
            esc_uncertain = sum(1 for s in uncertain_spans if s["escalation_reason"].startswith("UNCERTAINTY>"))
            logger.info(
                f"[{idx}/{len(json_files)}] Doc {doc_id}: "
                f"Escalating {escalated_count}/{len(spans)} spans to ACE Generator "
                f"(disagree={esc_disagree}, high_unc={esc_uncertain})..."
            )

            anchored_abstract = build_gated_anchored_abstract(abstract_text, uncertain_spans, confident_spans)
            candidate_spans_json = format_candidate_spans_with_jev_json(uncertain_spans, guidebook_map)

            # Build filtered dynamic guidebook with only the selected guidelines for escalated spans
            selected_gids = set()
            for s in uncertain_spans:
                gid = s.get("jev_selected_guideline", "None")
                if gid and gid != "None":
                    selected_gids.add(gid)

            filtered_guidebook_list = []
            for gid in sorted(selected_gids):
                if gid in guidebook_map:
                    g_item = guidebook_map[gid]
                    filtered_guidebook_list.append({
                        "id": gid,
                        "guideline": g_item.get("guideline", "")
                    })

            doc_guidebook_content = json.dumps(filtered_guidebook_list, indent=2, ensure_ascii=False) if filtered_guidebook_list else "[]"

            prompt_content = prompt_template
            prompt_content = prompt_content.replace("{abstract}", anchored_abstract)
            prompt_content = prompt_content.replace("{candidate_spans_str}", candidate_spans_json)
            prompt_content = prompt_content.replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', doc_guidebook_content)
            prompt_content = prompt_content.replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', doc_guidebook_content)
            prompt_content = prompt_content.replace("{dynamicGuidebook}", doc_guidebook_content)
            prompt_content = prompt_content.replace("{dynamicGuidelines}", doc_guidebook_content)
            prompt_content = prompt_content.replace("{{ABSTRACT_TEXT}}", anchored_abstract)
            prompt_content = prompt_content.replace("{{CANDIDATE_SPANS_JSON}}", candidate_spans_json)
            prompt_content = prompt_content.replace("{{DYNAMIC_GUIDEBOOK}}", doc_guidebook_content)

            prompt_file = output_dir / f"prompt_{doc_id}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt_content)
            logger.info(f"  -> Saved generator prompt: {prompt_file.name}")

            # Call LLM with retries
            t0 = time.time()
            llm_retries = 5
            llm_backoff = 4.0
            raw_response = "[]"

            for l_attempt in range(1, llm_retries + 1):
                try:
                    raw_response = llm.generate(
                        prompt=prompt_content,
                        temperature=0.0,
                        max_tokens=6128
                    )
                    dt = time.time() - t0
                    logger.info(f"  -> LLM response received in {dt:.2f}s")
                    break
                except Exception as e:
                    if l_attempt < llm_retries:
                        wait_t = llm_backoff * (2 ** (l_attempt - 1))
                        logger.warning(f"  -> LLM call failed for Doc {doc_id} on attempt {l_attempt} ({e}). Retrying in {wait_t:.1f}s...")
                        time.sleep(wait_t)
                    else:
                        logger.error(f"  -> LLM call permanently failed for Doc {doc_id} after {llm_retries} attempts: {e}")
                        raw_response = "[]"

            raw_resp_file = output_dir / f"raw_response_{doc_id}.txt"
            with open(raw_resp_file, "w", encoding="utf-8") as f:
                f.write(raw_response or "")

            # Parse Generator response
            reasoning_text, parsed_resolutions = parse_ace_generator_response(raw_response or "")
            res_by_id = {}
            for item in parsed_resolutions:
                if "id" in item:
                    res_by_id[item["id"]] = item
                    try:
                        res_by_id[int(item["id"])] = item
                    except Exception:
                        pass
                if "entity" in item and item["entity"]:
                    res_by_id[str(item["entity"]).strip().lower()] = item

            # Map LLM predictions back
            for item in uncertain_spans:
                cid = item["id"]
                ent_txt = item["span_text"].strip().lower()
                matched_ans = res_by_id.get(cid) or res_by_id.get(ent_txt) or {}
                if matched_ans:
                    raw_pred_lbl = matched_ans.get(
                        "llm_predicted_label",
                        matched_ans.get("category", matched_ans.get("predicted_label", matched_ans.get("label", item["bert_label"])))
                    )
                    assigned = normalize_label(raw_pred_lbl)
                    item["llm_label"] = assigned
                    item["final_label"] = assigned
                    item["bullet_ids"] = matched_ans.get("bullet_ids", [])
                    item["rationale"] = matched_ans.get("rationale", "Arbitrated by ACE Generator.")
                else:
                    item["llm_label"] = item["bert_label"]
                    item["final_label"] = item["bert_label"]
                    item["rationale"] = "LLM omitted span ID; defaulted to BERT prior."
        else:
            logger.info(f"[{idx}/{len(json_files)}] Doc {doc_id}: 0/{len(spans)} escalated. All resolved locally.")

        # Classify Triad for each span
        for item in all_spans_list:
            triad = classify_10_outcome_str(
                gt_label=item["gt_label"],
                spanner_label=item["bert_label"],
                final_label=item["final_label"],
                escalated=item["escalated"]
            )
            item["triad_class"] = triad

        # Generate per-document report
        report_file = output_dir / f"{doc_id}.txt"
        doc_summary = generate_doc_report(
            doc_id=doc_id,
            abstract_text=abstract_text,
            spans_data=all_spans_list,
            reasoning=reasoning_text,
            llm_prompt=prompt_content,
            llm_raw_response=raw_response,
            out_file=report_file
        )
        all_doc_results.append(doc_summary)

    t_total = time.time() - t_start_all
    logger.info("=" * 80)
    logger.info(f"COMPLETED EVALUATION IN {t_total:.2f}s")
    logger.info("Calculating standard multi-formulation benchmarks...")

    # Calculate and write standardized full_summary.txt
    try:
        from scripts.calculate_metrics import calculate_and_write_summary
        full_report_str, _ = calculate_and_write_summary(
            report_dir=output_dir,
            system_name=f"SpanNER + Two-Stage JEV Gated ACE Gen (Unc>{UNCERTAINTY_THRESHOLD})"
        )
        print("\n" + full_report_str)
    except Exception as e:
        logger.error(f"Could not calculate benchmark summary: {e}")

    logger.info(f"All document reports and full summary written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
