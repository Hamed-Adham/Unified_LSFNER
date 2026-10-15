"""
JEV-Gated ACE Generator v3 Evaluation Runner
============================================

Evaluates ACE_Generator_v3_bob.txt on all 80 validation abstracts from
data/cached/spanner_160_10_class/400/val using 9router LLM Backend with
JEV Gating arbitration:
- Condition to escalate to LLM: BERT Prediction != JEV Decision
- Spans where BERT == JEV are resolved locally and act as [ANCHOR] context
- Escalated spans receive rich diagnostic evidence (BERT prior + JEV decision + JEV probabilities + uncertainty)

Generates:
1. Detailed per-abstract reports in `./reports/jev_gated_ace_v3/{doc_id}.txt`
2. Prompt files for the first 3 escalated abstracts: `./reports/jev_gated_ace_v3/prompt_{doc_id}.txt`
3. Comprehensive global summary in `./reports/jev_gated_ace_v3/full_summary.txt`
   with full breakdown across all 10 canonical Triad outcome classes.
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    call_single_jev_decision,
    JevDecisionResult,
    extract_sentence_context,
    load_jev_state_template,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("jev_gated_ace_eval")

# ==============================================================================
# RUN CONFIGURATION (Parameters for direct IDE execution)
# ==============================================================================
DEFAULT_INPUT_DIR = "data/cached/spanner_160_10_class/400/val"
DEFAULT_OUTPUT_DIR = "./reports/jev_gated_ace_v3"
JEV_CACHE_DIR = "./reports/jev_cache"
PROMPT_TEMPLATE_PATH = "src/prompts/agentic_guidelines/ACE_Generator_v3_bob.txt"
GUIDEBOOK_PATH = "data/guidelines/dynamicGuidebook.json"
LLM_BACKEND = "9router"
JEV_WORKERS = 10
LIMIT = None  # None = all 80 files


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


def load_or_fetch_doc_jev_decisions(
    doc_id: str,
    abstract_text: str,
    spans: List[Dict[str, Any]],
    cache_dir: Path,
    max_workers: int = 10
) -> Dict[int, JevDecisionResult]:
    """
    Loads all JEV decisions for a document from reports/jev_cache/{doc_id}.json.
    If missing or incomplete, concurrently fetches missing spans via OpenRouter and updates the document cache.
    """
    cache_file = cache_dir / f"{doc_id}.json"
    doc_cache: Dict[str, Any] = {}

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                doc_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read document cache {cache_file}: {e}")
            doc_cache = {}

    results: Dict[int, JevDecisionResult] = {}
    missing_spans = []

    for s_idx, span in enumerate(spans, 1):
        str_idx = str(s_idx)
        if str_idx in doc_cache:
            entry = doc_cache[str_idx]
            results[s_idx] = JevDecisionResult(
                span_id=entry.get("span_id", f"span_{s_idx}"),
                span_text=entry.get("span_text", span.get("span_text", "")),
                is_lsf_entity=entry.get("is_lsf_entity", 0.0),
                predicted_category=entry.get("predicted_category", "O"),
                category_probabilities=entry.get("category_probabilities", {}),
                gating_uncertainty=entry.get("gating_uncertainty", 0.0),
                should_escalate_to_llm=entry.get("should_escalate_to_llm", False),
                raw_response=entry.get("raw_response", {})
            )
        else:
            missing_spans.append((s_idx, span))

    if missing_spans:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(call_single_jev_decision, s, abstract_text): s_idx
                for s_idx, s in missing_spans
            }
            for future in as_completed(future_to_idx):
                s_idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[s_idx] = res
                    doc_cache[str(s_idx)] = {
                        "span_id": res.span_id,
                        "span_text": res.span_text,
                        "predicted_category": res.predicted_category,
                        "is_lsf_entity": res.is_lsf_entity,
                        "gating_uncertainty": res.gating_uncertainty,
                        "category_probabilities": res.category_probabilities,
                        "should_escalate_to_llm": res.should_escalate_to_llm,
                    }
                except Exception as e:
                    logger.error(f"JEV API call failed for doc {doc_id} span {s_idx}: {e}")
                    results[s_idx] = JevDecisionResult(
                        span_id=f"span_{s_idx}",
                        span_text=spans[s_idx - 1].get("span_text", ""),
                        is_lsf_entity=0.0,
                        predicted_category="O",
                        category_probabilities={"O": 1.0},
                        gating_uncertainty=1.0,
                        should_escalate_to_llm=True,
                        raw_response={}
                    )

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(doc_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write document cache {cache_file}: {e}")

    return results


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

    # Sort events: descending by character offset; for ties, close (0) before open (1)
    events.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    res = abstract
    for pos, _, _, _, tag in events:
        res = res[:pos] + tag + res[pos:]
    return res


def format_candidate_spans_with_jev_json(candidate_spans: List[Dict[str, Any]]) -> str:
    """Formats candidate spans with both BERT and JEV diagnostic evidence into JSON string."""
    evidence_list = []
    for c in candidate_spans:
        top_probs = dict(sorted(c.get("jev_category_probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:3])
        evidence_list.append({
            "id": c["id"],
            "span_text": c["span_text"],
            "bert_prior": {
                "predicted_label": c.get("bert_predicted_label", "O"),
                "confidence": round(float(c.get("bert_top1_prob", 0.0)), 4),
                "margin": round(float(c.get("bert_margin", 0.0)), 4),
            },
            "jev_decision": {
                "predicted_label": c.get("jev_predicted_label", "O"),
                "is_lsf_entity_score": round(float(c.get("jev_is_lsf_entity", 0.0)), 4),
                "uncertainty_score": round(float(c.get("jev_uncertainty", 0.0)), 4),
                "top_probabilities": top_probs,
            }
        })
    return json.dumps(evidence_list, indent=2)


def parse_ace_generator_response(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts reasoning block and JSON resolutions from ACE Generator v3 response."""
    reasoning = ""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    json_str = raw_text
    json_block = re.search(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", raw_text, re.DOTALL)
    if json_block:
        json_str = json_block.group(1)
    else:
        bracket_match = re.search(r"(\[\s*\{.*\}\s*\])", raw_text, re.DOTALL)
        if bracket_match:
            json_str = bracket_match.group(1)

    parsed_items = []
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            parsed_items = parsed
    except Exception:
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
        f"JEV-GATED ACE GENERATOR V3 REPORT: Document {doc_id}",
        "=" * 80,
        f"Generated At          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Spans           : {total_spans}",
        f"Agreed (Local BERT)   : {confident_count} ({confident_count/max(1, total_spans)*100:.1f}%)",
        f"Disagreed (Escalated) : {escalated_count} ({escalated_count/max(1, total_spans)*100:.1f}%)",
        "",
        "-" * 80,
        "DOCUMENT PERFORMANCE SUMMARY:",
        "-" * 80,
        f"BERT Alone         : P={bert_metrics['precision']:.4f} | R={bert_metrics['recall']:.4f} | Micro-F1={bert_metrics['f1']:.4f} (TP={bert_metrics['tp']}, FP={bert_metrics['fp']}, FN={bert_metrics['fn']})",
        f"JEV Alone          : P={jev_metrics['precision']:.4f} | R={jev_metrics['recall']:.4f} | Micro-F1={jev_metrics['f1']:.4f} (TP={jev_metrics['tp']}, FP={jev_metrics['fp']}, FN={jev_metrics['fn']})",
        f"JEV-Gated ACE Gen  : P={final_metrics['precision']:.4f} | R={final_metrics['recall']:.4f} | Micro-F1={final_metrics['f1']:.4f} (TP={final_metrics['tp']}, FP={final_metrics['fp']}, FN={final_metrics['fn']})",
        f"F1 Delta vs BERT   : {final_metrics['f1'] - bert_metrics['f1']:+.4f}",
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
        lines.extend([
            f"\n[SPAN {s['global_id']}/{total_spans}] \"{s['span_text']}\" (Offsets: [{s['start_char']}:{s['end_char']}])",
            f"  * Ground Truth (GT)   : {s['gt_label']} (is_entity: {s['gt_is_entity']})",
            f"  * BERT Prior          : {s['bert_label']} [Top1 Prob: {s['bert_top1_prob']:.4f}, Margin: {s['bert_margin']:.4f}]",
            f"  * JEV System One      : {s['jev_predicted_label']} [Entity Prob: {s['jev_is_lsf_entity']:.4f}, Unc: {s['jev_uncertainty']:.4f}, Probs: {{{top_probs_str}}}]",
            f"  * Gating Decision     : {'⚡ ESCALATED TO GENERATOR (BERT != JEV)' if s['escalated'] else '✓ RESOLVED LOCALLY (BERT == JEV)'}",
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
        "bert_metrics": bert_metrics,
        "jev_metrics": jev_metrics,
        "final_metrics": final_metrics,
        "triad_counts": triad_counts,
        "spans_data": spans_data
    }


def main():
    input_dir = Path(DEFAULT_INPUT_DIR)
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    cache_dir = Path(JEV_CACHE_DIR)
    prompt_path = Path(PROMPT_TEMPLATE_PATH)
    guidebook_path = Path(GUIDEBOOK_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING JEV-GATED ACE GENERATOR V3 EVALUATION")
    logger.info(f"Input Directory  : {input_dir}")
    logger.info(f"Output Directory : {output_dir}")
    logger.info(f"JEV Cache Dir    : {cache_dir}")
    logger.info(f"Prompt Template  : {prompt_path}")
    logger.info(f"DynamicGuidebook : {guidebook_path}")
    logger.info(f"Gating Policy    : Escalate when BERT predicted_label != JEV predicted_category")
    logger.info(f"JEV Worker Pool  : {JEV_WORKERS} concurrent threads")
    logger.info("=" * 80)

    # 1. Load Prompt Template & Guidebook
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    with open(guidebook_path, "r", encoding="utf-8") as f:
        dynamic_guidebook_content = f.read().strip()

    # 2. Initialize LLM Client
    llm = get_llm_provider(LLM_BACKEND)

    # 3. Find and sort JSON files
    json_files = sorted(list(input_dir.glob("*.json")))
    if LIMIT:
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

        # Fetch or load cached JEV decisions per document
        jev_decisions = load_or_fetch_doc_jev_decisions(
            doc_id=doc_id,
            abstract_text=abstract_text,
            spans=spans,
            cache_dir=cache_dir,
            max_workers=JEV_WORKERS
        )

        # 1. Gating step: BERT vs JEV comparison
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

            jev_res = jev_decisions[s_idx]
            jev_label = normalize_label(jev_res.predicted_category)

            # Core Gating Rule: Escalate if BERT != JEV
            escalate = (bert_label != jev_label)

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
                "jev_is_lsf_entity": jev_res.is_lsf_entity,
                "jev_uncertainty": jev_res.gating_uncertainty,
                "jev_category_probabilities": jev_res.category_probabilities,
                "escalated": escalate,
                "final_label": bert_label,
                "llm_label": "N/A",
                "bullet_ids": [],
                "rationale": "Resolved locally (BERT == JEV agreement)."
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
            logger.info(f"[{idx}/{len(json_files)}] Doc {doc_id}: Escalating {escalated_count}/{len(spans)} spans to ACE Generator v3...")

            anchored_abstract = build_gated_anchored_abstract(abstract_text, uncertain_spans, confident_spans)
            candidate_spans_json = format_candidate_spans_with_jev_json(uncertain_spans)

            prompt_content = prompt_template
            prompt_content = prompt_content.replace("{abstract}", anchored_abstract)
            prompt_content = prompt_content.replace("{candidate_spans_str}", candidate_spans_json)
            prompt_content = prompt_content.replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', dynamic_guidebook_content)
            prompt_content = prompt_content.replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', dynamic_guidebook_content)
            prompt_content = prompt_content.replace("{dynamicGuidebook}", dynamic_guidebook_content)
            prompt_content = prompt_content.replace("{dynamicGuidelines}", dynamic_guidebook_content)
            prompt_content = prompt_content.replace("{{ABSTRACT_TEXT}}", anchored_abstract)
            prompt_content = prompt_content.replace("{{CANDIDATE_SPANS_JSON}}", candidate_spans_json)
            prompt_content = prompt_content.replace("{{DYNAMIC_GUIDEBOOK}}", dynamic_guidebook_content)

            # Save the prompt for the first 3 escalated abstracts
            if saved_prompts_count < 3:
                prompt_file = output_dir / f"prompt_{doc_id}.txt"
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(prompt_content)
                saved_prompts_count += 1
                logger.info(f"  -> Saved filled prompt artifact [{saved_prompts_count}/3]: {prompt_file.name}")

            # Call LLM
            t0 = time.time()
            try:
                raw_response = llm.generate(
                    prompt=prompt_content,
                    temperature=0.0,
                    max_tokens=6128
                )
                dt = time.time() - t0
                logger.info(f"  -> LLM response received in {dt:.2f}s")
            except Exception as e:
                logger.error(f"  -> LLM call failed for Doc {doc_id}: {e}")
                raw_response = "[]"

            # Parse Generator response
            reasoning_text, parsed_resolutions = parse_ace_generator_response(raw_response or "")
            res_by_id = {item.get("id"): item for item in parsed_resolutions if "id" in item}

            # Map LLM predictions back
            for item in uncertain_spans:
                cid = item["id"]
                if cid in res_by_id:
                    res = res_by_id[cid]
                    assigned = normalize_label(res.get("category", "O"))
                    item["llm_label"] = assigned
                    item["final_label"] = assigned
                    item["bullet_ids"] = res.get("bullet_ids", [])
                    item["rationale"] = res.get("rationale", "")
                else:
                    item["llm_label"] = item["bert_label"]
                    item["final_label"] = item["bert_label"]
                    item["rationale"] = "LLM omitted span ID; defaulted to BERT prior."
        else:
            logger.info(f"[{idx}/{len(json_files)}] Doc {doc_id}: 0/{len(spans)} escalated. All resolved locally (BERT == JEV).")

        # Classify Triad for each span
        for item in all_spans_list:
            if item["escalated"]:
                triad = classify_10_outcome_str(item["gt_label"], item["bert_label"], item["final_label"])
            else:
                if item["gt_label"] == "O" and item["bert_label"] != "O":
                    triad = "GT_O_BERT_X"
                elif item["gt_label"] != "O" and item["bert_label"] == item["gt_label"]:
                    triad = "GT_X_BERT_X"
                elif item["gt_label"] != "O" and item["bert_label"] != "O" and item["bert_label"] != item["gt_label"]:
                    triad = "GT_X_BERT_Y"
                else:
                    triad = "GT_O_BERT_O"
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

    # Global Summarization
    t_total = time.time() - t_start_all
    logger.info("=" * 80)
    logger.info(f"COMPLETED EVALUATION IN {t_total:.2f}s")
    logger.info("Compiling global performance metrics across all 80 validation abstracts...")

    all_gt = []
    all_bert = []
    all_jev = []
    all_final = []
    all_triads = Counter()
    total_spans_global = 0
    total_escalated_global = 0

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for doc in all_doc_results:
        for s in doc["spans_data"]:
            total_spans_global += 1
            if s["escalated"]:
                total_escalated_global += 1
            all_gt.append(s["gt_label"])
            all_bert.append(s["bert_label"])
            all_jev.append(s["jev_predicted_label"])
            all_final.append(s["final_label"])
            all_triads[s["triad_class"]] += 1

            # Per class metrics
            gt_lbl = s["gt_label"]
            pred_lbl = s["final_label"]
            if gt_lbl != "O" and pred_lbl != "O":
                if gt_lbl == pred_lbl:
                    per_class_tp[gt_lbl] += 1
                else:
                    per_class_fp[pred_lbl] += 1
                    per_class_fn[gt_lbl] += 1
            elif gt_lbl == "O" and pred_lbl != "O":
                per_class_fp[pred_lbl] += 1
            elif gt_lbl != "O" and pred_lbl == "O":
                per_class_fn[gt_lbl] += 1

    global_bert = calculate_metrics_dict(all_gt, all_bert)
    global_jev = calculate_metrics_dict(all_gt, all_jev)
    global_final = calculate_metrics_dict(all_gt, all_final)

    summary_lines = [
        "=" * 80,
        "JEV-GATED ACE GENERATOR V3 EVALUATION: FULL VALIDATION SUMMARY",
        "=" * 80,
        f"Dataset Split         : {DEFAULT_INPUT_DIR}",
        f"Prompt Template       : {PROMPT_TEMPLATE_PATH}",
        f"DynamicGuidebook      : {GUIDEBOOK_PATH}",
        f"LLM Backend           : {LLM_BACKEND}",
        f"Gating Rule           : Escalate to LLM if BERT Predicted Label != JEV Decision",
        f"Execution Time        : {t_total:.2f}s (Average: {t_total/len(all_doc_results):.2f}s/doc)",
        f"Total Documents       : {len(all_doc_results)}",
        f"Total Candidate Spans : {total_spans_global}",
        f"Total Escalated Spans : {total_escalated_global} ({total_escalated_global/max(1, total_spans_global)*100:.2f}%)",
        f"Total Local (BERT==JEV): {total_spans_global - total_escalated_global} ({(total_spans_global - total_escalated_global)/max(1, total_spans_global)*100:.2f}%)",
        "",
        "=" * 80,
        "GLOBAL PERFORMANCE COMPARISON (MICRO-AVERAGED):",
        "=" * 80,
        "1. BERT Alone (Baseline):",
        f"   - Precision : {global_bert['precision']:.4f} ({global_bert['tp']}/{global_bert['tp']+global_bert['fp']})",
        f"   - Recall    : {global_bert['recall']:.4f} ({global_bert['tp']}/{global_bert['tp']+global_bert['fn']})",
        f"   - Micro-F1  : {global_bert['f1']:.4f}",
        "",
        "2. JEV System One Alone:",
        f"   - Precision : {global_jev['precision']:.4f} ({global_jev['tp']}/{global_jev['tp']+global_jev['fp']})",
        f"   - Recall    : {global_jev['recall']:.4f} ({global_jev['tp']}/{global_jev['tp']+global_jev['fn']})",
        f"   - Micro-F1  : {global_jev['f1']:.4f}",
        "",
        "3. JEV-Gated ACE Generator v3 (BERT + JEV Gating + ACE Generator):",
        f"   - Precision : {global_final['precision']:.4f} ({global_final['tp']}/{global_final['tp']+global_final['fp']})",
        f"   - Recall    : {global_final['recall']:.4f} ({global_final['tp']}/{global_final['tp']+global_final['fn']})",
        f"   - Micro-F1  : {global_final['f1']:.4f}",
        "",
        "4. Delta (JEV-Gated ACE Generator vs BERT Baseline):",
        f"   - Precision Delta : {global_final['precision'] - global_bert['precision']:+.4f}",
        f"   - Recall Delta    : {global_final['recall'] - global_bert['recall']:+.4f}",
        f"   - Micro-F1 Delta  : {global_final['f1'] - global_bert['f1']:+.4f}",
        "",
        "=" * 80,
        "10 CANONICAL TRIAD OUTCOME CLASSES BREAKDOWN:",
        "=" * 80,
        ""
    ]

    for triad_key in sorted(TRIAD_DESCRIPTIONS.keys()):
        count = all_triads.get(triad_key, 0)
        pct = (count / max(1, total_spans_global)) * 100
        desc = TRIAD_DESCRIPTIONS[triad_key]
        summary_lines.extend([
            f"- {triad_key} ({count} spans | {pct:.2f}%):",
            f"  {desc}",
            ""
        ])

    summary_lines.extend([
        "-" * 80,
        "CONFIDENT LOCAL SPANS (BERT == JEV AGREEMENT):",
        "-" * 80,
        f"- GT_O_BERT_X          : {all_triads.get('GT_O_BERT_X', 0)} spans ({all_triads.get('GT_O_BERT_X', 0)/max(1, total_spans_global)*100:.2f}%)",
        f"- GT_X_BERT_X          : {all_triads.get('GT_X_BERT_X', 0)} spans ({all_triads.get('GT_X_BERT_X', 0)/max(1, total_spans_global)*100:.2f}%)",
        f"- GT_X_BERT_Y          : {all_triads.get('GT_X_BERT_Y', 0)} spans ({all_triads.get('GT_X_BERT_Y', 0)/max(1, total_spans_global)*100:.2f}%)",
        "",
        "=" * 80,
        "PER-CLASS PERFORMANCE BREAKDOWN (JEV-GATED ACE GENERATOR V3):",
        "=" * 80,
        f"{'Category':<50} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}",
        "-" * 98,
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
        summary_lines.append(f"{cat:<50} {tp:<6} {fp:<6} {fn:<6} {p:<10.4f} {r:<10.4f} {f1:<10.4f}")

    summary_lines.append("=" * 80)

    summary_text = "\n".join(summary_lines)
    summary_file = output_dir / "full_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    logger.info(f"All 80 document reports and full summary written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
