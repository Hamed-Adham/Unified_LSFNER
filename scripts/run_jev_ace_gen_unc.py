"""
JEV Uncertainty-Gated ACE Generator Evaluation Runner
=====================================================

Evaluates ACE_Generator_v3_bob.txt on all 80 validation abstracts from
data/cached/spanner_160_10_class/400/val using 9router LLM Backend with
3-tier JEV Uncertainty Gating arbitration:

  Tier 1 (Local):    BERT == JEV  AND  jev_uncertainty < 0.4  -> resolved locally
  Tier 2 (Escalate): BERT == JEV  AND  jev_uncertainty >= 0.4 -> escalate to LLM
  Tier 3 (Escalate): BERT != JEV                               -> escalate to LLM

Generates:
1. Detailed per-abstract reports in  ./reports/jev_ace_gen_unc/{doc_id}.txt
2. Prompt files for the first 3 escalated abstracts: ./reports/jev_ace_gen_unc/prompt_{doc_id}.txt
3. Comprehensive global summary in  ./reports/jev_ace_gen_unc/full_summary.txt
   with full breakdown across all 10 canonical Triad outcome classes.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("jev_ace_gen_unc")

# ==============================================================================
# RUN CONFIGURATION
# ==============================================================================
DEFAULT_INPUT_DIR = "data/cached/spanner_160_10_class/400/val"
DEFAULT_OUTPUT_DIR = "./reports/jev_ace_gen_unc_06"
JEV_CACHE_DIR = "./reports/jev_cache"
PROMPT_TEMPLATE_PATH = "src/prompts/agentic_guidelines/ACE_Generator_v3_bob.txt"
GUIDEBOOK_PATH = "data/guidelines/dynamicGuidebook.json"
LLM_BACKEND = "9router"
UNCERTAINTY_THRESHOLD = 0.6   # Threshold to escalate when BERT == JEV
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


def load_jev_cache(doc_id: str, cache_dir: Path) -> Dict[str, Any]:
    """Loads JEV decisions from reports/jev_cache/{doc_id}.json."""
    cache_file = cache_dir / f"{doc_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read JEV cache {cache_file}: {e}")
    return {}


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
            },
            "escalation_reason": c.get("escalation_reason", "N/A"),
        })
    return json.dumps(evidence_list, indent=2)


def parse_ace_generator_response(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts reasoning block and JSON resolutions from ACE Generator v3 response."""
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
    # Count the two sub-types of escalation
    esc_disagree = sum(1 for s in spans_data if s.get("escalation_reason") == "BERT!=JEV")
    esc_uncertain = sum(1 for s in spans_data if str(s.get("escalation_reason", "")).startswith("UNCERTAINTY>="))


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
        f"JEV UNCERTAINTY-GATED ACE GENERATOR REPORT: Document {doc_id}",
        "=" * 80,
        f"Generated At          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Gating Uncertainty Th : {UNCERTAINTY_THRESHOLD}",
        f"Total Spans           : {total_spans}",
        f"Agreed + Low Unc (Local)    : {confident_count} ({confident_count/max(1, total_spans)*100:.1f}%)",
        f"Escalated (Total)           : {escalated_count} ({escalated_count/max(1, total_spans)*100:.1f}%)",
        f"  -> BERT!=JEV              : {esc_disagree}",
        f"  -> BERT==JEV, Unc >= {UNCERTAINTY_THRESHOLD}  : {esc_uncertain}",
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
            gating_label = f"RESOLVED LOCALLY (BERT==JEV, Unc={s['jev_uncertainty']:.4f} < {UNCERTAINTY_THRESHOLD})"
        lines.extend([
            f"\n[SPAN {s['global_id']}/{total_spans}] \"{s['span_text']}\" (Offsets: [{s['start_char']}:{s['end_char']}])",
            f"  * Ground Truth (GT)   : {s['gt_label']} (is_entity: {s['gt_is_entity']})",
            f"  * BERT Prior          : {s['bert_label']} [Top1 Prob: {s['bert_top1_prob']:.4f}, Margin: {s['bert_margin']:.4f}]",
            f"  * JEV System One      : {s['jev_predicted_label']} [Entity Prob: {s['jev_is_lsf_entity']:.4f}, Unc: {s['jev_uncertainty']:.4f}, Probs: {{{top_probs_str}}}]",
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
    input_dir = Path(DEFAULT_INPUT_DIR)
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    cache_dir = Path(JEV_CACHE_DIR)
    prompt_path = Path(PROMPT_TEMPLATE_PATH)
    guidebook_path = Path(GUIDEBOOK_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING JEV UNCERTAINTY-GATED ACE GENERATOR EVALUATION")
    logger.info(f"Input Directory  : {input_dir}")
    logger.info(f"Output Directory : {output_dir}")
    logger.info(f"JEV Cache Dir    : {cache_dir}")
    logger.info(f"Prompt Template  : {prompt_path}")
    logger.info(f"DynamicGuidebook : {guidebook_path}")
    logger.info(f"Gating Policy    : 3-Tier Uncertainty Gating")
    logger.info(f"  Tier 1 (Local) : BERT==JEV AND jev_uncertainty < {UNCERTAINTY_THRESHOLD}")
    logger.info(f"  Tier 2 (LLM)   : BERT==JEV AND jev_uncertainty >= {UNCERTAINTY_THRESHOLD}")
    logger.info(f"  Tier 3 (LLM)   : BERT != JEV")
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
    # Exclude the all_candidates.json aggregate file
    json_files = [f for f in json_files if f.stem != "all_candidates"]
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

        # Load cached JEV decisions for this document
        jev_cache = load_jev_cache(doc_id, cache_dir)

        # 1. 3-Tier Gating step: BERT vs JEV + uncertainty
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

            # Load JEV from cache
            str_idx = str(s_idx)
            jev_entry = jev_cache.get(str_idx, {})
            jev_label = normalize_label(jev_entry.get("predicted_category", "O"))
            jev_is_lsf = jev_entry.get("is_lsf_entity", 0.0)
            jev_uncertainty = jev_entry.get("gating_uncertainty", 1.0)
            jev_probs = jev_entry.get("category_probabilities", {})

            # ============================================================
            # 3-TIER GATING LOGIC
            # ============================================================
            if bert_label != jev_label:
                # Tier 3: Disagreement -> always escalate
                escalate = True
                escalation_reason = "BERT!=JEV"
            elif jev_uncertainty >= UNCERTAINTY_THRESHOLD:
                # Tier 2: Agreement but high uncertainty -> escalate
                escalate = True
                escalation_reason = f"UNCERTAINTY>={UNCERTAINTY_THRESHOLD}"
            else:
                # Tier 1: Agreement and low uncertainty -> keep local
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
                "jev_category_probabilities": jev_probs,
                "escalated": escalate,
                "escalation_reason": escalation_reason,
                "final_label": bert_label,
                "llm_label": "N/A",
                "bullet_ids": [],
                "rationale": f"Resolved locally (BERT==JEV, Unc={jev_uncertainty:.4f} < {UNCERTAINTY_THRESHOLD})." if not escalate else ""
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
            esc_uncertain = sum(1 for s in uncertain_spans if s["escalation_reason"] == "UNCERTAINTY>=0.4")
            logger.info(
                f"[{idx}/{len(json_files)}] Doc {doc_id}: "
                f"Escalating {escalated_count}/{len(spans)} spans to ACE Generator "
                f"(disagree={esc_disagree}, high_unc={esc_uncertain})..."
            )

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

            # Call LLM (one call per doc via ace_generator prompt)
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

    # ==========================================
    # Global Summarization
    # ==========================================
    t_total = time.time() - t_start_all
    logger.info("=" * 80)
    logger.info(f"COMPLETED EVALUATION IN {t_total:.2f}s")
    logger.info("Compiling global performance metrics across all validation abstracts...")

    all_gt = []
    all_bert = []
    all_jev = []
    all_final = []
    all_triads = Counter()
    total_spans_global = 0
    total_escalated_global = 0
    total_esc_disagree_global = 0
    total_esc_uncertain_global = 0

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for doc in all_doc_results:
        for s in doc["spans_data"]:
            total_spans_global += 1
            if s["escalated"]:
                total_escalated_global += 1
                if s.get("escalation_reason") == "BERT!=JEV":
                    total_esc_disagree_global += 1
                elif s.get("escalation_reason") == "UNCERTAINTY>=0.4":
                    total_esc_uncertain_global += 1
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

    total_local = total_spans_global - total_escalated_global

    summary_lines = [
        "=" * 80,
        "JEV UNCERTAINTY-GATED ACE GENERATOR EVALUATION: FULL VALIDATION SUMMARY",
        "=" * 80,
        f"Dataset Split         : {DEFAULT_INPUT_DIR}",
        f"Prompt Template       : {PROMPT_TEMPLATE_PATH}",
        f"DynamicGuidebook      : {GUIDEBOOK_PATH}",
        f"LLM Backend           : {LLM_BACKEND}",
        f"Gating Rule           : 3-Tier Uncertainty Gating (threshold={UNCERTAINTY_THRESHOLD})",
        f"  Tier 1 (Local)      : BERT==JEV AND unc < {UNCERTAINTY_THRESHOLD}",
        f"  Tier 2 (LLM)        : BERT==JEV AND unc >= {UNCERTAINTY_THRESHOLD}",
        f"  Tier 3 (LLM)        : BERT != JEV",
        f"Execution Time        : {t_total:.2f}s (Average: {t_total/max(1,len(all_doc_results)):.2f}s/doc)",
        f"Total Documents       : {len(all_doc_results)}",
        f"Total Candidate Spans : {total_spans_global}",
        f"Total Local (Tier 1)  : {total_local} ({total_local/max(1, total_spans_global)*100:.2f}%)",
        f"Total Escalated       : {total_escalated_global} ({total_escalated_global/max(1, total_spans_global)*100:.2f}%)",
        f"  -> BERT!=JEV (Tier 3) : {total_esc_disagree_global} ({total_esc_disagree_global/max(1, total_spans_global)*100:.2f}%)",
        f"  -> High Unc  (Tier 2) : {total_esc_uncertain_global} ({total_esc_uncertain_global/max(1, total_spans_global)*100:.2f}%)",
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
        "3. JEV Uncertainty-Gated ACE Generator (BERT + 3-Tier Gating + ACE Generator):",
        f"   - Precision : {global_final['precision']:.4f} ({global_final['tp']}/{global_final['tp']+global_final['fp']})",
        f"   - Recall    : {global_final['recall']:.4f} ({global_final['tp']}/{global_final['tp']+global_final['fn']})",
        f"   - Micro-F1  : {global_final['f1']:.4f}",
        "",
        "4. Delta (Unc-Gated ACE Generator vs BERT Baseline):",
        f"   - Precision Delta : {global_final['precision'] - global_bert['precision']:+.4f}",
        f"   - Recall Delta    : {global_final['recall'] - global_bert['recall']:+.4f}",
        f"   - Micro-F1 Delta  : {global_final['f1'] - global_bert['f1']:+.4f}",
        "",
        "5. Delta (Unc-Gated ACE Generator vs Original JEV-Gated ACE Generator):",
        f"   (Compare with reports/jev_gated_ace_v3/full_summary.txt for the original 2-tier results)",
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
        "CONFIDENT LOCAL SPANS (BERT == JEV, LOW UNCERTAINTY):",
        "-" * 80,
        f"- GT_O_BERT_X          : {all_triads.get('GT_O_BERT_X', 0)} spans ({all_triads.get('GT_O_BERT_X', 0)/max(1, total_spans_global)*100:.2f}%)",
        f"- GT_X_BERT_X          : {all_triads.get('GT_X_BERT_X', 0)} spans ({all_triads.get('GT_X_BERT_X', 0)/max(1, total_spans_global)*100:.2f}%)",
        f"- GT_X_BERT_Y          : {all_triads.get('GT_X_BERT_Y', 0)} spans ({all_triads.get('GT_X_BERT_Y', 0)/max(1, total_spans_global)*100:.2f}%)",
        "",
        "=" * 80,
        "PER-CLASS PERFORMANCE BREAKDOWN (JEV UNCERTAINTY-GATED ACE GENERATOR):",
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

    # Compute complete multi-formulation benchmarks (Phase 5 format)
    try:
        from scripts.calculate_metrics import calculate_and_write_summary
        full_report_str, _ = calculate_and_write_summary(
            report_dir=output_dir,
            system_name=f"SpanNER + JEV Uncertainty-Gated ACE Gen (Thresh={UNCERTAINTY_THRESHOLD})"
        )
        print("\n" + full_report_str)
    except Exception as e:
        logger.warning(f"Could not run calculate_and_write_summary: {e}")
        print("\n" + summary_text)

    logger.info(f"All document reports and full summary written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

