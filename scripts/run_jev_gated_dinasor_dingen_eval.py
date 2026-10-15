"""
JEV-Gated Dinasor-DinGenerator Evaluation Runner
================================================

Evaluates the 2-Step Dinasor-DinGenerator pipeline across all 80 validation abstracts from
data/cached/spanner_160_10_class/400/val using 9router LLM Backend with JEV Gating:
- Gating Rule: Escalate to LLM if BERT Predicted Label != JEV Decision
- Step 1 (Dinasor): Generates contextualized diagnostic hints and error analyses (Dinasor_primary_bob.txt)
- Step 2 (DinGenerator): Resolves candidate entities into final canonical categories based on hints (DinGenerator_v2_bob.txt)
- Caching: Utilizes reports/jev_cache/{doc_id}.json

Generates:
1. Detailed per-abstract reports in `./reports/jev_gated_dinasor_dingen/{doc_id}.txt`
2. Prompt files for the first 3 escalated abstracts (both Step 1 and Step 2 prompts)
3. Comprehensive global summary in `./reports/jev_gated_dinasor_dingen/full_summary.txt`
   with full breakdown across all 10 canonical Triad outcome classes.
"""

import os
import sys
import re
import json
import time
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("jev_dinasor_dingen_eval")

# ==============================================================================
# RUN CONFIGURATION (Editable directly for IDE execution)
# ==============================================================================
DEFAULT_INPUT_DIR = "data/cached/spanner_160_10_class/400/val"
DEFAULT_OUTPUT_DIR = "./reports/jev_gated_dinasor_dingen"
JEV_CACHE_DIR = "./reports/jev_cache"
DINASOR_PROMPT_PATH = "src/prompts/dinasor_dingen/Dinasor_primary_bob.txt"
DINGEN_PROMPT_PATH = "src/prompts/dinasor_dingen/DinGenerator_v2_bob.txt"
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
    """Loads cached JEV decisions or fetches missing spans."""
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


def classify_challenge_type(
    bert_label: str,
    bert_prob: float,
    top2_label: str,
    top2_prob: float,
    margin: float,
    p_o: float,
    novelty: float,
    u_score: float,
    jev_label: str
) -> str:
    """Categorizes the uncertainty challenge into one of 4 canonical escalation types."""
    if jev_label != bert_label and jev_label != "O" and bert_label != "O":
        return "TOP2_COMPETITION"
    elif p_o >= 0.25 or jev_label == "O" or bert_label == "O":
        return "BACKGROUND_BORDERLINE"
    elif novelty >= 0.50:
        return "NOVEL_UNSEEN_TERM"
    elif margin < 0.20:
        return "CATEGORY_AMBIGUITY"
    else:
        return "BOUNDARY_AMBIGUITY"


def build_anchored_abstract(
    abstract: str,
    uncertain_spans: List[Dict[str, Any]],
    confident_spans: List[Dict[str, Any]]
) -> str:
    """Builds abstract with [TARGET: text] for uncertain spans and [ANCHOR: text (label)] for confident spans."""
    events = []
    for c in uncertain_spans:
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
        events.append((end_char, 0, span_len, "]"))
        events.append((start_char, 1, -span_len, "[TARGET: "))

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
        events.append((end_char, 0, span_len, f" ({lbl})]"))
        events.append((start_char, 1, -span_len, "[ANCHOR: "))

    events.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    res = abstract
    for pos, _, _, tag in events:
        res = res[:pos] + tag + res[pos:]
    return res


def format_uncertain_entities_for_dinasor(uncertain_spans: List[Dict[str, Any]]) -> str:
    """Formats candidate mentions with BERT & JEV evidence for Dinasor prompt."""
    lines = []
    for u in uncertain_spans:
        span_text = u["span_text"]
        b_lbl = u["bert_predicted_label"]
        j_lbl = u["jev_predicted_label"]
        competing = [b_lbl, j_lbl] if b_lbl != j_lbl else [b_lbl, "Non_LSF"]
        top_probs = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(u.get("jev_category_probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:3]])
        
        line = (
            f"- Entity: '{span_text}' | Challenge Type: {u['challenge_type']} | "
            f"BERT Predicted: {b_lbl} (Conf: {u['bert_top1_prob']:.2f}, Margin: {u['bert_margin']:.2f}) | "
            f"JEV System One: {j_lbl} (Entity Prob: {u['jev_is_lsf_entity']:.2f}, Unc: {u['jev_uncertainty']:.2f}) | "
            f"Competing: {competing} | JEV Probs: {{{top_probs}}}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_uncertain_entities_for_dingen(uncertain_spans: List[Dict[str, Any]]) -> str:
    """Formats candidate mentions for DinGenerator prompt."""
    lines = []
    for u in uncertain_spans:
        span_text = u["span_text"]
        b_lbl = u["bert_predicted_label"]
        j_lbl = u["jev_predicted_label"]
        competing = [b_lbl, j_lbl] if b_lbl != j_lbl else [b_lbl, "Non_LSF"]
        line = (
            f"- Entity: '{span_text}' | Mode: {u['challenge_type']} | "
            f"BERT Prior: {b_lbl} | JEV Decision: {j_lbl} | "
            f"Allowed / Competing Categories: {competing}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_dinasor_hints_for_dingen(dinasor_output_raw: str) -> str:
    """Formats hints extracted from Dinasor Step 1 response for Step 2 DinGenerator prompt."""
    hints_text = ""
    try:
        # Try JSON parsing
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", dinasor_output_raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(dinasor_output_raw)

        hints_list = data.get("hints", [])
        if isinstance(hints_list, list) and hints_list:
            lines = []
            for idx, h in enumerate(hints_list, 1):
                t_ent = h.get("target_entity", "")
                r_hint = h.get("resolution_hint", "")
                g_app = h.get("guideline_applied", "")
                lines.append(f"- Hint {idx} for '{t_ent}' [{g_app}]: {r_hint}")
            hints_text = "\n".join(lines)
    except Exception:
        pass

    if not hints_text:
        hints_text = dinasor_output_raw.strip()

    return hints_text


def parse_dingen_response(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts reasoning and labeled_entities from DinGenerator Step 2 response."""
    reasoning = ""
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw_text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    labeled_entities = []
    json_str = raw_text
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_block:
        json_str = json_block.group(1)
    else:
        bracket_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if bracket_match:
            json_str = bracket_match.group(1)

    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            labeled_entities = data.get("labeled_entities", data.get("llm_answer", []))
            if not reasoning:
                reasoning = data.get("reasoning", "")
    except Exception:
        matches = re.findall(r'\{[^{}]*"entity"[^{}]*\}', raw_text, re.DOTALL)
        for m in matches:
            try:
                obj = json.loads(m)
                labeled_entities.append(obj)
            except Exception:
                continue

    return reasoning, labeled_entities


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
    dinasor_raw: str,
    dingen_reasoning: str,
    dingen_raw: str,
    out_file: Path
) -> Dict[str, Any]:
    """Writes structured per-document comparison report."""
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
        f"JEV-GATED DINASOR-DINGENERATOR REPORT: Document {doc_id}",
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
        f"Dinasor-DinGen     : P={final_metrics['precision']:.4f} | R={final_metrics['recall']:.4f} | Micro-F1={final_metrics['f1']:.4f} (TP={final_metrics['tp']}, FP={final_metrics['fp']}, FN={final_metrics['fn']})",
        f"F1 Delta vs BERT   : {final_metrics['f1'] - bert_metrics['f1']:+.4f}",
        "",
        "-" * 80,
        "TRIAD OUTCOME BREAKDOWN (THIS DOCUMENT):",
        "-" * 80,
    ]

    for triad_key, count in triad_counts.most_common():
        desc = TRIAD_DESCRIPTIONS.get(triad_key, "Other / Non-canonical combination")
        lines.append(f"  * {triad_key:<20} : {count} ({count/max(1, total_spans)*100:.1f}%) -> {desc}")

    if dinasor_raw:
        lines.extend([
            "",
            "-" * 80,
            "STEP 1: DINASOR DIAGNOSTIC HINTS (RAW RESPONSE):",
            "-" * 80,
            dinasor_raw.strip(),
        ])

    if dingen_reasoning or dingen_raw:
        lines.extend([
            "",
            "-" * 80,
            "STEP 2: DINGENERATOR REASONING & RESOLUTION (RAW RESPONSE):",
            "-" * 80,
            dingen_raw.strip(),
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
        top_probs_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(s.get('jev_category_probabilities', {}).items(), key=lambda x: x[1], reverse=True)[:3]])
        lines.extend([
            f"\n[SPAN {s['global_id']}/{total_spans}] \"{s['span_text']}\" (Offsets: [{s['start_char']}:{s['end_char']}])",
            f"  * Ground Truth (GT)   : {s['gt_label']} (is_entity: {s['gt_is_entity']})",
            f"  * BERT Prior          : {s['bert_label']} [Top1 Prob: {s['bert_top1_prob']:.4f}, Margin: {s['bert_margin']:.4f}]",
            f"  * JEV System One      : {s['jev_predicted_label']} [Entity Prob: {s['jev_is_lsf_entity']:.4f}, Unc: {s['jev_uncertainty']:.4f}, Probs: {{{top_probs_str}}}]",
            f"  * Challenge Type      : {s.get('challenge_type', 'N/A')}",
            f"  * Gating Decision     : {'⚡ ESCALATED TO 2-STEP DINASOR-DINGEN (BERT != JEV)' if s['escalated'] else '✓ RESOLVED LOCALLY (BERT == JEV)'}",
            f"  * DinGen Final Label  : {s['llm_label']}",
            f"  * Final Assigned Label: {s['final_label']}",
            f"  * Triad Outcome Class : {s['triad_class']}",
            f"  * Generator Rationale : {s.get('rationale', 'N/A')}",
            f"  * Status / Correctness: BERT Correct: {s['gt_label'] == s['bert_label']} | Final Correct: {s['gt_label'] == s['final_label']}",
            "-" * 80
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
    dinasor_prompt_path = Path(DINASOR_PROMPT_PATH)
    dingen_prompt_path = Path(DINGEN_PROMPT_PATH)
    guidebook_path = Path(GUIDEBOOK_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING JEV-GATED DINASOR-DINGENERATOR EVALUATION")
    logger.info(f"Input Directory      : {input_dir}")
    logger.info(f"Output Directory     : {output_dir}")
    logger.info(f"JEV Cache Dir        : {cache_dir}")
    logger.info(f"Dinasor Prompt Path  : {dinasor_prompt_path}")
    logger.info(f"DinGen Prompt Path   : {dingen_prompt_path}")
    logger.info(f"DynamicGuidebook     : {guidebook_path}")
    logger.info(f"Gating Policy        : Escalate when BERT predicted_label != JEV predicted_category")
    logger.info("=" * 80)

    # 1. Load Prompts & Guidebook
    with open(dinasor_prompt_path, "r", encoding="utf-8") as f:
        dinasor_prompt_template = f.read()

    with open(dingen_prompt_path, "r", encoding="utf-8") as f:
        dingen_prompt_template = f.read()

    with open(guidebook_path, "r", encoding="utf-8") as f:
        dynamic_guidebook_content = f.read().strip()

    # 2. Initialize LLM
    llm = get_llm_provider(LLM_BACKEND)

    # 3. List validation files
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
            top2_label = normalize_label(spanner_info.get("second_best_label", "O"))
            bert_prob = spanner_info.get("top1_prob", 0.0)
            top2_prob = spanner_info.get("top2_prob", 0.0)
            bert_margin = spanner_info.get("margin", 0.0)
            p_o = spanner_info.get("p_background_o", 0.0)
            u_score = spanner_info.get("uncertainty", 0.0)
            nov_score = spanner_info.get("novelty_score", 0.0)

            jev_res = jev_decisions[s_idx]
            jev_label = normalize_label(jev_res.predicted_category)

            # Core Gating Rule: Escalate if BERT != JEV
            escalate = (bert_label != jev_label)

            challenge_type = classify_challenge_type(
                bert_label=bert_label,
                bert_prob=bert_prob,
                top2_label=top2_label,
                top2_prob=top2_prob,
                margin=bert_margin,
                p_o=p_o,
                novelty=nov_score,
                u_score=u_score,
                jev_label=jev_label
            )

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
                "challenge_type": challenge_type,
                "escalated": escalate,
                "final_label": bert_label,
                "llm_label": "N/A",
                "rationale": "Resolved locally (BERT == JEV agreement)."
            }

            all_spans_list.append(span_item)
            if escalate:
                uncertain_spans.append(span_item)
            else:
                confident_spans.append(span_item)

        escalated_count = len(uncertain_spans)
        dinasor_raw = ""
        dingen_reasoning = ""
        dingen_raw = ""

        if escalated_count > 0:
            logger.info(f"[{idx}/{len(json_files)}] Doc {doc_id}: Escalating {escalated_count}/{len(spans)} spans to 2-step Dinasor-DinGen...")

            anchored_abstract = build_anchored_abstract(abstract_text, uncertain_spans, confident_spans)
            dinasor_entities_str = format_uncertain_entities_for_dinasor(uncertain_spans)

            # -------------------------------------------------------------
            # STEP 1: Dinasor Hinting Prompt
            # -------------------------------------------------------------
            dinasor_prompt = dinasor_prompt_template
            dinasor_prompt = dinasor_prompt.replace("{annotated_abstract}", anchored_abstract)
            dinasor_prompt = dinasor_prompt.replace("{abstract}", anchored_abstract)
            dinasor_prompt = dinasor_prompt.replace("{uncertain_entities_str}", dinasor_entities_str)
            dinasor_prompt = dinasor_prompt.replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', dynamic_guidebook_content)
            dinasor_prompt = dinasor_prompt.replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', dynamic_guidebook_content)
            dinasor_prompt = dinasor_prompt.replace("{dynamicGuidebook}", dynamic_guidebook_content)
            dinasor_prompt = dinasor_prompt.replace("{dynamicGuidelines}", dynamic_guidebook_content)

            # Save Step 1 prompt for first 3 escalated docs
            if saved_prompts_count < 3:
                p_file_1 = output_dir / f"prompt_dinasor_{doc_id}.txt"
                with open(p_file_1, "w", encoding="utf-8") as f:
                    f.write(dinasor_prompt)

            t0 = time.time()
            try:
                dinasor_raw = llm.generate(
                    prompt=dinasor_prompt,
                    temperature=0.0,
                    max_tokens=4096
                )
                logger.info(f"  -> Step 1 (Dinasor) hints received in {time.time() - t0:.2f}s")
            except Exception as e:
                logger.error(f"  -> Step 1 (Dinasor) failed for Doc {doc_id}: {e}")
                dinasor_raw = "{}"

            dinasor_hints_formatted = format_dinasor_hints_for_dingen(dinasor_raw)

            # -------------------------------------------------------------
            # STEP 2: DinGenerator Resolution Prompt
            # -------------------------------------------------------------
            dingen_entities_str = format_uncertain_entities_for_dingen(uncertain_spans)

            dingen_prompt = dingen_prompt_template
            dingen_prompt = dingen_prompt.replace("{abstract}", anchored_abstract)
            dingen_prompt = dingen_prompt.replace("{annotated_abstract}", anchored_abstract)
            dingen_prompt = dingen_prompt.replace("{uncertain_entities_str}", dingen_entities_str)
            dingen_prompt = dingen_prompt.replace("{hints}", dinasor_hints_formatted)

            # Save Step 2 prompt for first 3 escalated docs
            if saved_prompts_count < 3:
                p_file_2 = output_dir / f"prompt_dingen_{doc_id}.txt"
                with open(p_file_2, "w", encoding="utf-8") as f:
                    f.write(dingen_prompt)
                saved_prompts_count += 1
                logger.info(f"  -> Saved filled prompt artifacts [doc {doc_id}]: prompt_dinasor_{doc_id}.txt and prompt_dingen_{doc_id}.txt")

            t1 = time.time()
            try:
                dingen_raw = llm.generate(
                    prompt=dingen_prompt,
                    temperature=0.0,
                    max_tokens=4096
                )
                logger.info(f"  -> Step 2 (DinGenerator) resolution received in {time.time() - t1:.2f}s")
            except Exception as e:
                logger.error(f"  -> Step 2 (DinGenerator) failed for Doc {doc_id}: {e}")
                dingen_raw = "{}"

            dingen_reasoning, labeled_entities = parse_dingen_response(dingen_raw)

            # Map predictions by verbatim span text
            labeled_by_text = {}
            for item in labeled_entities:
                ent_text = item.get("entity", "").strip().lower()
                if ent_text:
                    labeled_by_text[ent_text] = item

            for item in uncertain_spans:
                s_lower = item["span_text"].strip().lower()
                if s_lower in labeled_by_text:
                    res_item = labeled_by_text[s_lower]
                    assigned = normalize_label(res_item.get("label", res_item.get("category", "O")))
                    item["llm_label"] = assigned
                    item["final_label"] = assigned
                    item["rationale"] = res_item.get("rationale", "")
                else:
                    item["llm_label"] = item["bert_label"]
                    item["final_label"] = item["bert_label"]
                    item["rationale"] = "Omitted by DinGenerator; defaulted to BERT prior."
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
            dinasor_raw=dinasor_raw,
            dingen_reasoning=dingen_reasoning,
            dingen_raw=dingen_raw,
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
        "JEV-GATED DINASOR-DINGENERATOR EVALUATION: FULL VALIDATION SUMMARY",
        "=" * 80,
        f"Dataset Split         : {DEFAULT_INPUT_DIR}",
        f"Step 1 Prompt         : {DINASOR_PROMPT_PATH}",
        f"Step 2 Prompt         : {DINGEN_PROMPT_PATH}",
        f"DynamicGuidebook      : {GUIDEBOOK_PATH}",
        f"LLM Backend           : {LLM_BACKEND}",
        f"Gating Rule           : Escalate to 2-Step LLM if BERT Predicted Label != JEV Decision",
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
        "3. JEV-Gated Dinasor-DinGen (BERT + JEV Gating + Dinasor + DinGenerator):",
        f"   - Precision : {global_final['precision']:.4f} ({global_final['tp']}/{global_final['tp']+global_final['fp']})",
        f"   - Recall    : {global_final['recall']:.4f} ({global_final['tp']}/{global_final['tp']+global_final['fn']})",
        f"   - Micro-F1  : {global_final['f1']:.4f}",
        "",
        "4. Delta (JEV-Gated Dinasor-DinGen vs BERT Baseline):",
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
        "PER-CLASS PERFORMANCE BREAKDOWN (JEV-GATED DINASOR-DINGEN):",
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
    logger.info(f"All document reports and full summary written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
