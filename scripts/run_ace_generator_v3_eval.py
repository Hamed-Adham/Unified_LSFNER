"""
Gated ACE Generator v3 Evaluation Runner
========================================

Evaluates ACE_Generator_v3_bob.txt on all 80 validation abstracts from
data/cached/spanner_160_10_class/400/val using 9router with Weighted Gating
(0.80*uncertainty + 0.20*novelty >= 0.40).

Generates:
1. Detailed per-abstract reports in `./reports/ace_generator_v3/{doc_id}.txt`
2. Dedicated prompt files for the first 3 escalated abstracts: `./reports/ace_generator_v3/prompt_{doc_id}.txt`
3. Comprehensive global summary in `./reports/ace_generator_v3/full_summary.txt`
   with full breakdown across the 10 canonical Triad outcome classes.
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
logger = logging.getLogger("ace_gen_eval")

# ==============================================================================
# RUN CONFIGURATION (Editable directly for IDE execution)
# ==============================================================================
DEFAULT_INPUT_DIR = "data/cached/spanner_160_10_class/400/val"
DEFAULT_OUTPUT_DIR = "./reports/ace_generator_v3"
PROMPT_TEMPLATE_PATH = "src/prompts/agentic_guidelines/ACE_Generator_v3_bob.txt"
GUIDEBOOK_PATH = "data/guidelines/dynamicGuidebook.json"
LLM_BACKEND = "9router"                   # 9Router local endpoint from .env
W_UNCERTAINTY = 0.80                      # Uncertainty weight in DinGen/GatingNN
W_NOVELTY = 0.20                          # Novelty weight in DinGen/GatingNN
UNRELIABILITY_THRESHOLD = 0.40            # Threshold above which span escalates to LLM
LIMIT = None                              # None = all 80 files


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


def format_candidate_spans_json(candidate_spans: List[Dict[str, Any]]) -> str:
    """Formats candidate spans with diagnostic BERT evidence into a clean JSON string."""
    evidence_list = []
    for c in candidate_spans:
        evidence_list.append({
            "id": c["id"],
            "entity": c["span_text"],
            "bert_predicted_label": c["bert_predicted_label"],
            "second_best_label": c.get("second_best_label", "O"),
            "uncertainty": round(float(c.get("uncertainty", 0.0)), 4),
            "novelty_score": round(float(c.get("novelty_score", 0.0)), 4),
            "margin": round(float(c.get("margin", 0.0)), 4)
        })
    return json.dumps(evidence_list, indent=2, ensure_ascii=False)


def calculate_metrics_dict(gold_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
    """Calculates TP, FP, FN, Precision, Recall, and Micro-F1."""
    tp = 0
    fp = 0
    fn = 0
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for g, p in zip(gold_labels, pred_labels):
        g_clean = normalize_label(g)
        p_clean = normalize_label(p)

        if g_clean != "O" and p_clean != "O":
            if g_clean == p_clean:
                tp += 1
                class_stats[g_clean]["tp"] += 1
            else:
                fp += 1
                fn += 1
                class_stats[p_clean]["fp"] += 1
                class_stats[g_clean]["fn"] += 1
        elif g_clean == "O" and p_clean != "O":
            fp += 1
            class_stats[p_clean]["fp"] += 1
        elif g_clean != "O" and p_clean == "O":
            fn += 1
            class_stats[g_clean]["fn"] += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "class_stats": dict(class_stats)
    }


def call_ace_generator(
    abstract_text: str,
    uncertain_spans: List[Dict[str, Any]],
    confident_spans: List[Dict[str, Any]],
    dynamic_guidebook_str: str,
    llm_provider: Any,
    prompt_template_str: str
) -> Tuple[Dict[str, Any], str, str]:
    """Builds prompt, calls LLM, and parses structured output."""
    anchored_abstract = build_gated_anchored_abstract(abstract_text, uncertain_spans, confident_spans)
    candidate_spans_str = format_candidate_spans_json(uncertain_spans)

    prompt = prompt_template_str.replace("{abstract}", anchored_abstract)
    prompt = prompt.replace("{candidate_spans_str}", candidate_spans_str)
    prompt = prompt.replace(
        '{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}',
        dynamic_guidebook_str
    )
    prompt = prompt.replace("{dynamicGuidebook}", dynamic_guidebook_str)
    prompt = prompt.replace("{dynamicGuidelines}", dynamic_guidebook_str)

    try:
        raw_response = llm_provider.generate(prompt)
    except Exception as e:
        logger.error(f"Error calling LLM provider: {e}")
        raw_response = "{}"

    cleaned_json_str = raw_response.strip()
    if "```json" in cleaned_json_str:
        cleaned_json_str = cleaned_json_str.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned_json_str:
        cleaned_json_str = cleaned_json_str.split("```", 1)[1].split("```", 1)[0].strip()

    parsed_result = {"reasoning": "", "llm_answer": []}
    try:
        parsed = json.loads(cleaned_json_str)
        if isinstance(parsed, dict):
            parsed_result["reasoning"] = parsed.get("reasoning", "")
            ans = parsed.get("llm_answer", parsed.get("labeled_entities", []))
            if isinstance(ans, list):
                parsed_result["llm_answer"] = ans
    except Exception as e:
        logger.warning(f"json.loads failed ({e}); attempting regex fallback extraction...")
        pattern = r'\{\s*"id":\s*(\d+)[^}]*?"llm_predicted_label":\s*"([^"]+)"(?:[^}]*?"rationale":\s*"([^"]*)")?[^}]*?\}'
        matches = re.findall(pattern, cleaned_json_str)
        for m in matches:
            parsed_result["llm_answer"].append({
                "id": int(m[0]),
                "llm_predicted_label": m[1],
                "bullet_ids": [],
                "rationale": m[2] if len(m) > 2 else "Extracted via fallback parser"
            })

    return parsed_result, prompt, raw_response


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
    llm_labels = [s["final_label"] for s in spans_data]

    bert_metrics = calculate_metrics_dict(gt_labels, bert_labels)
    llm_metrics = calculate_metrics_dict(gt_labels, llm_labels)

    triad_counts = Counter(s["triad_class"] for s in spans_data)

    lines = [
        "=" * 80,
        f"GATED ACE GENERATOR V3 REPORT: Document {doc_id}",
        "=" * 80,
        f"Generated At          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Spans           : {total_spans}",
        f"Confident (Local)     : {confident_count} ({confident_count/max(1, total_spans)*100:.1f}%)",
        f"Escalated (Generator) : {escalated_count} ({escalated_count/max(1, total_spans)*100:.1f}%)",
        "",
        "-" * 80,
        "DOCUMENT PERFORMANCE SUMMARY:",
        "-" * 80,
        f"BERT Baseline      : P={bert_metrics['precision']:.4f} | R={bert_metrics['recall']:.4f} | Micro-F1={bert_metrics['f1']:.4f} (TP={bert_metrics['tp']}, FP={bert_metrics['fp']}, FN={bert_metrics['fn']})",
        f"Gated ACE Generator: P={llm_metrics['precision']:.4f} | R={llm_metrics['recall']:.4f} | Micro-F1={llm_metrics['f1']:.4f} (TP={llm_metrics['tp']}, FP={llm_metrics['fp']}, FN={llm_metrics['fn']})",
        f"F1 Delta           : {llm_metrics['f1'] - bert_metrics['f1']:+.4f}",
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
        lines.extend([
            f"\n[SPAN {s['global_id']}/{total_spans}] \"{s['span_text']}\" (Offsets: [{s['start_char']}:{s['end_char']}])",
            f"  * Ground Truth (GT)   : {s['gt_label']} (is_entity: {s['gt_is_entity']})",
            f"  * BERT Prior          : {s['bert_label']} [Top1 Prob: {s['bert_top1_prob']:.4f}, Margin: {s['bert_margin']:.4f}]",
            f"  * Neural Evidence     : Uncertainty: {s['uncertainty']:.4f} | Novelty: {s['novelty']:.4f} | P(O): {s['p_o']:.4f}",
            f"  * Gating Decision     : {'⚡ ESCALATED TO GENERATOR' if s['escalated'] else '✓ CONFIDENT LOCAL (BERT)'} [Unreliability: {s['unreliability']:.4f}, Thresh: {UNRELIABILITY_THRESHOLD}]",
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
            str(llm_raw_response).strip(),
            "=" * 80,
        ])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")

    return {
        "doc_id": doc_id,
        "total_spans": total_spans,
        "confident_count": confident_count,
        "escalated_count": escalated_count,
        "bert_metrics": bert_metrics,
        "llm_metrics": llm_metrics,
        "triad_counts": triad_counts,
        "spans_data": spans_data
    }


def run_ace_generator_evaluation(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    prompt_path: str = PROMPT_TEMPLATE_PATH,
    guidebook_path: str = GUIDEBOOK_PATH,
    unreliability_threshold: float = UNRELIABILITY_THRESHOLD,
    w_uncertainty: float = W_UNCERTAINTY,
    w_novelty: float = W_NOVELTY,
    limit: Optional[int] = LIMIT
):
    """Main evaluation pipeline for Gated ACE Generator v3 across all validation files."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_path}")

    json_files = sorted([f for f in in_path.glob("*.json") if f.name != "all_candidates.json"])
    if limit and limit > 0:
        json_files = json_files[:limit]

    total_files = len(json_files)
    logger.info(f"Loaded {total_files} document(s) from {in_path}")

    # Load DynamicGuidebook
    guidebook_file = Path(guidebook_path)
    if guidebook_file.exists():
        with open(guidebook_file, "r", encoding="utf-8") as f:
            gb_raw = json.load(f)
        guidebook_list = gb_raw.get("guidebook", gb_raw) if isinstance(gb_raw, dict) else gb_raw
        guidebook_str = json.dumps(guidebook_list, indent=2, ensure_ascii=False)
        logger.info(f"Loaded DynamicGuidebook ({len(guidebook_list)} rules) from {guidebook_file}")
    else:
        guidebook_str = "No dynamicGuidebook available"
        logger.warning(f"Guidebook file not found at {guidebook_file}; proceeding without guidebook.")

    prompt_template_file = Path(prompt_path)
    if not prompt_template_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_template_file}")
    prompt_template_str = prompt_template_file.read_text(encoding="utf-8")

    logger.info(f"Initializing LLM Provider ({LLM_BACKEND})...")
    llm_provider = get_llm_provider(LLM_BACKEND)

    global_gt = []
    global_bert = []
    global_llm = []
    global_triad_counts = Counter()
    total_spans_all = 0
    total_escalated_all = 0
    saved_prompts_count = 0

    t_start_all = time.time()

    for idx, f_path in enumerate(json_files, 1):
        with open(f_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_id = doc_data.get("doc_id", f_path.stem)
        abstract_text = doc_data.get("text", "")
        spans = doc_data.get("spans", [])

        # 1. Gating step using DinGen weighted combination
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
            bert_margin = spanner_info.get("margin", 0.0)
            u_score = spanner_info.get("uncertainty", 0.0)
            nov_score = spanner_info.get("novelty_score", 0.0)
            p_o = spanner_info.get("p_background_o", 0.0)

            # DinGen Weighted formula: unreliability = (w_u * u) + (w_nov * nov)
            unreliability = (w_uncertainty * u_score) + (w_novelty * nov_score)
            escalate = unreliability >= unreliability_threshold

            span_item = {
                "global_id": s_idx,
                "span_text": span_text,
                "start_char": start_char,
                "end_char": end_char,
                "gt_label": gt_label,
                "gt_is_entity": gt_is_entity,
                "bert_label": bert_label,
                "bert_predicted_label": bert_label,
                "second_best_label": top2_label,
                "bert_top1_prob": bert_prob,
                "bert_margin": bert_margin,
                "uncertainty": u_score,
                "novelty_score": nov_score,
                "novelty": nov_score,
                "p_o": p_o,
                "unreliability": unreliability,
                "escalated": escalate,
                "final_label": bert_label,
                "llm_label": "N/A",
                "bullet_ids": [],
                "rationale": "Resolved locally by high-confidence BERT."
            }

            all_spans_list.append(span_item)
            if escalate:
                uncertain_spans.append(span_item)
            else:
                confident_spans.append(span_item)

        total_spans_all += len(all_spans_list)
        total_escalated_all += len(uncertain_spans)

        # 2. Call ACE Generator v3 for uncertain candidate spans
        llm_prompt = None
        llm_raw = None
        reasoning_text = ""

        if uncertain_spans:
            # Re-index uncertain spans sequentially as 1, 2, ..., K for Generator prompt
            for u_idx, u_span in enumerate(uncertain_spans, 1):
                u_span["id"] = u_idx

            logger.info(f"[{idx}/{total_files}] Doc {doc_id}: Escalating {len(uncertain_spans)}/{len(all_spans_list)} spans to ACE Generator v3...")

            parsed_result, llm_prompt, llm_raw = call_ace_generator(
                abstract_text=abstract_text,
                uncertain_spans=uncertain_spans,
                confident_spans=confident_spans,
                dynamic_guidebook_str=guidebook_str,
                llm_provider=llm_provider,
                prompt_template_str=prompt_template_str
            )
            reasoning_text = parsed_result.get("reasoning", "")

            # Save filled prompt for the first 3 documents with LLM calls
            if saved_prompts_count < 3:
                prompt_save_path = out_path / f"prompt_{doc_id}.txt"
                prompt_save_path.write_text(llm_prompt, encoding="utf-8")
                saved_prompts_count += 1
                logger.info(f"Saved filled prompt ({saved_prompts_count}/3) to: {prompt_save_path}")

            # Match LLM predictions to uncertain spans
            llm_answers_map = {}
            for ans in parsed_result.get("llm_answer", []):
                ans_id = ans.get("id")
                if ans_id is not None:
                    try:
                        llm_answers_map[int(ans_id)] = ans
                    except (ValueError, TypeError):
                        pass
                ent_txt = str(ans.get("entity", "")).strip().lower()
                if ent_txt:
                    llm_answers_map[ent_txt] = ans

            for u_span in uncertain_spans:
                u_id = u_span["id"]
                ent_txt = u_span["span_text"].strip().lower()

                matched_ans = llm_answers_map.get(u_id) or llm_answers_map.get(ent_txt) or {}
                raw_pred_lbl = matched_ans.get("llm_predicted_label", matched_ans.get("label", u_span["bert_label"]))
                llm_lbl = normalize_label(raw_pred_lbl)
                bullet_ids = matched_ans.get("bullet_ids", [])
                rationale = matched_ans.get("rationale", "Arbitrated by ACE Generator.")

                u_span["llm_label"] = llm_lbl
                u_span["final_label"] = llm_lbl
                u_span["bullet_ids"] = bullet_ids
                u_span["rationale"] = rationale
        else:
            logger.info(f"[{idx}/{total_files}] Doc {doc_id}: 0/{len(all_spans_list)} escalated. Resolved locally by BERT.")

        # Classify Triad outcome for every candidate span
        for s in all_spans_list:
            triad = classify_10_outcome_str(
                gt_label=s["gt_label"],
                spanner_label=s["bert_label"],
                final_label=s["final_label"],
                escalated=s["escalated"]
            )
            s["triad_class"] = triad

            global_gt.append(s["gt_label"])
            global_bert.append(s["bert_label"])
            global_llm.append(s["final_label"])
            global_triad_counts[triad] += 1

        # Write per-document report
        doc_out_file = out_path / f"{doc_id}.txt"
        generate_doc_report(
            doc_id=doc_id,
            abstract_text=abstract_text,
            spans_data=all_spans_list,
            reasoning=reasoning_text,
            llm_prompt=llm_prompt,
            llm_raw_response=llm_raw,
            out_file=doc_out_file
        )

    t_elapsed = time.time() - t_start_all

    # Global Summary Calculation
    global_bert_metrics = calculate_metrics_dict(global_gt, global_bert)
    global_llm_metrics = calculate_metrics_dict(global_gt, global_llm)

    summary_file = out_path / "full_summary.txt"
    summary_lines = [
        "=" * 80,
        "GATED ACE GENERATOR V3 EVALUATION: FULL VALIDATION SUMMARY",
        "=" * 80,
        f"Dataset Split         : {input_dir}",
        f"Prompt Template       : {prompt_path}",
        f"DynamicGuidebook      : {guidebook_path}",
        f"LLM Backend           : {LLM_BACKEND}",
        f"Gating Mode           : Weighted (0.80*unc + 0.20*nov >= {unreliability_threshold})",
        f"Execution Time        : {t_elapsed:.2f}s (Average: {t_elapsed/max(1, total_files):.2f}s/doc)",
        f"Total Documents       : {total_files}",
        f"Total Candidate Spans : {total_spans_all}",
        f"Total Escalated Spans : {total_escalated_all} ({total_escalated_all/max(1, total_spans_all)*100:.2f}%)",
        f"Total Local (BERT)    : {total_spans_all - total_escalated_all} ({(total_spans_all - total_escalated_all)/max(1, total_spans_all)*100:.2f}%)",
        "",
        "=" * 80,
        "GLOBAL PERFORMANCE COMPARISON (MICRO-AVERAGED):",
        "=" * 80,
        f"1. BERT Alone (Baseline):",
        f"   - Precision : {global_bert_metrics['precision']:.4f} ({global_bert_metrics['tp']}/{global_bert_metrics['tp'] + global_bert_metrics['fp']})",
        f"   - Recall    : {global_bert_metrics['recall']:.4f} ({global_bert_metrics['tp']}/{global_bert_metrics['tp'] + global_bert_metrics['fn']})",
        f"   - Micro-F1  : {global_bert_metrics['f1']:.4f}",
        "",
        f"2. Gated ACE Generator v3 (BERT + Weighted Gating + ACE Generator):",
        f"   - Precision : {global_llm_metrics['precision']:.4f} ({global_llm_metrics['tp']}/{global_llm_metrics['tp'] + global_llm_metrics['fp']})",
        f"   - Recall    : {global_llm_metrics['recall']:.4f} ({global_llm_metrics['tp']}/{global_llm_metrics['tp'] + global_llm_metrics['fn']})",
        f"   - Micro-F1  : {global_llm_metrics['f1']:.4f}",
        "",
        f"3. Delta (Gated ACE Generator vs BERT Baseline):",
        f"   - Precision Delta : {global_llm_metrics['precision'] - global_bert_metrics['precision']:+.4f}",
        f"   - Recall Delta    : {global_llm_metrics['recall'] - global_bert_metrics['recall']:+.4f}",
        f"   - Micro-F1 Delta  : {global_llm_metrics['f1'] - global_bert_metrics['f1']:+.4f}",
        "",
        "=" * 80,
        "10 CANONICAL TRIAD OUTCOME CLASSES BREAKDOWN:",
        "=" * 80,
    ]

    canonical_triad_order = [
        "GT_X_BERT_Y_LLM_X",
        "GT_O_BERT_X_LLM_O",
        "GT_X_BERT_X_LLM_X",
        "GT_X_BERT_X_LLM_Y",
        "GT_O_BERT_X_LLM_X",
        "GT_O_BERT_X_LLM_Y",
        "GT_X_BERT_X_LLM_O",
        "GT_X_BERT_Y_LLM_O",
        "GT_X_BERT_Y_LLM_Z",
        "GT_X_BERT_Y_LLM_Y",
    ]

    for t_key in canonical_triad_order:
        cnt = global_triad_counts.get(t_key, 0)
        pct = (cnt / max(1, total_spans_all)) * 100
        desc = TRIAD_DESCRIPTIONS.get(t_key, "")
        summary_lines.append(f"\n- {t_key} ({cnt} spans | {pct:.2f}%):")
        summary_lines.append(f"  {desc}")

    # Record any un-escalated baseline classes
    other_triads = {k: v for k, v in global_triad_counts.items() if k not in canonical_triad_order}
    if other_triads:
        summary_lines.append("\n" + "-" * 80)
        summary_lines.append("CONFIDENT LOCAL SPANS (NOT ESCALATED):")
        summary_lines.append("-" * 80)
        for k, v in sorted(other_triads.items()):
            summary_lines.append(f"- {k:<20} : {v} spans ({(v/max(1, total_spans_all))*100:.2f}%)")

    summary_lines.extend([
        "",
        "=" * 80,
        "PER-CLASS PERFORMANCE BREAKDOWN (GATED ACE GENERATOR V3):",
        "=" * 80,
        f"{'Category':<50} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}",
        "-" * 98,
    ])

    all_categories = sorted(list(set(CANONICAL_10_LABELS) - {"O"}))
    for cat in all_categories:
        stats = global_llm_metrics["class_stats"].get(cat, {"tp": 0, "fp": 0, "fn": 0})
        c_tp = stats["tp"]
        c_fp = stats["fp"]
        c_fn = stats["fn"]
        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
        summary_lines.append(
            f"{cat:<50} {c_tp:<6} {c_fp:<6} {c_fn:<6} {c_p:<10.4f} {c_r:<10.4f} {c_f1:<10.4f}"
        )

    summary_lines.append("=" * 80)
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(f"All {total_files} document reports and full summary written to {out_path.resolve()}")
    print("\n" + "\n".join(summary_lines[:50]))


if __name__ == "__main__":
    run_ace_generator_evaluation()
