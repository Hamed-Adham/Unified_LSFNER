"""
Jev Gating Runner & Reporter
============================

Executes Jev System One decision evaluations on cached BERT/SpanNER candidate spans
and writes detailed per-document and per-call structured comparison reports into `./jev_reports`.

Outputs clear comparisons between:
1. Ground Truth (GT)
2. BERT / SpanNER prior prediction
3. Jev System One decision (noul probability, category distribution, uncertainty score, LLM escalation)
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.jev_gating.jev_call import (
    call_single_jev_decision,
    extract_sentence_context,
    JevDecisionResult,
    DEFAULT_JEV_MODEL,
    OPENROUTER_DECISIONS_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("jev_runner")


def format_probability_bar(prob: float, width: int = 20) -> str:
    """Renders a simple ASCII probability bar."""
    filled = int(round(prob * width))
    bar = "█" * filled + " " * (width - filled)
    return f"[{bar}] {prob:.4f}"


def generate_document_report(
    doc_data: Dict[str, Any],
    decisions: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Writes a structured text comparison report for a single document."""
    doc_id = doc_data.get("doc_id", "unknown_doc")
    dataset_split = doc_data.get("dataset_split", "N/A")
    abstract_text = doc_data.get("text", "")
    total_spans = len(decisions)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 80,
        f"JEV GATING & CLASSIFICATION REPORT: Document {doc_id}",
        "=" * 80,
        f"Dataset Split : {dataset_split}",
        f"Generated At  : {timestamp}",
        f"Total Spans   : {total_spans}",
        "",
        "-" * 80,
        "DOCUMENT ABSTRACT TEXT:",
        "-" * 80,
        abstract_text.strip(),
        "",
        "=" * 80,
        "CANDIDATE SPAN EVALUATIONS & LABEL COMPARISONS:",
        "=" * 80,
    ]

    for idx, item in enumerate(decisions, start=1):
        span = item["span"]
        decision: JevDecisionResult = item["decision"]
        duration = item.get("latency_seconds", 0.0)

        span_id = span.get("span_id", f"span_{idx}")
        span_text = span.get("span_text", "")
        boundaries = span.get("boundaries", {})
        start_char = boundaries.get("start_char", 0)
        end_char = boundaries.get("end_char", len(span_text))

        # 1. Ground Truth Info
        gt_info = span.get("ground_truth", {})
        gt_label = gt_info.get("gt_label", "O")
        gt_is_entity = gt_info.get("is_entity", gt_label != "O")

        # 2. BERT / SpanNER Info
        spanner_info = span.get("spanner_evidence", {})
        bert_label = spanner_info.get("predicted_label", "Unknown")
        bert_prob = spanner_info.get("top1_prob", 0.0)
        bert_uncertainty = spanner_info.get("uncertainty", 0.0)

        # 3. Local Sentence Context
        sentence_ctx = extract_sentence_context(abstract_text, start_char, end_char)

        # Agreement Indicators
        agree_gt_bert = (bert_label == gt_label)
        agree_gt_jev = (decision.predicted_category == gt_label)
        agree_bert_jev = (decision.predicted_category == bert_label)

        lines.extend([
            f"\n[CANDIDATE SPAN {idx}/{total_spans}]",
            f"  - Span ID       : {span_id}",
            f"  - Mention Text  : \"{span_text}\"",
            f"  - Char Offsets  : [{start_char}:{end_char}] (length: {end_char - start_char})",
            f"  - Local Sentence: \"{sentence_ctx}\"",
            f"  - API Latency   : {duration:.3f}s",
            "",
            "  >>> 3-WAY LABEL COMPARISON:",
            f"  * Ground Truth (GT)  : {gt_label} (is_entity: {gt_is_entity})",
            f"  * BERT / SpanNER     : {bert_label} [Conf: {bert_prob:.4f}, Uncert: {bert_uncertainty:.4f}]",
            f"  * Jev System One     : {decision.predicted_category} [Noul P(LSF): {decision.is_lsf_entity:.4f}]",
            f"  * Status / Matches   : GT==BERT: {agree_gt_bert} | GT==JEV: {agree_gt_jev} | BERT==JEV: {agree_bert_jev}",
            "",
            "  >>> JEV DECISION DETAILS:",
            f"  * is_lsf_entity (Noul) : {decision.is_lsf_entity:.4f} ({decision.is_lsf_entity * 100:.1f}% probability of being LSF)",
            f"  * predicted_category   : {decision.predicted_category}",
            f"  * gating_uncertainty   : {decision.gating_uncertainty:.4f} (Rubric 0=Clear, 1=Moderate, 2=Ambiguous)",
            f"  * should_escalate_to_llm: {'YES (Escalate to LLM Agent)' if decision.should_escalate_to_llm else 'NO (Accept Jev Decision)'}",
            "",
            "  >>> JEV CATEGORY PROBABILITIES DISTRIBUTION:"
        ])

        sorted_cats = sorted(decision.category_probabilities.items(), key=lambda x: x[1], reverse=True)
        for cat_name, prob in sorted_cats:
            marker = " <-- [PREDICTED]" if cat_name == decision.predicted_category else ""
            lines.append(f"    - {cat_name:<50} {format_probability_bar(prob)}{marker}")

        lines.extend([
            "",
            "  >>> RAW JEV API PAYLOAD & ANSWERS:",
            json.dumps(decision.raw_response, indent=4),
            "-" * 80
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved document report: {output_path}")


def collect_input_files(target_path: Path) -> List[Path]:
    """Collects candidate JSON files from a file or directory path."""
    if not target_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {target_path}")

    if target_path.is_file():
        return [target_path]

    # Search for all json files in directory recursively
    json_files = sorted(list(target_path.glob("**/*.json")))
    return json_files


def run_jev_pipeline(
    input_path: str,
    output_dir: str = "./jev_reports",
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    uncertainty_threshold: float = 1.0,
    timeout: float = 15.0,
) -> None:
    """
    Main evaluation pipeline that iterates over candidate documents, calls Jev System One,
    and writes structured reports to the output folder.
    """
    target_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(target_path)
    if limit and limit > 0:
        files = files[:limit]

    total_files = len(files)
    logger.info(f"Found {total_files} document(s) to process from {target_path}")
    logger.info(f"Reports will be saved to: {out_dir.resolve()}")

    total_spans_evaluated = 0
    total_escalations = 0
    gt_jev_matches = 0
    gt_bert_matches = 0

    for f_idx, file_path in enumerate(files, start=1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            continue

        doc_id = doc_data.get("doc_id", file_path.stem)
        abstract_text = doc_data.get("text", "")
        spans = doc_data.get("spans", [])

        logger.info(f"[{f_idx}/{total_files}] Processing Doc {doc_id} with {len(spans)} candidate span(s)...")

        decisions_for_doc: List[Dict[str, Any]] = []

        for s_idx, span in enumerate(spans, start=1):
            span_text = span.get("span_text", "")
            gt_label = span.get("ground_truth", {}).get("gt_label", "O")
            bert_label = span.get("spanner_evidence", {}).get("predicted_label", "Unknown")

            start_t = time.time()
            try:
                decision = call_single_jev_decision(
                    span_data=span,
                    abstract_text=abstract_text,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    uncertainty_escalation_threshold=uncertainty_threshold,
                    timeout=timeout,
                )
                duration = time.time() - start_t

                decisions_for_doc.append({
                    "span": span,
                    "decision": decision,
                    "latency_seconds": duration,
                })

                total_spans_evaluated += 1
                if decision.should_escalate_to_llm:
                    total_escalations += 1
                if decision.predicted_category == gt_label:
                    gt_jev_matches += 1
                if bert_label == gt_label:
                    gt_bert_matches += 1

                print(
                    f"   [Span {s_idx}/{len(spans)}] '{span_text}' | "
                    f"GT: {gt_label} | BERT: {bert_label} | JEV: {decision.predicted_category} "
                    f"(P(LSF)={decision.is_lsf_entity:.2f}, Uncert={decision.gating_uncertainty:.2f}, "
                    f"Escalate={decision.should_escalate_to_llm})"
                )

            except Exception as e:
                logger.error(f"Error calling Jev for span '{span_text}' in Doc {doc_id}: {e}")

        # Write per-document report file
        doc_report_file = out_dir / f"{doc_id}.txt"
        generate_document_report(doc_data, decisions_for_doc, doc_report_file)

    # Write summary report file
    summary_file = out_dir / "summary_report.txt"
    summary_lines = [
        "=" * 80,
        "JEV EVALUATION SUMMARY OVERVIEW",
        "=" * 80,
        f"Total Documents Processed : {total_files}",
        f"Total Candidate Spans     : {total_spans_evaluated}",
        f"Escalated to LLM          : {total_escalations} ({total_escalations / max(1, total_spans_evaluated) * 100:.2f}%)",
        f"GT == BERT Accuracy       : {gt_bert_matches} / {total_spans_evaluated} ({gt_bert_matches / max(1, total_spans_evaluated) * 100:.2f}%)",
        f"GT == JEV Accuracy        : {gt_jev_matches} / {total_spans_evaluated} ({gt_jev_matches / max(1, total_spans_evaluated) * 100:.2f}%)",
        "=" * 80,
    ]
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(f"Summary report saved to: {summary_file}")



# RUN CONFIGURATION (Edit these parameters directly when running in your IDE)
# ==============================================================================
# input_path can be:
#   1) A single cached candidate JSON file (e.g., "data/cached/spanner_160_10_class/200/train/10390945.json")
#   2) A directory of cached candidate JSON files (e.g., "data/cached/spanner_160_10_class/200/train")
DEFAULT_INPUT_PATH = "data/cached/spanner_160_10_class/200/train/41697.json"
DEFAULT_OUTPUT_DIR = "./jev_reports"
DEFAULT_LIMIT = None                      # e.g. 5, 10, or None to process all documents
DEFAULT_UNCERTAINTY_THRESHOLD = 1.0       # Threshold above which span is flagged for LLM escalation
DEFAULT_REQUEST_TIMEOUT = 15.0            # Seconds before timing out per request


def main(
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = DEFAULT_LIMIT,
    model: str = DEFAULT_JEV_MODEL,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
):
    """
    Main entry point. Can be invoked directly with Python parameters or via IDE run button.
    """
    # If command-line arguments are provided, use argparse; otherwise use the direct parameters above
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Run Jev Gating decisions and save structured reports into ./jev_reports."
        )
        parser.add_argument(
            "--input_path", "-i",
            type=str,
            default=input_path,
            help="Path to a candidate JSON file or folder containing cached candidate JSON files."
        )
        parser.add_argument(
            "--output_dir", "-o",
            type=str,
            default=output_dir,
            help="Directory to save per-document structured text reports."
        )
        parser.add_argument(
            "--limit", "-n",
            type=int,
            default=limit,
            help="Maximum number of candidate documents to process."
        )
        parser.add_argument(
            "--model", "-m",
            type=str,
            default=model,
            help="Jev model identifier on OpenRouter Decisions API."
        )
        parser.add_argument(
            "--uncertainty_threshold", "-u",
            type=float,
            default=uncertainty_threshold,
            help="Gating uncertainty score threshold for LLM escalation."
        )
        parser.add_argument(
            "--timeout", "-t",
            type=float,
            default=timeout,
            help="HTTP timeout in seconds per Jev decision request."
        )
        args = parser.parse_args()
        input_path = args.input_path
        output_dir = args.output_dir
        limit = args.limit
        model = args.model
        uncertainty_threshold = args.uncertainty_threshold
        timeout = args.timeout

    run_jev_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        limit=limit,
        model=model,
        uncertainty_threshold=uncertainty_threshold,
        timeout=timeout,
    )


if __name__ == "__main__":
    # You can change parameters right here or in the CONFIGURATION section above:
    main(
        input_path=DEFAULT_INPUT_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        limit=DEFAULT_LIMIT,
        model=DEFAULT_JEV_MODEL,
        uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )

