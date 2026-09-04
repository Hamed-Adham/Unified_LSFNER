#!/usr/bin/env python3
"""
Universal Evaluation & Benchmarking CLI for Hybrid LinkNER.

Runs the complete SpanNER Baseline vs. Hybrid LinkNER evaluation across:
1. Strict CoNLL (1) - Exact boundary & exact class match (1-to-1)
2. Partial MUC-7 (3) - Overlapping boundary & matching class (1-to-1)
3. Candidate Mention (4) - Mention-level classification accuracy on candidate pool

Supports all configuration combinations:
- Gating Operators: PROB_OR, WEIGHTED, OR, AND, COPULA, HARMONIC
- Uncertainty Metrics: MARGIN (with MCD), MCD, PE, LC, ENN
- Novelty Metrics: LOF, KNN, MAHALANOBIS, NONE
- Prompt Templates: FINAL_v1_with_label (Prompt v1 with BERT evidence), v2_no_label, legacy
- NMS Policy: ON (Longest-Span First) or OFF (preserve all candidate spans)
- Margin Safeguard Veto: Protections against false deletion to background 'O'
"""

import os
import sys
import json
import glob
import time
import argparse
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.config import LinkNERRunConfig, DEFAULT_CONFIG

from src.architectures.hybrid_linkner.pipeline import (
    LinkNERPipeline,
    OpenAILLMProvider,
    OpenRouterLLMProvider,
    OllamaLLMProvider,
    get_default_llm_provider
)
from src.architectures.hybrid_linkner.novelty_detector import build_lof_novelty_detector
from src.common.evaluation_framework import (
    evaluate_doc_4_formulations,
    aggregate_4_formulations,
    format_4_formulations_summary_box,
    calc_prf
)
from src.common.label_mapping import canonicalize_label, to_new_label
from src.data.dataset_converter import parse_ann_file


def resolve_dataset_items(dataset_arg: str, split: str = "val", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads evaluation documents with text and ground-truth annotations from:
    1. A BRAT directory containing .txt and .ann files (e.g. data/splits/Genereted_Subsets/subset_3_size_10)
    2. A processed JSON file (e.g. data/processed/spanner_dataset.json)
    3. Shorthand alias names
    """
    alias_map = {
        "subset_1": "data/splits/Genereted_Subsets/subset_1_size_40",
        "subset_2": "data/splits/Genereted_Subsets/subset_2_size_20",
        "subset_3": "data/splits/Genereted_Subsets/subset_3_size_10",
        "subset_4": "data/splits/Genereted_Subsets/subset_4_size_10",
        "subset_5": "data/splits/Genereted_Subsets/subset_5_size_10",
        "subset_10": "data/splits/Genereted_Subsets/subset_10_size_10",
        "test_folder": "data/Relabaled_test_folder_noDisease",
        "balanced_val": "data/splits/Genereted_Subsets/subset_3_size_10",
    }
    
    target_path = alias_map.get(dataset_arg, dataset_arg)
    if not os.path.isabs(target_path):
        target_path = os.path.join(PROJECT_ROOT, target_path)

    items = []

    # Case 1: Directory with BRAT .txt and .ann files
    if os.path.isdir(target_path):
        txt_files = sorted(glob.glob(os.path.join(target_path, "*.txt")))
        for txt_path in txt_files:
            ann_path = txt_path[:-4] + ".ann"
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            gts = parse_ann_file(ann_path, target_schema="new") if os.path.exists(ann_path) else []
            doc_id = os.path.basename(txt_path).replace(".txt", "")
            items.append({
                "doc_id": doc_id,
                "text": text,
                "ground_truth": gts
            })

    # Case 2: JSON file
    elif os.path.isfile(target_path) and target_path.endswith(".json"):
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and split in data:
            raw_items = data[split]
        elif isinstance(data, list):
            raw_items = data
        else:
            raw_items = []

        for idx, it in enumerate(raw_items):
            text = it.get("text", "")
            if not text and "tokens" in it:
                text = " ".join(it["tokens"])
            gts = it.get("entities", it.get("ground_truth", []))
            norm_gts = []
            for g in gts:
                norm_gts.append({
                    "start_char": g.get("start_char", g.get("start", 0)),
                    "end_char": g.get("end_char", g.get("end", 0)),
                    "label": canonicalize_label(g.get("label", g.get("type", "O")), use_new_labels=True),
                    "span_text": g.get("span_text", g.get("text", ""))
                })
            items.append({
                "doc_id": it.get("doc_id", f"doc_{idx}"),
                "text": text,
                "ground_truth": norm_gts
            })
    else:
        raise FileNotFoundError(f"Could not find valid dataset directory or JSON file at: {target_path}")

    if limit is not None and limit > 0:
        items = items[:limit]

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid LinkNER Evaluation with Customizable Gating, Prompts, NMS, and 3-Formulation Metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset & Run Parameters
    parser.add_argument("--dataset", type=str, default="data/splits/Genereted_Subsets/subset_3_size_10",
                        help="Dataset directory (.txt + .ann) or JSON file path or alias (e.g. subset_3, balanced_val)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split if JSON format (val/test/train)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to evaluate (optional)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for SpanNER & parallel dispatch")

    # Gating Configuration Parameters
    parser.add_argument("--gating", type=str, default="prob_or",
                        choices=["prob_or", "weighted", "or", "and", "copula", "harmonic"],
                        help="Gating aggregation operator")
    parser.add_argument("--w_nov", type=float, default=0.30, help="Novelty weight w_novelty")
    parser.add_argument("--w_unc", type=float, default=0.70, help="Uncertainty weight w_uncertainty")
    parser.add_argument("--unc", type=str, default="margin",
                        choices=["margin", "mcd", "pe", "lc", "enn"],
                        help="Uncertainty metric (margin uses MCD margin distribution when mcd_passes > 1)")
    parser.add_argument("--nov", type=str, default="lof",
                        choices=["lof", "knn", "mahalanobis", "none"],
                        help="Novelty metric")
    parser.add_argument("--tau", "--threshold", dest="threshold", type=float, default=0.35,
                        help="Unreliability escalation threshold Tau")
    parser.add_argument("--margin_veto", type=float, default=0.50,
                        help="Margin veto threshold protecting decisive SpanNER predictions against LLM deletion to 'O'")

    # Prompt & NMS Configuration
    parser.add_argument("--prompt", type=str, default="v1_with_label",
                        choices=["v1_with_label", "v1", "v2_no_label", "v2", "legacy"],
                        help="LLM prompt template version (v1_with_label uses FINAL_v1_with_label.md)")
    parser.add_argument("--nms", dest="use_nms", action="store_true", help="Enable Longest-Span Non-Maximum Suppression")
    parser.add_argument("--no-nms", dest="use_nms", action="store_false", help="Disable NMS (preserve all candidate spans)")
    parser.set_defaults(use_nms=False)

    parser.add_argument("--margin_safeguard", dest="use_margin_safeguard", action="store_true", help="Enable margin safeguard veto")
    parser.add_argument("--no-margin_safeguard", dest="use_margin_safeguard", action="store_false", help="Disable margin safeguard veto")
    parser.set_defaults(use_margin_safeguard=True)

    parser.add_argument("--mcd_passes", type=int, default=5, help="Number of Monte Carlo Dropout stochastic passes")

    # Evaluation Metrics Selection
    parser.add_argument("--metrics", nargs="+", default=["strict", "partial", "mention"],
                        help="Formulation metrics to compute (strict, nested_aware, partial, mention)")
    
    # LLM Provider & Report Options
    parser.add_argument("--provider", type=str, default=None, choices=["openai", "openrouter", "ollama"], help="LLM Provider")
    parser.add_argument("--model", type=str, default=None, help="LLM Model name")
    parser.add_argument("--output_report", type=str, default=None, help="Path to save evaluation summary JSON/Markdown")

    args = parser.parse_args()

    # Construct LinkNERRunConfig
    run_config = LinkNERRunConfig(
        gating_operator=args.gating.upper(),
        w_novelty=args.w_nov,
        w_uncertainty=args.w_unc,
        uncertainty_metric=args.unc.upper(),
        novelty_metric=args.nov.upper(),
        threshold=args.threshold,
        margin_veto_threshold=args.margin_veto,
        use_nms=args.use_nms,
        prompt_template_version=args.prompt,
        use_margin_safeguard=args.use_margin_safeguard,
        mcd_passes=args.mcd_passes,
        evaluation_metrics=args.metrics,
        batch_size=args.batch_size
    )

    print("\n" + "=" * 118)
    print("           ⚡ INITIALIZING HYBRID LINKNER STANDARDIZED EVALUATION BENCHMARK")
    print("=" * 118)
    print(f"  • Dataset Source       : {args.dataset}")
    print(f"  • Gating Operator      : {run_config.gating_operator}")
    print(f"  • Weight Combination   : w_novelty={run_config.w_novelty:.2f}, w_uncertainty={run_config.w_uncertainty:.2f}")
    print(f"  • Uncertainty Metric   : {run_config.uncertainty_metric} (MCD passes={run_config.mcd_passes})")
    print(f"  • Novelty Metric       : {run_config.novelty_metric}")
    print(f"  • Threshold (Tau)      : {run_config.threshold:.2f}")
    print(f"  • Margin Veto Thresh   : {run_config.margin_veto_threshold:.2f} (Safeguard: {run_config.use_margin_safeguard})")
    print(f"  • NMS Policy           : {'ENABLED' if run_config.use_nms else 'DISABLED (Preserves Raw Candidates)'}")
    print(f"  • Prompt Template      : {run_config.prompt_template_version}")
    print(f"  • Active Metrics       : {', '.join(run_config.evaluation_metrics)}")
    print("=" * 118 + "\n")

    # Load Dataset Items
    items = resolve_dataset_items(args.dataset, split=args.split, limit=args.limit)
    print(f"📚 Loaded {len(items)} evaluation documents (Total GT Entities: {sum(len(it['ground_truth']) for it in items)}).")

    # Initialize Novelty Detector
    novelty_scorer = None
    if run_config.novelty_metric != "NONE":
        try:
            novelty_scorer = build_lof_novelty_detector()
            print("🧠 LOF Novelty Scorer initialized successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize LOF Novelty Scorer ({e}). Running without novelty.")
            novelty_scorer = None

    # Initialize LLM Provider
    llm_provider = None
    if args.provider == "openai":
        llm_provider = OpenAILLMProvider(model_name=args.model or "gpt-4o-mini")
    elif args.provider == "openrouter":
        llm_provider = OpenRouterLLMProvider(model_name=args.model or "meta-llama/llama-3.3-70b-instruct")
    elif args.provider == "ollama":
        llm_provider = OllamaLLMProvider(model_name=args.model or "qwen2.5:7b-instruct-q8_0")
    else:
        llm_provider = get_default_llm_provider()

    # Initialize Pipeline
    pipeline = LinkNERPipeline(
        novelty_scorer=novelty_scorer,
        llm_provider=llm_provider,
        config=run_config
    )

    texts = [it["text"] for it in items]

    # Stage A: Run SpanNER Baseline Predictions (Escalation disabled, raw predictions)
    print("\n[Stage A] Running Primary SpanNER Baseline Predictions...")
    spanner_docs_preds = pipeline.predict_batch(
        texts=texts,
        batch_size=run_config.batch_size,
        uncertainty_threshold=1.01,  # Never escalate (100% SpanNER local predictions)
        use_nms=run_config.use_nms,
        verbose=False
    )

    # Stage B: Run Hybrid LinkNER Predictions (With Gating & LLM Arbitration)
    print("[Stage B] Running Hybrid LinkNER (Gating + Single-Pass LLM Arbitration)...")
    start_t = time.time()
    hybrid_docs_preds = pipeline.predict_batch(
        texts=texts,
        batch_size=run_config.batch_size,
        uncertainty_method=run_config.uncertainty_metric.lower(),
        uncertainty_threshold=run_config.threshold,
        novelty_threshold=0.60,
        gating_mode=run_config.gating_operator.lower(),
        w_novelty=run_config.w_novelty,
        w_uncertainty=run_config.w_uncertainty,
        mcd_passes=run_config.mcd_passes,
        use_nms=run_config.use_nms,
        prompt_version=run_config.prompt_template_version,
        margin_threshold=run_config.margin_veto_threshold,
        use_margin_safeguard=run_config.use_margin_safeguard,
        verbose=True
    )
    elapsed = time.time() - start_t

    # Stage C: Calculate Multi-Tier Metrics for Both Runs
    spanner_doc_metrics = []
    hybrid_doc_metrics = []
    
    total_spans_eval = 0
    total_local_accepted = 0
    total_llm_escalated = 0

    for i, it in enumerate(items):
        gts = it["ground_truth"]
        s_preds = spanner_docs_preds[i]
        h_preds = hybrid_docs_preds[i]

        s_doc_m = evaluate_doc_4_formulations(s_preds, gts)
        h_doc_m = evaluate_doc_4_formulations(h_preds, gts)

        spanner_doc_metrics.append(s_doc_m)
        hybrid_doc_metrics.append(h_doc_m)

        for p in h_preds:
            total_spans_eval += 1
            if p.get("escalated_to_llm", False):
                total_llm_escalated += 1
            else:
                total_local_accepted += 1

    spanner_agg = aggregate_4_formulations(spanner_doc_metrics)
    hybrid_agg = aggregate_4_formulations(hybrid_doc_metrics)

    # Local computation percentage
    local_pct = (total_local_accepted / total_spans_eval * 100.0) if total_spans_eval > 0 else 100.0

    # Format the Benchmark Box
    summary_box = format_4_formulations_summary_box(
        spanner_results=spanner_agg,
        hybrid_results=hybrid_agg,
        active_metrics=run_config.evaluation_metrics,
        system_name=f"Hybrid LinkNER ({run_config.prompt_template_version})"
    )

    print("\n" + summary_box)

    # Print Summary Statistics Card
    print(f"\n📊 OPERATIONAL EFFICIENCY & SUMMARY METRICS:")
    print(f"  • Total Documents Evaluated   : {len(items)}")
    print(f"  • Total Span Decisions        : {total_spans_eval}")
    print(f"  • Locally Accepted by SpanNER : {total_local_accepted} ({local_pct:.2f}% local computation saved)")
    print(f"  • Escalated to Generative LLM : {total_llm_escalated} ({100.0 - local_pct:.2f}%)")
    print(f"  • Total Benchmark Runtime     : {elapsed:.2f}s ({elapsed/max(1, len(items)):.2f}s / doc)")

    # Save Output Report if specified
    if args.output_report:
        report_data = {
            "config": run_config.__dict__,
            "spanner_baseline": spanner_agg,
            "hybrid_linkner": hybrid_agg,
            "local_computation_percent": local_pct,
            "total_documents": len(items),
            "total_spans": total_spans_eval,
            "elapsed_seconds": elapsed
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        if args.output_report.endswith(".json"):
            with open(args.output_report, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
        elif args.output_report.endswith(".md"):
            with open(args.output_report, "w", encoding="utf-8") as f:
                f.write(f"# Hybrid LinkNER Evaluation Report\n\n```\n{summary_box}\n```\n")
        print(f"💾 Benchmark report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
