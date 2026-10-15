"""
Run Sample Example for JEV Gating Pipeline
==========================================

Demonstrates loading a sample abstract with candidate mentions, executing the JEV Gating Pipeline,
and inspecting the structured JSON output (local_remained vs escale_to_llm).
"""

import os
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.jev_gating.pipeline import JevGatingPipeline, run_jev_gating_pipeline

SAMPLE_DOC_ID = "10398936"
CACHE_DIR = PROJECT_ROOT / "reports" / "jev_cache"
CANDIDATES_CACHE_FILE = PROJECT_ROOT / "data" / "cached" / "spanner_160_10_class" / "400" / "val" / f"{SAMPLE_DOC_ID}.json"


def main():
    print("=" * 80)
    print(f"🔬 JEV GATING PIPELINE SAMPLE EXECUTION ON ABSTRACT: {SAMPLE_DOC_ID}")
    print("=" * 80)

    # 1. Load candidate data
    if not CANDIDATES_CACHE_FILE.exists():
        print(f"⚠️ Candidates file not found: {CANDIDATES_CACHE_FILE}")
        return

    with open(CANDIDATES_CACHE_FILE, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    abstract_text = doc_data.get("text", "")
    candidates = doc_data.get("spans", doc_data.get("candidates", []))

    print(f"📄 Abstract length: {len(abstract_text)} characters")
    print(f"🎯 Total candidate spans proposed by BERT/SpanNER: {len(candidates)}")
    print(f"💾 JEV Cache directory: {CACHE_DIR}")

    # 2. Run JEV Gating Pipeline
    pipeline = JevGatingPipeline(cache_dir=CACHE_DIR, max_workers=5)
    result = pipeline.gate_document_candidates(
        doc_id=SAMPLE_DOC_ID,
        abstract_text=abstract_text,
        candidates=candidates
    )

    # 3. Print Summary
    summary = result["summary"]
    print("\n" + "-" * 80)
    print("📊 GATING RESULTS SUMMARY:")
    print("-" * 80)
    print(f"  • Total Spans Evaluated : {summary['total_candidate_spans']}")
    print(f"  • Locally Retained (0 LLM): {summary['local_remained_count']} ({summary['local_retained_pct']}%)")
    print(f"  • Escalated to LLM      : {summary['escalated_to_llm_count']} ({summary['escalation_rate_pct']}%)")
    print(f"  • Gating Rule Applied   : {summary['gating_rule']}")

    # 4. Display Escalated Spans (Detailed)
    print("\n" + "=" * 80)
    print("🚀 ESCALATED SPANS (escale_to_llm) -> Passing to Generator LLM:")
    print("=" * 80)
    for s in result["escale_to_llm"]:
        print(f"  • [ID {s['id']}] \"{s['span_text']}\" (Offsets: [{s['start_char']}:{s['end_char']}])")
        print(f"    - BERT Label : {s['bert_predicted_label']} (Uncertainty: {s['uncertainty']:.2f}, Margin: {s['margin']:.2f})")
        print(f"    - JEV Label  : {s['jev_predicted_label']} (LSF Prob: {s['jev_is_lsf_probability']:.2f}, Gating Unc: {s['jev_gating_uncertainty']:.2f})")
        print(f"    - Reason     : {s['gating_reason']}")
        print()

    # 5. Display sample of locally retained spans
    print("=" * 80)
    print(f"🛡️ LOCALLY RETAINED SPANS (local_remained) -> Confident [{len(result['local_remained'])} spans]:")
    print("=" * 80)
    for s in result["local_remained"][:5]:
        print(f"  • [ID {s['id']}] \"{s['span_text']}\" -> Assigned Label: {s['assigned_label']} (BERT: {s['bert_predicted_label']}, JEV: {s['jev_predicted_label']})")
    if len(result["local_remained"]) > 5:
        print(f"    ... and {len(result['local_remained']) - 5} more confident spans.")

    # 6. Export sample JSON
    sample_output_path = PROJECT_ROOT / "reports" / "sample_jev_gated_output.json"
    sample_output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n✅ Exported full structured output JSON to: {sample_output_path}")


if __name__ == "__main__":
    main()
