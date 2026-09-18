import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.architectures.agentic_guidelines.bertized_ace import (
    BertizedGenerator,
    BertizedReflector,
    format_candidate_alignment_text
)

def run_test():
    print("=" * 70)
    print(">> Running Agentic Guidelines Generator & Reflector Test")
    print("=" * 70)

    # 1. Output directory
    results_dir = PROJECT_ROOT / "babili_test" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Model & Backend configuration
    model_name = os.getenv("LLM_MODEL", "Ner-gm3.7m")
    backend = os.getenv("LLM_BACKEND", "api")
    print(f"Backend  : {backend}")
    print(f"Model    : {model_name}")
    print(f"Base URL : {os.getenv('LLM_API_BASE_URL', 'http://localhost:20128/v1')}")
    print("-" * 70)

    # 3. Load input sample (first row from data/cached/spanner_160_10_class/400/train)
    sample_path = PROJECT_ROOT / "data" / "cached" / "spanner_160_10_class" / "400" / "train" / "1916502.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    doc_id = sample.get("doc_id", "1916502")
    abstract_text = sample.get("text", "")
    spans_data = sample.get("spans", [])

    candidate_spans = []
    for idx, s in enumerate(spans_data, 1):
        ev = s.get("spanner_evidence", {})
        gt = s.get("ground_truth", {})
        boundaries = s.get("boundaries", {})
        candidate_spans.append({
            "id": idx,
            "entity": s.get("span_text", ""),
            "span_text": s.get("span_text", ""),
            "boundaries": boundaries,
            "start_char": boundaries.get("start_char"),
            "end_char": boundaries.get("end_char"),
            "predicted_label": ev.get("predicted_label", "O"),
            "second_best_label": ev.get("second_best_label", "O"),
            "uncertainty": ev.get("uncertainty", 0.0),
            "novelty_score": ev.get("novelty_score", 0.0),
            "margin": ev.get("margin", 0.0),
            "gt_label": gt.get("gt_label", "O")
        })

    print(f"Loaded Document ID: {doc_id}")
    print(f"Abstract preview: {abstract_text[:120]}...")
    print(f"Total candidate spans: {len(candidate_spans)}")
    print("-" * 70)

    # 4. Load Dynamic Guidelines (Pure Read-Only: No VectorStore / ChromaDB modifications)
    guidebook_path = PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidebook.json"
    with open(guidebook_path, "r", encoding="utf-8") as f:
        guidebook_data = json.load(f)
    
    # Format generator view (id, span, sentence, bert_label, triad, guideline, etc.)
    raw_guidelines = guidebook_data.get("guidebook", [])
    gen_guidelines_view = []
    for g in raw_guidelines:
        gen_guidelines_view.append({
            "id": g.get("id", ""),
            "span": g.get("span", ""),
            "span_sentence": g.get("span_sentence", ""),
            "bert_label": g.get("bert_label", ""),
            "ground_truth_label": g.get("ground_truth_label", ""),
            "llm_label": g.get("llm_label", ""),
            "triad": g.get("triad", {}),
            "guideline": g.get("guideline", ""),
            "similar_spans": g.get("similar_spans", [])
        })
    guidelines_str = json.dumps(gen_guidelines_view, indent=2, ensure_ascii=False)
    print(f"Loaded Dynamic Guidelines (Read-Only: {len(gen_guidelines_view)} rules, {len(guidelines_str.splitlines())} lines)")

    # 5. Initialize & Run Generator
    print("\n[Step 1] Initializing & Running BertizedGenerator (ACE_Generator_v3)...")
    generator = BertizedGenerator(model_name=model_name, backend=backend)
    
    # Save populated Generator prompt
    from src.architectures.agentic_guidelines.bertized_ace import build_anchored_abstract
    anchored_abstract = build_anchored_abstract(abstract_text, candidate_spans)
    cand_list = []
    for idx, c in enumerate(candidate_spans, 1):
        cid = c.get("id", idx)
        cand_list.append({
            "id": cid,
            "entity": c.get("span_text", c.get("entity", "")),
            "bert_predicted_label": c.get("bert_predicted_label", c.get("predicted_label", "O")),
            "second_best_label": c.get("second_best_label", "O"),
            "uncertainty": round(float(c.get("uncertainty", 0.0)), 4),
            "novelty_score": round(float(c.get("novelty_score", 0.0)), 4),
            "margin": round(float(c.get("margin", 0.0)), 4)
        })
    candidate_spans_str = json.dumps(cand_list, indent=2)
    populated_gen_prompt = generator.template.replace("{abstract}", anchored_abstract) \
                                             .replace("{candidate_spans_str}", candidate_spans_str) \
                                             .replace('{dynamicGuidebook if dynamicGuidebook else (dynamicGuidelines if dynamicGuidelines else "No dynamicGuidebook available")}', guidelines_str) \
                                             .replace('{dynamicGuidelines if dynamicGuidelines else "No dynamicGuidelines available"}', guidelines_str) \
                                             .replace("{dynamicGuidebook}", guidelines_str) \
                                             .replace("{dynamicGuidelines}", guidelines_str)
    
    gen_prompt_path = results_dir / f"{doc_id}_generator_prompt.txt"
    with open(gen_prompt_path, "w", encoding="utf-8") as f:
        f.write(populated_gen_prompt)
    print(f"Saved populated Generator prompt to: {gen_prompt_path.name}")

    gen_output_path = results_dir / f"{doc_id}_generator_raw.txt"
    gen_result = generator.run(
        abstract=abstract_text,
        candidate_spans=candidate_spans,
        dynamicGuidelines=guidelines_str,
        save_output_path=str(gen_output_path)
    )

    # Robust local parse for display if raw text had template formatting
    raw_gen_text = gen_result.get("raw_response", "")
    gen_ans = gen_result.get("llm_answer", gen_result.get("final_answer", []))
    if not gen_ans and raw_gen_text:
        try:
            cleaned = raw_gen_text.replace("{{", "{").replace("}}", "}")
            p_obj = json.loads(cleaned)
            gen_ans = p_obj.get("llm_answer", p_obj.get("final_answer", []))
            gen_result["llm_answer"] = gen_ans
            gen_result["final_answer"] = gen_ans
            if not gen_result.get("reasoning"):
                gen_result["reasoning"] = p_obj.get("reasoning", "")
            if not gen_result.get("bullet_ids"):
                gen_result["bullet_ids"] = p_obj.get("bullet_ids", [])
        except Exception:
            pass

    print("\n--- Generator Output Summary ---")
    print(f"Reasoning length: {len(gen_result.get('reasoning', ''))} chars")
    print(f"Bullet IDs cited: {gen_result.get('bullet_ids', [])}")
    print(f"Classified entities: {len(gen_ans)}")
    for item in gen_ans:
        p_label = item.get("llm_predicted_label", item.get("final_label"))
        print(f"  * [id: {item.get('id', '?')}] [{item.get('entity')}] -> BERT: {item.get('bert_predicted_label')} | LLM: {p_label}")

    # 6. Build Candidate Alignment & Run Reflector
    print("\n[Step 2] Building Candidate Span Alignment & Running BertizedReflector (ACE_Reflector_v3)...")
    alignment_text = format_candidate_alignment_text(
        candidate_spans=candidate_spans,
        generator_output=gen_result,
        use_outcome_class=True
    )
    print("\n--- Candidate Alignment Table ---")
    print(alignment_text)

    reflector = BertizedReflector(model_name=model_name, backend=backend)
    
    # Save populated Reflector prompt
    used_bullets = gen_result.get("bullet_ids", [])
    bullet_ids_str = json.dumps(used_bullets) if used_bullets else "None"
    ref_anchored_abstract = gen_result.get("anchored_abstract", anchored_abstract)
    populated_ref_prompt = reflector.template.replace("{abstract}", ref_anchored_abstract) \
                                             .replace("{alignment_text}", alignment_text) \
                                             .replace('{predicted.get("reasoning", "")}', gen_result.get("reasoning", "")) \
                                             .replace('{predicted.get("bullet_ids", "")}', bullet_ids_str)
    ref_prompt_path = results_dir / f"{doc_id}_reflector_prompt.txt"
    with open(ref_prompt_path, "w", encoding="utf-8") as f:
        f.write(populated_ref_prompt)
    print(f"Saved populated Reflector prompt to: {ref_prompt_path.name}")

    ref_output_path = results_dir / f"{doc_id}_reflector_raw.txt"
    ref_result = reflector.run(
        abstract=abstract_text,
        candidate_spans=candidate_spans,
        generator_output=gen_result,
        dynamicGuidelines=guidelines_str,
        save_output_path=str(ref_output_path),
        alignment_text=alignment_text
    )

    print("\n--- Reflector Output Summary ---")
    if isinstance(ref_result, list):
        print(f"Total Diagnosed Errors: {len(ref_result)}")
        for idx, diag in enumerate(ref_result, 1):
            print(f"\n  [Error #{idx}] Outcome: {diag.get('primary_error_outcome', '')}")
            print(f"    - Error Identification: {diag.get('error_identification', '')}")
            print(f"    - Root Cause: {diag.get('root_cause_analysis', '')}")
            print(f"    - Correct Approach: {diag.get('correct_approach', '')}")
            print(f"    - Key Insight: {diag.get('key_insight', '')}")
            print(f"    - Bullet Tags: {diag.get('bullet_tags', [])}")
    else:
        print(f"Reasoning length: {len(ref_result.get('reasoning', ''))} chars")
        print(f"Primary Error Outcome: {ref_result.get('primary_error_outcome', '')}")
        print(f"Error Identification: {ref_result.get('error_identification', '')}")
        print(f"Root Cause: {ref_result.get('root_cause_analysis', '')}")
        print(f"Correct Approach: {ref_result.get('correct_approach', '')}")
        print(f"Key Insight: {ref_result.get('key_insight', '')}")
        print(f"Bullet Tags: {ref_result.get('bullet_tags', [])}")

    # 7. Save structured results
    final_output = {
        "doc_id": doc_id,
        "abstract": abstract_text,
        "candidate_spans": candidate_spans,
        "generator_result": {
            "reasoning": gen_result.get("reasoning", ""),
            "bullet_ids": gen_result.get("bullet_ids", []),
            "llm_answer": gen_result.get("llm_answer", gen_result.get("final_answer", [])),
            "final_answer": gen_result.get("final_answer", gen_result.get("llm_answer", []))
        },
        "alignment_table": alignment_text,
        "reflector_result": ref_result
    }

    json_out_path = results_dir / f"{doc_id}_pipeline_result.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    align_txt_path = results_dir / f"{doc_id}_alignment_table.txt"
    with open(align_txt_path, "w", encoding="utf-8") as f:
        f.write(alignment_text)

    print("\n" + "=" * 70)
    print(f"[OK] Outputs successfully saved to: {results_dir}")
    print(f"  - {json_out_path.name}")
    print(f"  - {gen_prompt_path.name}")
    print(f"  - {gen_output_path.name}")
    print(f"  - {ref_prompt_path.name}")
    print(f"  - {ref_output_path.name}")
    print(f"  - {align_txt_path.name}")
    print("=" * 70)
    print("=" * 70)

if __name__ == "__main__":
    run_test()
