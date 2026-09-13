import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
results_dir = PROJECT_ROOT / "babili_test" / "results"

for align_file in results_dir.glob("*_alignment_table.txt"):
    doc_id = align_file.name.replace("_alignment_table.txt", "")
    print(f"Processing doc {doc_id}...")

    res_path = results_dir / f"{doc_id}_pipeline_result.json"
    if not res_path.exists():
        continue

    with open(res_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    gen_res = res.get("generator_result", {})
    used_bullets = gen_res.get("bullet_ids", [])
    bullet_ids_str = json.dumps(used_bullets) if used_bullets else "None"

    ref_template_path = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "ACE_Reflector_v3_bob.txt"
    with open(ref_template_path, "r", encoding="utf-8") as f:
        ref_template = f.read()

    from src.architectures.agentic_guidelines.bertized_ace import build_anchored_abstract, format_candidate_alignment_text
    abstract_text = res.get("abstract", "")
    candidate_spans = res.get("candidate_spans", [])
    anchored_abstract = gen_res.get("anchored_abstract") or build_anchored_abstract(abstract_text, candidate_spans)
    alignment_text = format_candidate_alignment_text(candidate_spans, gen_res, use_outcome_class=True)

    with open(align_file, "w", encoding="utf-8") as f:
        f.write(alignment_text)

    ref_prompt = ref_template.replace("{abstract}", anchored_abstract) \
                             .replace("{alignment_text}", alignment_text) \
                             .replace('{predicted.get("reasoning", "")}', gen_res.get("reasoning", "")) \
                             .replace('{predicted.get("bullet_ids", "")}', bullet_ids_str)

    ref_prompt_path = results_dir / f"{doc_id}_reflector_prompt.txt"
    with open(ref_prompt_path, "w", encoding="utf-8") as f:
        f.write(ref_prompt)
    print(f"Updated {ref_prompt_path.name}")
