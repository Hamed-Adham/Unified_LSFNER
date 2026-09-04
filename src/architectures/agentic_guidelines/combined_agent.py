import json
import os
from pathlib import Path

from src.common.llm_client import generate_llm_response as call_llm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "reflector_curator_prompt.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "reflector_curator_prompt.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "prompts" / "reflector_curator_prompt.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "data" / "promtps" / "reflector_curator_prompt.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT.parent / "LSF-NER-main" / "data" / "promtps" / "reflector_curator_prompt.txt"


class ReflectorCuratorCombined:
    """
    Single-call combination of Reflector + Curator.

    This module is intended to replace two separate LLM calls:
    - Reflector: analyzes model errors and tags guideline bullets
    - Curator: proposes new guideline operations based on the reflection

    The combined `run` method makes ONE LLM call and returns a JSON object
    that contains both the reflection fields and the curator-style operations.
    """

    def __init__(self, model_name, file_format, backend: str = "lmstudio"):
        self.model_name = model_name
        self.backend = backend
        self.file_format = file_format

    def run(
        self,
        abstract: str,
        predicted: dict,
        ground_truth: str,
        dynamicGuidelines_for_llm: dict,
        dynamicGuidelines_for_curator: dict,
        save_output_path: str = None,
    ) -> dict:
        """
        Runs reflection + curation in a single LLM call.

        Args:
            abstract: Input abstract text.
            predicted: Output from the generator (reasoning, final_answer, bullet_ids).
            ground_truth: Ground truth annotations.
            dynamicGuidelines_for_llm: Flattened dynamicGuidelines representation
                used by the generator / reflector (bullet-id strings).
            dynamicGuidelines_for_curator: Full dynamicGuidelines structure used
                for curator-style operations.
            save_output_path: Optional path to save raw LLM response.

        Returns:
            dict with at least the following keys:
                - reasoning
                - error_identification
                - root_cause_analysis
                - correct_approach
                - key_insight
                - bullet_tags
                - operations
                - raw_response   (always included, even on JSON parse failure)
        """

        # Load and format prompt template (mirrors reflector/curator style)
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = eval(f"f'''{prompt_template}'''")

        response = call_llm(prompt, self.model_name, "ReflectorCuratorCombined", self.backend)

        # Save raw response if requested
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path) if os.path.dirname(save_output_path) else ".", exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("COMBINED REFLECTOR+CURATOR RAW LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF COMBINED RESPONSE\n")
                f.write("=" * 80 + "\n")
            print(f"Combined Reflector+Curator output saved to: {save_output_path}")

        # Parse JSON with robust fallback
        try:
            parsed = json.loads(response)
            parsed["raw_response"] = response
            # Ensure required keys exist with sensible defaults
            parsed.setdefault("reasoning", "")
            parsed.setdefault("error_identification", "")
            parsed.setdefault("root_cause_analysis", "")
            parsed.setdefault("correct_approach", "")
            parsed.setdefault("key_insight", "")
            parsed.setdefault("bullet_tags", [])
            parsed.setdefault("operations", [])
            return parsed
        except json.JSONDecodeError:
            # Fallback: treat entire response as key_insight, everything else empty
            return {
                "reasoning": "",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "",
                "key_insight": response,
                "bullet_tags": [],
                "operations": [],
                "raw_response": response,
            }
