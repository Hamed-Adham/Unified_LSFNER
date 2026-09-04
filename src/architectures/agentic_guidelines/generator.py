import json
import os
from src.common.llm_client import generate_llm_response as call_llm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "Generator.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "Generator.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "prompts" / "Generator.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "data" / "promtps" / "Generator.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT.parent / "LSF-NER-main" / "data" / "promtps" / "Generator.txt"

class Generator:
    def __init__(self, model_name, file_format, backend="lmstudio"):
        self.model_name = model_name
        self.backend = backend
        self.file_format = file_format

    def run(self, abstract: str, dynamicGuidelines: dict = None, save_output_path: str = None) -> dict:
        """
        Produces reasoning + BIO annotations in structured JSON.
        
        Args:
            abstract: Input abstract text
            dynamicGuidelines: dynamicGuidelines dictionary
            save_output_path: Optional path to save raw LLM response
        """
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            prompt = f.read()
        prompt = eval(f"f'''{prompt}'''")
        response = call_llm(prompt, self.model_name, "Generator", self.backend)
        
        # Save raw response if path is provided
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path) if os.path.dirname(save_output_path) else ".", exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("GENERATOR RAW LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF GENERATOR RESPONSE\n")
                f.write("=" * 80 + "\n")
            print(f"Generator output saved to: {save_output_path}")
        
        # Try parsing structured JSON
        try:
            parsed = json.loads(response)
            reasoning = parsed.get("reasoning", "")
            final_answer = parsed.get("final_answer", "")
            if isinstance(final_answer, list):
                lines = []
                for item in final_answer:
                    if isinstance(item, (list, tuple)):
                        lines.append("\t".join(str(subitem) for subitem in item))
                    else:
                        lines.append(str(item))
                final_answer = "\n".join(lines)
            bullets = parsed.get("bullet_ids", [])
        except json.JSONDecodeError:
            # Try to heuristically split if model didn't return JSON
            if "### Final Answer" in response:
                parts = response.split("### Final Answer", 1)
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                reasoning = response.strip()
                final_answer = ""
            bullets = []

        return {"reasoning": reasoning, "bullet_ids": bullets, "final_answer": final_answer, "raw_response": response}
