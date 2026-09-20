import json
import os
from src.common.llm_client import generate_llm_response as call_llm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "history" / "Curator.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "Curator.txt"

class Curator:
    def __init__(self, model_name, file_format, backend="lmstudio"):
        self.model_name = model_name
        self.backend = backend
        self.file_format = file_format

    def run(self, dynamicGuidelines: dict, reflection: dict, save_output_path: str = None) -> dict:
        """
        Adds new insights/rules to dynamicGuidelines based on reflection.
        
        Args:
            dynamicGuidelines: Current dynamicGuidelines dictionary
            reflection: Reflection output from reflector
            save_output_path: Optional path to save raw LLM response
        """
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            prompt = f.read()
        prompt = eval(f"f'''{prompt}'''")

        response = call_llm(prompt, self.model_name, "Curator", self.backend)
        
        # Save raw response if path is provided
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path) if os.path.dirname(save_output_path) else ".", exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("CURATOR RAW LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF CURATOR RESPONSE\n")
                f.write("=" * 80 + "\n")
            print(f"Curator output saved to: {save_output_path}")
        
        try:
            parsed = json.loads(response)
            parsed["raw_response"] = response
            return parsed
        except json.JSONDecodeError:
            return {"reasoning": response, "operations": [], "raw_response": response}
