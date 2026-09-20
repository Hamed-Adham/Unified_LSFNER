import json
import os
from src.common.llm_client import generate_llm_response as call_llm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "history" / "Reflector.txt"
if not FILE_PATH.exists():
    FILE_PATH = PROJECT_ROOT / "src" / "prompts" / "agentic_guidelines" / "Reflector.txt"


class Reflector:
    def __init__(self, model_name, file_format, backend="lmstudio"):
        self.model_name = model_name
        self.backend = backend
        self.file_format = file_format

    def run(self, abstract: str, predicted: dict, ground_truth: str, dynamicGuidelines: dict, save_output_path: str = None) -> dict:
        """
        Analyzes reasoning trace + final output against ground truth to suggest new rules.
        
        Args:
            abstract: Input abstract text
            predicted: Predicted output from generator
            ground_truth: Ground truth annotations
            dynamicGuidelines: dynamicGuidelines dictionary
            save_output_path: Optional path to save raw LLM response
        """
        
        from src.common.error_analysis import generate_eval_txt_content
        
        # Determine predicted_str for alignment comparison
        pred_ans = predicted.get("final_answer", "")
        if isinstance(pred_ans, list):
            # Convert list of tuples/lists to tab-separated BIO string
            predicted_str = "\n".join(f"{item[0]}\t{item[1]}" for item in pred_ans if len(item) >= 2)
        else:
            predicted_str = str(pred_ans)
            
        # Ground truth is expected to be a BIO string or list
        if isinstance(ground_truth, list):
            ground_truth_str = "\n".join(f"{item[0]}\t{item[1]}" for item in ground_truth if len(item) >= 2)
        else:
            ground_truth_str = str(ground_truth)
            
        # Generate the 4-column token alignment text
        alignment_text = generate_eval_txt_content(ground_truth_str, predicted_str)
        
        # Resolve bullet contents for the reflector prompt
        used_bullet_ids = predicted.get("bullet_ids", [])
        resolved_bullets = []
        if isinstance(used_bullet_ids, list):
            for bid in used_bullet_ids:
                found = False
                for section_bullets in dynamicGuidelines.values():
                    for bullet_str in section_bullets:
                        if bullet_str.startswith(f"{bid}:") or bullet_str.split(':')[0].strip() == bid:
                            resolved_bullets.append(bullet_str)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    resolved_bullets.append(bid)
        else:
            resolved_bullets = used_bullet_ids

        # Copy predicted dict and inject resolved bullets
        predicted_for_prompt = dict(predicted)
        predicted_for_prompt["bullet_ids"] = resolved_bullets
        
        # Override predicted variable in the local scope for eval
        predicted = predicted_for_prompt
 
        with open(FILE_PATH, "r") as f:
            prompt = f.read()
        prompt = eval(f"f'''{prompt}'''")

        response = call_llm(prompt, self.model_name, "Reflector", self.backend)
        
        # Save raw response if path is provided
        if save_output_path:
            os.makedirs(os.path.dirname(save_output_path) if os.path.dirname(save_output_path) else ".", exist_ok=True)
            with open(save_output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("REFLECTOR RAW LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF REFLECTOR RESPONSE\n")
                f.write("=" * 80 + "\n")
            print(f"Reflector output saved to: {save_output_path}")
        
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed["raw_response"] = response
            return parsed
        except json.JSONDecodeError:
            return []
