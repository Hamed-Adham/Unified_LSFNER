import os
import json
import datetime
from src.common.config import FORMATS
from src.architectures.agentic_guidelines.generator import Generator
from src.architectures.agentic_guidelines.reflector import Reflector
from src.architectures.agentic_guidelines.curator import Curator
from src.architectures.agentic_guidelines.manager import DynamicGuidelinesManager
from src.architectures.agentic_guidelines.combined_agent import ReflectorCuratorCombined


class LSF_NER_Pipeline:
    def __init__(
        self,
        model_name: str,
        backend: str = "lmstudio",
        dynamicGuidelines_path: str = None,
        file_format: str = "BIO",
        use_combined_reflector_curator: bool = False,
    ):
        """
        Args:
            model_name: LLM model identifier.
            backend: 'lmstudio', 'ollama', or 'api' (passed to llm_client).
            dynamicGuidelines_path: Path to dynamicGuidelines JSON (managed by DynamicGuidelinesManager).
            file_format: 'BIO' or 'BRAT' (key into FORMATS).
            use_combined_reflector_curator: If True, uses a single LLM call that
                combines reflector + curator logic; if False, uses the original
                two-step Reflector + Curator pipeline.
        """
        self.generator = Generator(model_name, FORMATS[file_format], backend)
        self.reflector = Reflector(model_name, FORMATS[file_format], backend)
        self.curator = Curator(model_name, FORMATS[file_format], backend)
        self.combined_reflector_curator = ReflectorCuratorCombined(
            model_name, FORMATS[file_format], backend
        )
        self.use_combined_reflector_curator = use_combined_reflector_curator
        self.file_format = FORMATS[file_format]
        self.reflections = []
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0

        # Initialize DynamicGuidelinesManager
        self.dynamicGuidelines_manager = DynamicGuidelinesManager(dynamicGuidelines_file_path=dynamicGuidelines_path)
        self.dynamicGuidelines_path = self.dynamicGuidelines_manager.dynamicGuidelines_file_path

    def _get_dynamicGuidelines_for_llm(self):
        """
        Get dynamicGuidelines in format suitable for LLM prompts (includes bullet_ids)
        Returns a flat structure with bullet_ids visible for generator/reflector
        """
        return self.dynamicGuidelines_manager.get_dynamicGuidelines_for_generator()

    def process_abstract(self, abstract: str, ground_truth: str = None, 
                        save_outputs_dir: str = None, output_prefix: str = None, file_name=None):
        """
            Train mode → if ground_truth is provided (creates or updates dynamicGuidelines)
            Test mode → if ground_truth is None (uses existing dynamicGuidelines)
            
            Args:
                abstract: Input abstract text
                ground_truth: Ground truth annotations (optional)
                save_outputs_dir: Directory to save step-by-step outputs (optional)
                output_prefix: Prefix for output filenames (optional, defaults to timestamp)
        """
        import time
        import src.common.llm_client as llm_client
        from src.common.llm_client import last_call_metadata

        # Reset last call metadata and abstract context
        llm_client.current_abstract_name = file_name
        last_call_metadata.update({"backend": None, "provider": None, "model": None})
        
        file_prompt_tokens = 0
        file_completion_tokens = 0
        
        # Metrics to format summary exactly as requested
        gen_time, gen_in, gen_out, gen_total = 0.0, 0, 0, 0
        ref_time, ref_in, ref_out, ref_total = 0.0, 0, 0, 0
        cur_time, cur_in, cur_out, cur_total = 0.0, 0, 0, 0
        
        # Set up output paths if save_outputs_dir is provided
        generator_output_path = None
        reflector_output_path = None
        curator_output_path = None
        combined_output_path = None

        if save_outputs_dir:
            os.makedirs(save_outputs_dir, exist_ok=True)
            if output_prefix is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_prefix = f"output_{timestamp}"

            generator_output_path = os.path.join(
                save_outputs_dir, f"{output_prefix}_generator.txt"
            )
            if ground_truth:
                if self.use_combined_reflector_curator:
                    combined_output_path = os.path.join(
                        save_outputs_dir, f"{output_prefix}_reflector_curator_combined.txt"
                    )
                else:
                    reflector_output_path = os.path.join(
                        save_outputs_dir, f"{output_prefix}_reflector.txt"
                    )
                    curator_output_path = os.path.join(
                        save_outputs_dir, f"{output_prefix}_curator.txt"
                    )
        
        # Get dynamicGuidelines in format for generator
        dynamicGuidelines_for_llm = self._get_dynamicGuidelines_for_llm()
        dynamicGuidelines_for_llm_md = self.dynamicGuidelines_manager.get_dynamicGuidelines_for_generator_md()
        
        t0 = time.time()
        predicted = self.generator.run(
            abstract,
            dynamicGuidelines=dynamicGuidelines_for_llm_md,
            save_output_path=generator_output_path
        )
        gen_time = time.time() - t0
        gen_in = last_call_metadata.get("prompt_tokens", 0)
        gen_out = last_call_metadata.get("completion_tokens", 0)
        gen_total = last_call_metadata.get("total_tokens", 0)
        file_prompt_tokens += gen_in
        file_completion_tokens += gen_out
        
        # Count detected entities
        entities_count = 0
        final_ans = predicted.get("final_answer", "")
        if final_ans:
            for line in final_ans.split("\n"):
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[-1] != "O":
                    entities_count += 1
                    
        # Log Generator step
        print(f"   ⚡ [Generator] model: {last_call_metadata['model']} ({last_call_metadata['provider']}) | time: {gen_time:.2f}s | tokens: {last_call_metadata['total_tokens']} (in: {last_call_metadata['prompt_tokens']}, out: {last_call_metadata['completion_tokens']}) | entities found: {entities_count}")
        
        # Track usage of bullet_ids from the prediction
        bullet_ids_used = predicted.get("bullet_ids", [])
        if bullet_ids_used:
            usage_result = self.dynamicGuidelines_manager.increment_usage_counts(bullet_ids_used)
        
        # Run reflection/curation only if ground truth exists
        if ground_truth:
            added, modified, rejected = 0, 0, 0
            if self.use_combined_reflector_curator:
                # Combined mode needs it beforehand. We can use abstract as the query_text.
                dynamicGuidelines_for_curator_md = (
                    self.dynamicGuidelines_manager.get_dynamicGuidelines_for_curator_md(query_text=abstract)
                )
                t0 = time.time()
                # Single LLM call: reflection + curator operations
                reflection = self.combined_reflector_curator.run(
                    abstract=abstract,
                    predicted=predicted,
                    ground_truth=ground_truth,
                    dynamicGuidelines_for_llm=dynamicGuidelines_for_llm_md,
                    dynamicGuidelines_for_curator=dynamicGuidelines_for_curator_md,
                    save_output_path=combined_output_path,
                )
                ref_time = time.time() - t0
                ref_in = last_call_metadata.get("prompt_tokens", 0)
                ref_out = last_call_metadata.get("completion_tokens", 0)
                ref_total = last_call_metadata.get("total_tokens", 0)
                file_prompt_tokens += ref_in
                file_completion_tokens += ref_out
                updates = {"operations": reflection.get("operations", [])}
                print(f"   🧠 [Combined Reflector+Curator] model: {last_call_metadata['model']} ({last_call_metadata['provider']}) | time: {ref_time:.2f}s | tokens: {last_call_metadata['total_tokens']} (in: {last_call_metadata['prompt_tokens']}, out: {last_call_metadata['completion_tokens']})")
            else:
                # Original 2-step pipeline: Reflector then Curator
                t0 = time.time()
                reflection = self.reflector.run(
                    abstract,
                    predicted,
                    ground_truth,
                    dynamicGuidelines_for_llm,
                    save_output_path=reflector_output_path,
                )
                ref_time = time.time() - t0
                ref_in = last_call_metadata.get("prompt_tokens", 0)
                ref_out = last_call_metadata.get("completion_tokens", 0)
                ref_total = last_call_metadata.get("total_tokens", 0)
                file_prompt_tokens += ref_in
                file_completion_tokens += ref_out
                print(f"   🧠 [Reflector] model: {last_call_metadata['model']} ({last_call_metadata['provider']}) | time: {ref_time:.2f}s | tokens: {last_call_metadata['total_tokens']} (in: {last_call_metadata['prompt_tokens']}, out: {last_call_metadata['completion_tokens']})")
                
                # Fetch optimized dynamic guidelines for Curator using key_insight/root cause from reflection
                query_text = reflection.get("key_insight") or reflection.get("root_cause_analysis") or abstract
                dynamicGuidelines_for_curator_md = (
                    self.dynamicGuidelines_manager.get_dynamicGuidelines_for_curator_md(query_text=query_text)
                )
                
                # Run curator to get new operations
                t0 = time.time()
                updates = self.curator.run(
                    dynamicGuidelines_for_curator_md,
                    reflection,
                    save_output_path=curator_output_path,
                )
                cur_time = time.time() - t0
                cur_in = last_call_metadata.get("prompt_tokens", 0)
                cur_out = last_call_metadata.get("completion_tokens", 0)
                cur_total = last_call_metadata.get("total_tokens", 0)
                file_prompt_tokens += cur_in
                file_completion_tokens += cur_out
                print(f"   📝 [Curator] model: {last_call_metadata['model']} ({last_call_metadata['provider']}) | time: {cur_time:.2f}s | tokens: {last_call_metadata['total_tokens']} (in: {last_call_metadata['prompt_tokens']}, out: {last_call_metadata['completion_tokens']})")

            self.reflections.append(reflection)

            # Process bullet tags from reflection (applies to both modes)
            bullet_tags = reflection.get("bullet_tags", [])
            for tag_item in bullet_tags:
                bullet_id = tag_item.get("id") or tag_item.get("bullet_id")
                tag = tag_item.get("tag", "").lower()

                if bullet_id and tag in ["helpful", "harmful"]:
                    section_name = self.dynamicGuidelines_manager.get_section_for_bullet_id(bullet_id)
                    if section_name:
                        self.dynamicGuidelines_manager.mark_bullet(
                            section_name, bullet_id, tag
                        )

            # Process curator operations using DynamicGuidelinesManager (both modes)
            if updates.get("operations"):
                results = self.dynamicGuidelines_manager.process_curator_operations(
                    updates, file_name=file_name, abstract=abstract, reflection=reflection
                )
                added = len(results['added'])
                modified = len(results['modified'])
                rejected = len(results['rejected'])
            
            # Print Dynamic Guidelines update report
            total_rules = 0
            for supercat in self.dynamicGuidelines_manager.dynamicGuidelines.get("dynamicGuidelines_sections", {}).values():
                for bullets in supercat.values():
                    total_rules += len(bullets)
            print(f"   📖 [Playbook] updated: +{added} added, ~{modified} modified, -{rejected} rejected | total rules in playbook: {total_rules}")
            if reflection and reflection.get("key_insight"):
                print(f"      💡 Key Insight: {reflection.get('key_insight')}")
        else:
            reflection = None

        # Accumulate file tokens in class attributes
        self.accumulated_prompt_tokens += file_prompt_tokens
        self.accumulated_completion_tokens += file_completion_tokens

        # Log total tokens and time for this abstract in the requested format
        print(f"\nGenerator: Time: {gen_time:.2f}s | Tokens: {gen_total:,} (in: {gen_in:,}, out: {gen_out:,})")
        if ground_truth:
            if self.use_combined_reflector_curator:
                print(f"Combined Reflector+Curator: Time: {ref_time:.2f}s | Tokens: {ref_total:,} (in: {ref_in:,}, out: {ref_out:,})")
            else:
                print(f"Reflector: Time: {ref_time:.2f}s | Tokens: {ref_total:,} (in: {ref_in:,}, out: {ref_out:,})")
                print(f"Curator: Time: {cur_time:.2f}s | Tokens: {cur_total:,} (in: {cur_in:,}, out: {cur_out:,})")
        
        total_time_secs = gen_time + ref_time + cur_time
        mins = int(total_time_secs // 60)
        secs = int(total_time_secs % 60)
        if mins > 0:
            time_str = f"{mins} min {secs} sec"
        else:
            time_str = f"{secs} sec"
            
        print(f"Accumulated info : Total time: {time_str} | Total tokens: {file_prompt_tokens + file_completion_tokens:,} (in: {file_prompt_tokens:,}, out: {file_completion_tokens:,})")

        # Return dynamicGuidelines in the format expected by existing code
        return predicted, reflection, self.dynamicGuidelines_manager.get_dynamicGuidelines()


# Canonical alias for pipeline
AgenticNERPipeline = LSF_NER_Pipeline

