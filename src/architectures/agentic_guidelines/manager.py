import os
from pathlib import Path
from src.architectures.agentic_guidelines.schema_store import SchemaStore
from src.architectures.agentic_guidelines.vector_store import VectorStore
from src.architectures.agentic_guidelines.bullet_ops import BulletOps
from src.architectures.agentic_guidelines.formatter import Formatter

class DynamicGuidelinesManager:
    DYNAMICGUIDELINES_CONFIG = SchemaStore.DYNAMICGUIDELINES_CONFIG

    def __init__(self, dynamicGuidelines_file_path=None, chroma_persist_directory=None):
        # Default paths match the original logic
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self._live_file_path = dynamicGuidelines_file_path or str(
            PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidelines.json"
        )
        self._live_chroma_dir = chroma_persist_directory or str(base_dir / "output" / "chroma_db")

        # Initialize submodules
        self.store = SchemaStore(self._live_file_path)
        self.vectors = VectorStore(self._live_chroma_dir)
        self.ops = BulletOps(self.store, self.vectors)
        self.fmt = Formatter(self.vectors)

        # Sync vector store with initial JSON contents
        self.store.load()
        self.vectors.sync_with_store(self.store)

    # --- Properties for backward compatibility ---
    @property
    def dynamicGuidelines_file_path(self):
        return self.store.dynamicGuidelines_file_path

    @property
    def chroma_persist_directory(self):
        return self.vectors.persist_directory

    @property
    def dynamicGuidelines(self):
        return self.store.dynamicGuidelines

    @property
    def bullets_added_since_pruning(self):
        return self.ops.bullets_added_since_pruning

    @bullets_added_since_pruning.setter
    def bullets_added_since_pruning(self, val):
        self.ops.bullets_added_since_pruning = val

    @property
    def bullets_before_pruning(self):
        return self.ops.bullets_before_pruning

    @bullets_before_pruning.setter
    def bullets_before_pruning(self, val):
        self.ops.bullets_before_pruning = val

    @property
    def pruning_score_threshold(self):
        return self.ops.pruning_score_threshold

    @pruning_score_threshold.setter
    def pruning_score_threshold(self, val):
        self.ops.pruning_score_threshold = val

    @property
    def processing_history(self):
        return self.ops.processing_history

    # --- Unified Export API & Legacy Getter Methods ---
    def export_guidelines(self, format="raw", query_text=None, n_similar=3):
        return self.fmt.export(self.store, format=format, query_text=query_text, n_similar=n_similar)

    def get_dynamicGuidelines(self):
        return self.export_guidelines(format="raw")

    def get_dynamicGuidelines_for_generator(self):
        return self.export_guidelines(format="flat")

    def get_dynamicGuidelines_for_generator_md(self):
        return self.export_guidelines(format="md_generator")

    def get_dynamicGuidelines_for_curator_md(self, query_text=None, n_similar=3):
        return self.export_guidelines(format="md_curator", query_text=query_text, n_similar=n_similar)

    # --- Similarity check API ---
    def search_similar_bullets(self, query_text, n_results=5, section_name=None):
        canonical_section = self._normalize_section_name(section_name) if section_name else None
        return self.vectors.find_similar(query_text, section_name=canonical_section, n_results=n_results)

    # --- Metrics & CRUD API ---
    def update_bullet_metrics(self, bullet_ids, *, helpful_delta=0, harmful_delta=0, usage_delta=0):
        return self.ops.update_bullet_metrics(bullet_ids, helpful_delta=helpful_delta, harmful_delta=harmful_delta, usage_delta=usage_delta)

    def mark_bullet(self, section_name, bullet_id, mark_type):
        if mark_type not in ["helpful", "harmful"]:
            return {"success": False, "message": "Invalid mark type"}
        helpful_delta = 1 if mark_type == "helpful" else 0
        harmful_delta = 1 if mark_type == "harmful" else 0
        
        res = self.update_bullet_metrics(bullet_id, helpful_delta=helpful_delta, harmful_delta=harmful_delta)
        if res["success"] and res["updated_count"] > 0:
            for bullet, _, _ in self.store.iter_bullets():
                if bullet.get("bullet_id") == bullet_id:
                    return {
                        "success": True,
                        "bullet_id": bullet_id,
                        "helpful": bullet["helpful"],
                        "harmful": bullet["harmful"]
                    }
        return {"success": False, "message": "Bullet ID not found"}

    def increment_usage_counts(self, bullet_ids):
        res = self.update_bullet_metrics(bullet_ids, usage_delta=1)
        return {"success": res["success"], "incremented_count": res["updated_count"]}

    def modify_bullet(self, bullet_id, new_content, file_name=None, reason=None, error_examples=None):
        return self.ops.modify_bullet(bullet_id, new_content, file_name=file_name, reason=reason, error_examples=error_examples)

    # --- Curation Pipeline & Operations API ---
    def process_curator_operations(self, curator_output, similarity_threshold=0.8, file_name=None, abstract=None, reflection=None):
        return self.ops.process_curator_operations(curator_output, similarity_threshold=similarity_threshold, file_name=file_name, abstract=abstract, reflection=reflection)

    def score_based_pruning(self, section_name=None, score_threshold=None):
        return self.ops.score_based_pruning(section_name=section_name, score_threshold=score_threshold)

    def generate_curator_report(self, output_path=None):
        return self.ops.generate_curator_report(output_path=output_path)

    # --- Internal/private adapters for compatibility ---
    def _normalize_section_name(self, name):
        return self.store.resolve_section(name).canonical_name

    def _get_section_info(self, name):
        info = self.store.resolve_section(name)
        return info.supercategory, info.abbr

    def get_section_for_bullet_id(self, bullet_id):
        return self.store.get_section_for_bullet_id(bullet_id)

    def _extract_sentence_examples(self, abstract, reflection, file_name):
        return self.ops.parse_reflection_for_errors(abstract, reflection, file_name)

    def filter_and_cap_error_examples(self, error_examples, max_limit=4):
        return self.ops.filter_and_cap_error_examples(error_examples, max_limit)

    def save_dynamicGuidelines(self, backup=True):
        return self.store.save(backup=backup)
