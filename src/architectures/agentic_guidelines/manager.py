import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from src.architectures.agentic_guidelines.schema_store import GuidebookStore, SchemaStore
from src.architectures.agentic_guidelines.vector_store import VectorStore
from src.architectures.agentic_guidelines.bullet_ops import BulletOps
from src.architectures.agentic_guidelines.formatter import Formatter

class DynamicGuidebookManager:
    """
    Central manager for the Dynamic Guidebook system:
    - Persistent flat JSON storage (data/guidelines/dynamicGuidebook.json)
    - ChromaDB Composite Triad vector synchronization
    - Dual view projections (blind for Dinasor, diagnostic for Generator ACE)
    - Lifecycle metrics & versioned history tracking
    """
    def __init__(
        self,
        guidebook_file_path: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
        dynamicGuidelines_file_path: Optional[str] = None  # legacy argument alias
    ):
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        
        target_path = guidebook_file_path or dynamicGuidelines_file_path
        if target_path:
            self._live_file_path = str(target_path)
        else:
            canonical_path = PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidebook.json"
            legacy_path = PROJECT_ROOT / "output" / "bertized_ace" / "dynamicGuidelines.json"
            if canonical_path.exists() or not legacy_path.exists():
                self._live_file_path = str(canonical_path)
            else:
                self._live_file_path = str(legacy_path)

        self._live_chroma_dir = chroma_persist_directory or str(PROJECT_ROOT / "data" / "guidelines" / "chroma_db")

        # Initialize submodules
        self.store = GuidebookStore(self._live_file_path)
        self.vectors = VectorStore(self._live_chroma_dir)
        self.ops = BulletOps(self.store, self.vectors)
        self.fmt = Formatter(self.vectors)

        # Sync store and vector database
        self.store.load()
        self.vectors.sync_with_store(self.store)

    # --- Core Dynamic Guidebook API ---
    def get_guidelines_for_dinasor(
        self,
        query_text: Optional[str] = None,
        n_results: int = 3,
        as_json_str: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Retrieves guidelines projected for the Dinasor prompt (hiding ground truth, triads, and metrics).
        """
        fmt_name = "dinasor_json" if as_json_str else "dinasor"
        return self.fmt.export(self.store, format=fmt_name, query_text=query_text, n_similar=n_results)

    def get_guidelines_for_generator(
        self,
        query_text: Optional[str] = None,
        n_results: int = 3,
        as_json_str: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Retrieves guidelines projected for Generator ACE (including triad, bert_info, similar_spans).
        """
        fmt_name = "generator_json" if as_json_str else "generator"
        return self.fmt.export(self.store, format=fmt_name, query_text=query_text, n_similar=n_results)

    def get_guidelines_for_curator(
        self,
        query_text: Optional[str] = None,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves complete guideline records for Curator reflection & maintenance.
        """
        return self.fmt._get_selected_guidelines(self.store, query_text=query_text, n_results=n_results)

    # DynamicGuidebook alias methods
    get_guidebook_for_dinasor = get_guidelines_for_dinasor
    get_guidebook_for_generator = get_guidelines_for_generator
    get_guidebook_for_curator = get_guidelines_for_curator

    def add_guideline(self, guideline_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Adds a new guideline conforming to the rich schema."""
        return self.ops.add_guideline(guideline_dict)

    def modify_guideline(
        self,
        guideline_id: str,
        new_rule_text: str,
        reason: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Modifies rule text of an existing guideline."""
        return self.ops.modify_guideline(guideline_id, new_rule_text, reason=reason, file_name=file_name)

    def update_metrics(
        self,
        guideline_ids: Union[str, List[str]],
        *,
        helpful: int = 0,
        harmful: int = 0,
        neutral: int = 0,
        usage: int = 0
    ) -> Dict[str, Any]:
        """Updates feedback metrics for one or more guidelines."""
        return self.ops.update_bullet_metrics(
            guideline_ids,
            helpful_delta=helpful,
            harmful_delta=harmful,
            neutral_delta=neutral,
            usage_delta=usage
        )

    # --- Properties & Legacy Compatibility ---
    @property
    def guidebook_file_path(self) -> str:
        return self.store.guidebook_file_path

    @property
    def dynamicGuidelines_file_path(self) -> str:
        return self.store.guidebook_file_path

    @property
    def chroma_persist_directory(self) -> str:
        return self.vectors.persist_directory

    @property
    def guidebook(self) -> Dict[str, Any]:
        return self.store.guidebook

    @property
    def dynamicGuidelines(self) -> Dict[str, Any]:
        return self.store.guidebook

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

    def export_guidelines(self, format: str = "raw", query_text: Optional[str] = None, n_similar: int = 3):
        return self.fmt.export(self.store, format=format, query_text=query_text, n_similar=n_similar)

    export_guidebook = export_guidelines

    def get_dynamicGuidebook(self):
        return self.export_guidelines(format="raw")

    get_dynamicGuidelines = get_dynamicGuidebook

    def get_dynamicGuidebook_for_generator(self):
        return self.export_guidelines(format="generator")

    get_dynamicGuidelines_for_generator = get_dynamicGuidebook_for_generator

    def get_dynamicGuidebook_for_generator_md(self):
        return self.export_guidelines(format="generator_json")

    get_dynamicGuidelines_for_generator_md = get_dynamicGuidebook_for_generator_md

    def get_dynamicGuidebook_for_curator_md(self, query_text=None, n_similar=3):
        return self.export_guidelines(format="md_curator", query_text=query_text, n_similar=n_similar)

    get_dynamicGuidelines_for_curator_md = get_dynamicGuidebook_for_curator_md

    def search_similar_bullets(self, query_text, n_results=5, category=None):
        return self.vectors.find_similar(query_text, category=category, n_results=n_results)

    def update_bullet_metrics(self, bullet_ids, *, helpful_delta=0, harmful_delta=0, neutral_delta=0, usage_delta=0):
        return self.ops.update_bullet_metrics(
            bullet_ids,
            helpful_delta=helpful_delta,
            harmful_delta=harmful_delta,
            neutral_delta=neutral_delta,
            usage_delta=usage_delta
        )

    def increment_usage_counts(self, bullet_ids):
        res = self.ops.update_bullet_metrics(bullet_ids, usage_delta=1)
        return {"success": res["success"], "incremented_count": res["updated_count"]}

    def process_curator_operations(self, curator_output, similarity_threshold=0.8, file_name=None, abstract=None, reflection=None):
        return self.ops.process_curator_operations(
            curator_output=curator_output,
            similarity_threshold=similarity_threshold,
            file_name=file_name,
            abstract=abstract,
            reflection=reflection
        )

    def save_dynamicGuidebook(self, backup: bool = True) -> bool:
        return self.store.save(backup=backup)

    save_dynamicGuidelines = save_dynamicGuidebook

    def reset_to_empty(self, archive_dir: Optional[str] = None) -> str:
        """
        Archives current dynamic guidebook JSON with a timestamp and resets
        the store to a clean empty guidebook state (and syncs ChromaDB).
        Returns the path to the created archive backup.
        """
        from datetime import datetime
        import shutil
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(archive_dir) if archive_dir else Path(self._live_file_path).parent / "archive"
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / f"dynamicGuidebook_pre_400train_{timestamp_str}.json"

        # 1. Archive current file if it exists
        if os.path.exists(self._live_file_path):
            shutil.copy2(self._live_file_path, archive_path)

        # 2. Reset store structure
        self.store.guidebook = {"guidebook": []}
        self.store._guidelines_by_id = {}
        self.store._max_id_num = 0
        self.store.save(backup=False)

        # 3. Clear/Reset ChromaDB collection
        if self.vectors and self.vectors.collection:
            try:
                self.vectors.client.delete_collection("dynamic_guidebook")
                self.vectors.collection = self.vectors.client.create_collection("dynamic_guidebook")
            except Exception:
                pass

        return str(archive_path)


# Backward compatibility alias
DynamicGuidelinesManager = DynamicGuidebookManager
