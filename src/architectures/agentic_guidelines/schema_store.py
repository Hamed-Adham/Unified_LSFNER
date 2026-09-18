import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator

class GuidebookStore:
    """
    Flat persistent storage and O(1) indexing for Dynamic Guidebook records.
    The store is completely flat and category-agnostic: each guideline record
    is a standalone document self-containing all its fields (span, guideline, triad, labels, etc.).
    """
    def __init__(self, guidebook_file_path: Optional[str] = None):
        if not guidebook_file_path:
            guidebook_file_path = "data/guidelines/dynamicGuidebook.json"
        self.guidebook_file_path = str(guidebook_file_path)
        self.backup_directory = str(Path(self.guidebook_file_path).parent / "backup")
        
        os.makedirs(os.path.dirname(self.guidebook_file_path), exist_ok=True)
        os.makedirs(self.backup_directory, exist_ok=True)
        
        self._max_id_num = 0
        self.guidebook: Dict[str, Any] = {"guidebook": []}
        self._guidelines_by_id: Dict[str, Dict[str, Any]] = {}

    # --- Backward-compatibility properties ---
    @property
    def dynamicGuidelines_file_path(self) -> str:
        return self.guidebook_file_path

    @dynamicGuidelines_file_path.setter
    def dynamicGuidelines_file_path(self, val: str):
        self.guidebook_file_path = str(val)

    @property
    def dg_backup_directory(self) -> str:
        return self.backup_directory

    @property
    def dynamicGuidelines(self) -> Dict[str, Any]:
        return self.guidebook

    @dynamicGuidelines.setter
    def dynamicGuidelines(self, val: Dict[str, Any]):
        self.guidebook = val

    def load(self):
        """Loads guidebook from JSON, supporting flat {"guidebook": [...]}, {"guidelines": [...]}, or raw list."""
        try:
            with open(self.guidebook_file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                if "guidebook" in data:
                    self.guidebook = data
                elif "guidelines" in data:
                    self.guidebook = {"guidebook": data["guidelines"]}
                elif "dynamicGuidelines_sections" in data:
                    # Minimal decoder for legacy nested archive files
                    legacy_bullets = list(self._iter_legacy_dict(data))
                    self.guidebook = {"guidebook": legacy_bullets}
                else:
                    self.guidebook = {"guidebook": []}
            elif isinstance(data, list):
                self.guidebook = {"guidebook": data}
            else:
                self.guidebook = {"guidebook": []}
            print(f"Loaded dynamicGuidebook from {self.guidebook_file_path}")
        except FileNotFoundError:
            print(f"DynamicGuidebook file not found at {self.guidebook_file_path}. Initializing empty store.")
            self.guidebook = {"guidebook": []}
        except Exception as e:
            print(f"Warning: error reading {self.guidebook_file_path}: {e}. Initializing empty store.")
            self.guidebook = {"guidebook": []}

        self._rebuild_index()
        self.save(backup=False)

    def _iter_legacy_dict(self, data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Unpacks legacy nested dictionary into flat items."""
        for supercategory, sections in data.get("dynamicGuidelines_sections", {}).items():
            for section_name, bullets in sections.items():
                for bullet in bullets:
                    if isinstance(bullet, dict):
                        b_copy = dict(bullet)
                        b_copy.setdefault("ground_truth_label", section_name)
                        b_copy.setdefault("id", b_copy.get("bullet_id", ""))
                        b_copy.setdefault("guideline", b_copy.get("content", ""))
                        yield b_copy

    def _rebuild_index(self):
        """Builds in-memory fast index by ID and tracks max ID number."""
        self._guidelines_by_id = {}
        max_num = 0

        records = self.guidebook.get("guidebook") or self.guidebook.get("guidelines") or []
        for g in records:
            if isinstance(g, dict):
                gid = str(g.get("id") or g.get("bullet_id") or "")
                if gid:
                    self._guidelines_by_id[gid] = g
                    if gid.startswith("DG-"):
                        try:
                            num = int(gid.split("-")[1])
                            max_num = max(max_num, num)
                        except (IndexError, ValueError):
                            pass

        self._max_id_num = max_num

    def save(self, backup: bool = True) -> bool:
        """Saves dynamicGuidebook JSON with timestamped backup."""
        if backup and os.path.exists(self.guidebook_file_path):
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"dynamicGuidebook_backup_{ts}.json"
                backup_path = os.path.join(self.backup_directory, backup_name)
                shutil.copy2(self.guidebook_file_path, backup_path)
            except Exception:
                pass

        try:
            with open(self.guidebook_file_path, "w", encoding='utf-8') as f:
                json.dump(self.guidebook, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save dynamicGuidebook: {e}")
            return False

    def next_id(self, prefix: str = "DG") -> str:
        """Generates the next sequential ID (e.g. DG-0001, DG-0002)."""
        self._max_id_num += 1
        return f"{prefix}-{self._max_id_num:04d}"

    def get_by_id(self, gid: str) -> Optional[Dict[str, Any]]:
        """Retrieves a guideline by ID in O(1)."""
        return self._guidelines_by_id.get(gid)

    def add_or_update(self, guideline: Dict[str, Any]) -> str:
        """Adds or updates a guideline in the store."""
        gid = guideline.get("id")
        if not gid:
            gid = self.next_id()
            guideline["id"] = gid

        # Ensure usage_metrics exists with default values
        guideline.setdefault("usage_metrics", {
            "helpful": 0,
            "harmful": 0,
            "neutral": 0,
            "usage_count": 0,
            "modification_count": 0
        })
        guideline.setdefault("history", [])

        records = self.guidebook.setdefault("guidebook", [])
        if gid not in self._guidelines_by_id:
            records.append(guideline)
        else:
            # In-place update
            for i, existing in enumerate(records):
                if existing.get("id") == gid or existing.get("bullet_id") == gid:
                    records[i] = guideline
                    break

        self._guidelines_by_id[gid] = guideline
        return gid

    def iter_guidelines(self) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding every guideline dict in the store."""
        records = self.guidebook.get("guidebook") or self.guidebook.get("guidelines") or []
        for g in records:
            if isinstance(g, dict) and ("guideline" in g or "content" in g):
                yield g

    def iter_bullets(self):
        """Legacy compatibility alias for iter_guidelines."""
        yield from self.iter_guidelines()


# Backward compatibility alias
SchemaStore = GuidebookStore
