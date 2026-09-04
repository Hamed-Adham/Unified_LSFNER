import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from collections import namedtuple

# Define SectionInfo namedtuple
SectionInfo = namedtuple('SectionInfo', ['supercategory', 'abbr', 'canonical_name'])

class SchemaStore:
    DYNAMICGUIDELINES_CONFIG = {
        "General NER Sections": {
            "sections": {
                "annotation_fundamentals": "ANF",
                "context_scope_handling": "CSH",
                "entity_relationship_rules": "ERR"
            }
        },
        "LSF categories": {
            "sections": {
                "Nutrition": "NUT",
                "Socioeconomic_factors": "SEF",
                "Environmental_exposures": "ENV",
                "Substance_use": "SUB",
                "Physical_activities": "PA",
                "Non_physical_leisure_time_activities": "NPLA",
                "Personal_care_products_and_cosmetic_procedures": "PCP",
                "Sleep": "SLP",
                "Mental_health_practices": "MHP"
            }
        }
    }

    def __init__(self, dynamicGuidelines_file_path):
        self.dynamicGuidelines_file_path = dynamicGuidelines_file_path
        self.dg_backup_directory = str(Path(self.dynamicGuidelines_file_path).parent / "backup")
        
        os.makedirs(os.path.dirname(self.dynamicGuidelines_file_path), exist_ok=True)
        os.makedirs(self.dg_backup_directory, exist_ok=True)
        
        self._initialize_mappings()
        self._max_ids_cache = {}
        self.dynamicGuidelines = {"dynamicGuidelines_sections": {}}

    def _initialize_mappings(self):
        """Derive flat helper dictionaries from the centralized config."""
        self.section_abbreviations = {}
        self.section_to_supercategory = {}
        self.prefix_to_section = {}  # Reverse lookup: prefix -> section_name
        for supercategory, data in self.DYNAMICGUIDELINES_CONFIG.items():
            for section_name, abbr in data["sections"].items():
                self.section_abbreviations[section_name] = abbr
                self.section_to_supercategory[section_name] = supercategory
                self.prefix_to_section[abbr] = section_name  # e.g., "NUT" -> "Nutrition"

    def resolve_section(self, name) -> SectionInfo:
        if not name:
            return SectionInfo(None, None, None)
        canonical_name = name.strip()
        upper_name = canonical_name.upper()
        if upper_name in self.prefix_to_section:
            canonical_name = self.prefix_to_section[upper_name]
        
        supercategory = self.section_to_supercategory.get(canonical_name)
        abbr = self.section_abbreviations.get(canonical_name)
        return SectionInfo(supercategory, abbr, canonical_name)

    def get_section_for_bullet_id(self, bullet_id):
        """O(1) lookup of section name from bullet ID prefix."""
        prefix = bullet_id.split("-")[0] if "-" in bullet_id else None
        return self.prefix_to_section.get(prefix)

    def load(self):
        try:
            with open(self.dynamicGuidelines_file_path, "r", encoding='utf-8') as f:
                self.dynamicGuidelines = json.load(f)
            print(f"Loaded dynamicGuidelines from {self.dynamicGuidelines_file_path}")
        except FileNotFoundError:
            print(f"DynamicGuidelines file not found. Creating a new dynamicGuidelines.")
            self.dynamicGuidelines = {"dynamicGuidelines_sections": {}}
        self._ensure_all_sections_exist()
        self.save(backup=False)

    def _ensure_all_sections_exist(self):
        self.dynamicGuidelines.setdefault("dynamicGuidelines_sections", {})
        for supercategory, data in self.DYNAMICGUIDELINES_CONFIG.items():
            self.dynamicGuidelines["dynamicGuidelines_sections"].setdefault(supercategory, {})
            for section in data["sections"]:
                self.dynamicGuidelines["dynamicGuidelines_sections"][supercategory].setdefault(section, [])
                for bullet in self.dynamicGuidelines["dynamicGuidelines_sections"][supercategory][section]:
                    if isinstance(bullet, dict):
                        bullet.setdefault("helpful", 0)
                        bullet.setdefault("harmful", 0)
                        bullet.setdefault("usage_count", 0)
                        bullet.setdefault("modification_count", 0)
                        bullet.setdefault("content_history", [])
                        bullet.setdefault("history", [])
                        bullet.setdefault("diagnostic_context", {
                            "created_from_file": "unknown",
                            "root_cause_of_creation": "",
                            "error_examples": []
                        })
                        dc = bullet["diagnostic_context"]
                        if isinstance(dc, dict) and "error_examples" in dc:
                            # Use delayed import to avoid circular dependencies
                            from src.architectures.agentic_guidelines.bullet_ops import BulletOps
                            dc["error_examples"] = BulletOps.filter_and_cap_error_examples_static(dc["error_examples"])

    def save(self, backup=True):
        """Save dynamicGuidelines JSON. If backup=True, write a timestamped copy
        into the backup/ subfolder before overwriting the live file."""
        if backup and os.path.exists(self.dynamicGuidelines_file_path):
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"dynamicGuidelines_backup_{ts}.json"
                backup_path = os.path.join(self.dg_backup_directory, backup_name)
                shutil.copy2(self.dynamicGuidelines_file_path, backup_path)
            except Exception:
                pass  # Never let a backup failure block the main save

        try:
            with open(self.dynamicGuidelines_file_path, "w", encoding='utf-8') as f:
                json.dump(self.dynamicGuidelines, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def iter_bullets(self):
        """Generator yielding (bullet_dict, supercategory, section_name) for every bullet."""
        for supercategory, sections in self.dynamicGuidelines.get("dynamicGuidelines_sections", {}).items():
            for section_name, bullets in sections.items():
                for bullet in bullets:
                    if isinstance(bullet, dict) and "content" in bullet:
                        yield bullet, supercategory, section_name

    def next_id(self, section_abbr):
        """Generate the next sequential ID for a section, using cache for efficiency."""
        if section_abbr not in self._max_ids_cache:
            max_id = 0
            for bullet, _, _ in self.iter_bullets():
                if bullet["bullet_id"].startswith(f"{section_abbr}-"):
                    try:
                        max_id = max(max_id, int(bullet["bullet_id"].split("-")[1]))
                    except ValueError:
                        pass
            self._max_ids_cache[section_abbr] = max_id
        self._max_ids_cache[section_abbr] += 1
        return f"{section_abbr}-{self._max_ids_cache[section_abbr]:04d}"
