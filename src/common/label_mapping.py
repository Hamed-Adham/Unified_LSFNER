"""
LSF Domain Label Mapping and Schema Conversion Utilities.
Provides bidirectional mappings and translation helpers between legacy and new (relabeled) ontologies:

Legacy Schema:
  - Beauty_and_Cleaning
  - Drugs
  - Physical_activity
  - Nutrition, Socioeconomic_factors, Environmental_exposures, Sleep,
    Mental_health_practices, Non_physical_leisure_time_activities, Lifestyle_factor, O

New / Relabeled Schema:
  - Personal_care_products_and_cosmetic_procedures
  - Substance_use
  - Physical_activities
  - Nutrition, Socioeconomic_factors, Environmental_exposures, Sleep,
    Mental_health_practices, Non_physical_leisure_time_activities, Lifestyle_factor, O
"""

from typing import Dict, List, Any, Optional, Union

# Bidirectional Mappings between Legacy and Relabeled Schemas
OLD_TO_NEW_LABELS: Dict[str, str] = {
    "Beauty_and_Cleaning": "Personal_care_products_and_cosmetic_procedures",
    "Drugs": "Substance_use",
    "Physical_activity": "Physical_activities",
    "Nutrition": "Nutrition",
    "Socioeconomic_factors": "Socioeconomic_factors",
    "Environmental_exposures": "Environmental_exposures",
    "Sleep": "Sleep",
    "Mental_health_practices": "Mental_health_practices",
    "Non_physical_leisure_time_activities": "Non_physical_leisure_time_activities",
    "Lifestyle_factor": "Lifestyle_factor",
    "O": "O",
}

NEW_TO_OLD_LABELS: Dict[str, str] = {
    "Personal_care_products_and_cosmetic_procedures": "Beauty_and_Cleaning",
    "Substance_use": "Drugs",
    "Physical_activities": "Physical_activity",
    "Nutrition": "Nutrition",
    "Socioeconomic_factors": "Socioeconomic_factors",
    "Environmental_exposures": "Environmental_exposures",
    "Sleep": "Sleep",
    "Mental_health_practices": "Mental_health_practices",
    "Non_physical_leisure_time_activities": "Non_physical_leisure_time_activities",
    "Lifestyle_factor": "Lifestyle_factor",
    "O": "O",
}

# Ordered Label Lists
LEGACY_LABEL_LIST: List[str] = [
    "O",
    "Beauty_and_Cleaning",
    "Drugs",
    "Environmental_exposures",
    "Lifestyle_factor",
    "Mental_health_practices",
    "Non_physical_leisure_time_activities",
    "Nutrition",
    "Physical_activity",
    "Sleep",
    "Socioeconomic_factors",
]

RELABELED_LABEL_LIST: List[str] = [
    "O",
    "Personal_care_products_and_cosmetic_procedures",
    "Substance_use",
    "Environmental_exposures",
    "Lifestyle_factor",
    "Mental_health_practices",
    "Non_physical_leisure_time_activities",
    "Nutrition",
    "Physical_activities",
    "Sleep",
    "Socioeconomic_factors",
]

# 9 Core Benchmark Lifestyle Categories (matching the original LSFO-expansion 9-branch benchmark)
NINE_CORE_LEGACY_LABELS: List[str] = [
    "Beauty_and_Cleaning",
    "Drugs",
    "Environmental_exposures",
    "Mental_health_practices",
    "Non_physical_leisure_time_activities",
    "Nutrition",
    "Physical_activity",
    "Sleep",
    "Socioeconomic_factors",
]

NINE_CORE_RELABELED_LABELS: List[str] = [
    "Personal_care_products_and_cosmetic_procedures",
    "Substance_use",
    "Environmental_exposures",
    "Mental_health_practices",
    "Non_physical_leisure_time_activities",
    "Nutrition",
    "Physical_activities",
    "Sleep",
    "Socioeconomic_factors",
]

# 10 Canonical Output Classes (Background O + 9 Core Relabeled Categories, excluding Lifestyle_factor)
CANONICAL_10_LABELS: List[str] = ["O"] + NINE_CORE_RELABELED_LABELS
CANONICAL_10_LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(CANONICAL_10_LABELS)}
CANONICAL_10_ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(CANONICAL_10_LABELS)}

# Labels excluded from precision/recall/F1 metrics (Background O and root umbrella Lifestyle_factor)
EXCLUDED_METRIC_LABELS = {"o", "lifestyle_factor"}

# Set of all known valid lifestyle labels across both schemas
ALL_KNOWN_LABELS = set(LEGACY_LABEL_LIST) | set(RELABELED_LABEL_LIST) | set(CANONICAL_10_LABELS)


def to_new_label(label: str) -> str:
    """Map a single legacy label (or IOB tag) to the new relabeled ontology schema."""
    if not label:
        return "O"
    
    # Handle IOB prefixes (e.g. B-Beauty_and_Cleaning -> B-Personal_care_products_and_cosmetic_procedures)
    prefix = ""
    clean_label = label.strip()
    if clean_label.startswith("B-") or clean_label.startswith("I-"):
        prefix = clean_label[:2]
        clean_label = clean_label[2:]
        
    mapped = OLD_TO_NEW_LABELS.get(clean_label, clean_label)
    return f"{prefix}{mapped}" if prefix else mapped


def to_old_label(label: str) -> str:
    """Map a single new relabeled label (or IOB tag) to the legacy ontology schema."""
    if not label:
        return "O"
    
    # Handle IOB prefixes
    prefix = ""
    clean_label = label.strip()
    if clean_label.startswith("B-") or clean_label.startswith("I-"):
        prefix = clean_label[:2]
        clean_label = clean_label[2:]
        
    mapped = NEW_TO_OLD_LABELS.get(clean_label, clean_label)
    return f"{prefix}{mapped}" if prefix else mapped


def map_label(label: str, target_schema: str = "new") -> str:
    """
    Map a label to the requested target schema.
    
    Args:
        label: The category label string.
        target_schema: 'new' (or 'relabeled') vs 'old' (or 'legacy').
    """
    if not label:
        return "O"
    schema = target_schema.lower().strip()
    if schema in ("new", "relabeled", "v2", "true"):
        return to_new_label(label)
    elif schema in ("old", "legacy", "v1", "false"):
        return to_old_label(label)
    return label


def canonicalize_label(label: str, use_new_labels: bool = False) -> str:
    """
    Normalizes a label string, resolving synonyms/variations to the target schema.
    """
    if not label:
        return "O"
    l_str = label.strip()
    
    # 1. Exact match in target schema
    if use_new_labels:
        if l_str in NEW_TO_OLD_LABELS:
            return l_str
        if l_str in OLD_TO_NEW_LABELS:
            return OLD_TO_NEW_LABELS[l_str]
    else:
        if l_str in OLD_TO_NEW_LABELS:
            return l_str
        if l_str in NEW_TO_OLD_LABELS:
            return NEW_TO_OLD_LABELS[l_str]
            
    # 2. Map LSF_out_of_context and non-entity variants to Non-LSF
    l_lower = l_str.lower()
    if l_lower in ("lsf_out_of_context", "out_of_context"):
        return "Non-LSF"
    if l_lower in ("non_lsf", "non-lsf"):
        return "Non-LSF"

    # 3. Case-insensitive match across all known labels
    for old_k, new_v in OLD_TO_NEW_LABELS.items():
        if old_k.lower() == l_lower or new_v.lower() == l_lower:
            return new_v if use_new_labels else old_k
            
    return l_str


def map_prediction_item(pred: Dict[str, Any], use_new_labels: bool = True) -> Dict[str, Any]:
    """
    Maps all label fields in a pipeline prediction dictionary to the target schema.
    """
    p_copy = dict(pred)
    if "label" in p_copy:
        p_copy["label"] = to_new_label(p_copy["label"]) if use_new_labels else to_old_label(p_copy["label"])
    if "spanner_label" in p_copy and p_copy["spanner_label"] is not None:
        p_copy["spanner_label"] = to_new_label(p_copy["spanner_label"]) if use_new_labels else to_old_label(p_copy["spanner_label"])
    if "llm_label" in p_copy and p_copy["llm_label"] is not None and not p_copy["llm_label"].startswith("❌"):
        p_copy["llm_label"] = to_new_label(p_copy["llm_label"]) if use_new_labels else to_old_label(p_copy["llm_label"])
    if "final_label" in p_copy and p_copy["final_label"] is not None:
        p_copy["final_label"] = to_new_label(p_copy["final_label"]) if use_new_labels else to_old_label(p_copy["final_label"])
    return p_copy


def map_predictions(preds: List[Dict[str, Any]], use_new_labels: bool = True) -> List[Dict[str, Any]]:
    """Map a list of prediction dictionaries to the target schema."""
    return [map_prediction_item(p, use_new_labels=use_new_labels) for p in preds]


def map_ground_truth(entities: List[Dict[str, Any]], use_new_labels: bool = True) -> List[Dict[str, Any]]:
    """Map a list of ground truth entity dictionaries to the target schema."""
    mapped = []
    for ent in entities:
        e_copy = dict(ent)
        if "label" in e_copy:
            e_copy["label"] = to_new_label(e_copy["label"]) if use_new_labels else to_old_label(e_copy["label"])
        mapped.append(e_copy)
    return mapped


def detect_schema_from_path(path: str) -> bool:
    """
    Automatically detects whether a given dataset path uses the new (relabeled)
    schema or the legacy schema.
    
    Args:
        path: Directory path or file path of dataset annotations.
        
    Returns:
        True if the path corresponds to the new / relabeled schema.
        False if the path corresponds to the legacy schema.
    """
    import os
    if not path:
        return False

    p_norm = os.path.abspath(path).replace("\\", "/").lower()

    # 1. Path-based heuristics
    if "/data/new" in p_norm or "/new/" in p_norm or "relabaled" in p_norm or "relabeled" in p_norm:
        return True
    if "/data/legacy" in p_norm or "/legacy/" in p_norm or "lsfo" in p_norm or "lsf_train" in p_norm or "lsf_val" in p_norm:
        return False

    # 2. Content-based inspection of .ann files
    if os.path.isdir(path):
        import glob
        ann_files = glob.glob(os.path.join(path, "*.ann"))[:10]
        for ann in ann_files:
            try:
                with open(ann, "r", encoding="utf-8") as f:
                    content = f.read()
                    if any(l in content for l in ["Personal_care_products_and_cosmetic_procedures", "Substance_use", "Physical_activities"]):
                        return True
                    if any(l in content for l in ["Beauty_and_Cleaning", "Drugs", "Physical_activity"]):
                        return False
            except Exception:
                pass

    return False


# 10-Class Consolidated Ontology Mappings
OLD_TO_NEW_LABELS = {
    'Behavioral_factors': 'Behavioral_factors',
    'Body_metrics': 'Body_metrics',
    'Clinical_parameters': 'Clinical_parameters',
    'Dietary_factors': 'Dietary_factors',
    'Environmental_factors': 'Environmental_factors',
    'Medical_history': 'Medical_history',
    'Medical_interventions': 'Medical_interventions',
    'Physical_activity': 'Physical_activity',
    'Psychological_factors': 'Psychological_factors',
    'Sleep_factors': 'Sleep_factors',
    'O': 'O'
}

NEW_TO_OLD_LABELS = {v: k for k, v in OLD_TO_NEW_LABELS.items()}

# 10-Class IDs
LABEL2ID = {
    'O': 0,
    'Behavioral_factors': 1,
    'Body_metrics': 2,
    'Clinical_parameters': 3,
    'Dietary_factors': 4,
    'Environmental_factors': 5,
    'Medical_history': 6,
    'Medical_interventions': 7,
    'Physical_activity': 8,
    'Psychological_factors': 9,
    'Sleep_factors': 10
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 21-Tag BIO IDs for RoBERTa Token Classification
BIO21_LABELS = [
    'O',
    'B-Behavioral_factors', 'I-Behavioral_factors',
    'B-Body_metrics', 'I-Body_metrics',
    'B-Clinical_parameters', 'I-Clinical_parameters',
    'B-Dietary_factors', 'I-Dietary_factors',
    'B-Environmental_factors', 'I-Environmental_factors',
    'B-Medical_history', 'I-Medical_history',
    'B-Medical_interventions', 'I-Medical_interventions',
    'B-Physical_activity', 'I-Physical_activity',
    'B-Psychological_factors', 'I-Psychological_factors',
    'B-Sleep_factors', 'I-Sleep_factors'
]
BIO21_LABEL2ID = {l: i for i, l in enumerate(BIO21_LABELS)}
BIO21_ID2LABEL = {i: l for i, l in enumerate(BIO21_LABELS)}
