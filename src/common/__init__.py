"""
Common foundational utilities for Unified Hybrid ALinkNER.
"""

from src.common.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    PROMPTS_DIR,
    OUTPUT_DIR,
    GUIDELINES_DIR,
    VECTOR_DB_DIR,
    SPANNER_160_MODEL_PATH,
    SPANNER_DEFAULT_MODEL_PATH,
    ROBERTA_TOKEN_MODEL_DIR,
    DATA_SPLITS,
    FORMATS,
    DEFAULT_CONFIG
)

from src.common.label_mapping import (
    OLD_TO_NEW_LABELS,
    NEW_TO_OLD_LABELS,
    to_new_label,
    to_old_label,
    canonicalize_label,
    map_label,
    map_prediction_item,
    map_predictions,
    ALL_KNOWN_LABELS,
    LABEL2ID,
    ID2LABEL,
    BIO21_LABEL2ID,
    BIO21_ID2LABEL
)

from src.common.llm_client import (
    BaseLLMProvider,
    DummyLLMProvider,
    OpenAIProvider,
    GenericAPIProvider,
    get_llm_provider,
    generate_llm_response,
    last_call_metadata
)

from src.common.audit_logger import (
    compute_audit_breakdown,
    format_audit_breakdown_table,
    format_audit_summary_box
)
