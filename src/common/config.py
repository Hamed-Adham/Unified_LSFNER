"""
Centralized Configuration & Path Resolvers for Unified Hybrid ALinkNER.
Dynamically resolves paths relative to the project root.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

# Automatically find the project root (Unified projectfolder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env file from project root if present
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Global Candidate Suppression & NMS Control
APPLY_NMS = os.getenv("APPLY_NMS", "False").strip().lower() in ("true", "1", "yes")

# Base Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"
LINKNER_DIR = OUTPUT_DIR / "linkner"
DINASOR_DINGEN_DIR = OUTPUT_DIR / "dinasor_dingen"
GATING_NN_DIR = OUTPUT_DIR / "gating_nn"
BERTIZED_ACE_DIR = OUTPUT_DIR / "bertized_ace"
ABLATION_DIR = OUTPUT_DIR / "ablation_reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
GUIDELINES_DIR = DATA_DIR / "guidelines"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Model Checkpoint Paths
SPANNER_160_MODEL_PATH = MODELS_DIR / "SpanNER_LSF" / "best_spanner_160train.pt"
SPANNER_DEFAULT_MODEL_PATH = MODELS_DIR / "SpanNER_LSF" / "best_spanner_160train.pt"
ROBERTA_TOKEN_MODEL_DIR = MODELS_DIR / "NER_Model" / "trained_NER_model"

# Data Splits
DATA_SPLITS = {
    "400_train": DATA_DIR / "new" / "400Abstracts",
    "200_test": DATA_DIR / "new" / "200Abstracts",
    "600_all": DATA_DIR / "600Abstracts",
    "lsf200_relabeled": DATA_DIR / "Relabaled_LSF200_NoDisease",
    "train_relabeled": DATA_DIR / "Relabaled_Train_folder_noDisease_statistics_Added",
    "test_relabeled": DATA_DIR / "Relabaled_test_folder_noDisease",
    "subsets": DATA_DIR / "Genereted_Subsets",
}

# Supported Data Formats
FORMATS = {
    "BIO": {
        "File_format": "bio",
        "description": "BIO format (B-Entity, I-Entity, O)",
    },
    "BRAT": {
        "File_format": "ann",
        "description": "BRAT standoff format (T1 Label Start End Text)",
    },
}

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Default Hyperparameters
DEFAULT_CONFIG = {
    "spanner_threshold": 0.35,
    "margin_threshold": 0.50,
    "w_uncertainty": 0.70,
    "w_novelty": 0.30,
    "uncertainty_method": "margin",
    "novelty_metric": "lof",
    "gating_mode": "prob_or",
    "mcd_passes": 5,
    "use_new_labels": True,
    "use_nms": False,
    "prompt_template_version": "v1_with_label",
    "max_workers": 4,
    "batch_size": 5,
    "evaluation_metrics": ["strict", "partial", "mention"],
}


@dataclass
class LinkNERRunConfig:
    """
    Standardized, modular configuration for Hybrid LinkNER runs and benchmarks.
    Allows configuring NMS, gating operator, weights, uncertainty/novelty estimators,
    thresholds, prompt versions, and evaluation metrics per run.
    """
    # Gating operator & weights
    gating_operator: str = "prob_or"       # 'prob_or', 'weighted', 'or', 'and', 'copula', 'harmonic'
    w_novelty: float = 0.30
    w_uncertainty: float = 0.70
    uncertainty_metric: str = "margin"     # 'margin', 'mcd', 'pe', 'lc'
    novelty_metric: str = "lof"            # 'lof', 'knn', 'mahalanobis', 'none'
    threshold: float = 0.35                # Tau escalation threshold
    margin_veto_threshold: float = 0.50    # Margin safeguard threshold
    use_margin_safeguard: bool = True
    mcd_passes: int = 5

    # Candidate extraction & Prompt template
    use_nms: bool = False                  # True = apply Longest-Span First NMS, False = drop NMS
    prompt_template_version: str = "v1_with_label"  # 'v1_with_label', 'v2_no_label', 'legacy'
    use_new_labels: bool = True

    # Execution & Evaluation
    evaluation_metrics: List[str] = field(default_factory=lambda: ["strict", "partial", "mention"])
    batch_size: int = 5
    max_workers: int = 4

    @classmethod
    def optimal_config(cls) -> "LinkNERRunConfig":
        """Returns the recommended optimal configuration."""
        return cls(
            gating_operator="prob_or",
            w_novelty=0.30,
            w_uncertainty=0.70,
            uncertainty_metric="margin",
            novelty_metric="lof",
            threshold=0.35,
            margin_veto_threshold=0.50,
            use_margin_safeguard=True,
            use_nms=False,
            prompt_template_version="v1_with_label",
            mcd_passes=5,
            evaluation_metrics=["strict", "partial", "mention"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gating_operator": self.gating_operator,
            "w_novelty": self.w_novelty,
            "w_uncertainty": self.w_uncertainty,
            "uncertainty_metric": self.uncertainty_metric,
            "novelty_metric": self.novelty_metric,
            "threshold": self.threshold,
            "margin_veto_threshold": self.margin_veto_threshold,
            "use_margin_safeguard": self.use_margin_safeguard,
            "mcd_passes": self.mcd_passes,
            "use_nms": self.use_nms,
            "prompt_template_version": self.prompt_template_version,
            "use_new_labels": self.use_new_labels,
            "evaluation_metrics": self.evaluation_metrics,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }

