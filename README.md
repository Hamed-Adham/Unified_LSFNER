# Unified Hybrid ALinkNER: Production-Grade Framework

A unified, modular framework combining neural span representations ([SpanNER](src/architectures/hybrid_linkner/spanner_model.py) and [TwoStageSpanNER](src/architectures/hybrid_linkner/two_stage_spanner_model.py)), uncertainty-novelty unreliability gating, 10-outcome escalation diagnostic auditing, multi-agent reflection ([Dinasor](src/architectures/dinasor_dingen/dinasor_agent.py) and [LangGraph](src/graphs/)), and dynamic self-curated annotation guidelines ([BERTized ACE](src/architectures/agentic_guidelines/bertized_ace.py)).

---

## 📁 Repository Structure

```text
Unified_LSFNER/
├── notebooks/                                 # Consolidated interactive Jupyter Notebooks
│   ├── 01_extract_entity_embeddings_vectordb.ipynb
│   ├── 02_escalation_dataset_preparation_gating_nn.ipynb
│   ├── 03_linkner_pipeline_audit.ipynb        # 10-outcome escalation audit of hybrid LinkNER
│   ├── 04_dinasor_dingen_pipeline.ipynb       # 4-step progressive multi-agent pipeline
│   ├── 04_dinasor_dingen_pipeline_graph.ipynb # LangGraph stateful multi-agent pipeline
│   ├── 05_ablation_sweeps_and_pareto.ipynb    # 6-stage hyperparameter sweeps & Pareto engine
│   ├── 06_agentic_guideline_curation_ner.ipynb# Pure LLM guideline curation loop
│   └── 07_bertized_ace.ipynb                  # BERTized ACE dynamic guidebook optimization
│
├── scripts/                                   # Centralized CLI tools & standalone utilities
│   ├── benchmark_spanner_models.py            # Single-stage vs Two-stage SpanNER benchmarking
│   ├── cache_spanner_candidates.py            # Pre-compute & cache candidate spans for 0-cost eval
│   ├── compute_iaa.py                         # Inter-Annotator Agreement (IAA) CLI
│   ├── generate_subsets.py                    # Dataset subset sampler CLI
│   ├── manage_guidelines.py                   # Guideline inspection, search, and export CLI
│   ├── prepare_two_stage_datasets.py          # Stage 1 binary and Stage 2 9-class dataset builder
│   ├── run_evaluation.py                      # Benchmarking CLI (hybrid LinkNER)
│   └── train_two_stage_spanner.py             # Training script for two-stage SpanNER models
│
├── src/
│   ├── common/                                # Shared foundational modules
│   │   ├── config.py                          # Dynamic paths & configuration
│   │   ├── llm_client.py                      # Multi-backend LLM provider (OpenAI, LMStudio, Ollama, Dummy)
│   │   ├── label_mapping.py                   # Canonical 20-class <-> 10-class mappings
│   │   ├── dataset_utils.py                   # BRAT <-> BIO formatters, sentence splitters
│   │   ├── evaluation.py                      # Exact / Partial PRF metrics & IAA scoring
│   │   ├── audit_logger.py                    # 10-outcome escalation diagnostic logger
│   │   └── weave_wrapper.py                   # W&B / Weave tracking integration
│   │
│   ├── data/                                  # Data loaders and format converters
│   │   ├── cached_loader.py                   # Cached candidate span proposal loader
│   │   └── dataset_converter.py               # BRAT / BIO dataset converters
│   │
│   ├── graphs/                                # LangGraph workflow architecture
│   │   ├── state.py                           # UnifiedNERState schemas
│   │   └── nodes/                             # Neural, Gating, and Audit pipeline nodes
│   │       ├── neural_nodes.py
│   │       ├── gating_nodes.py
│   │       └── audit_nodes.py
│   │
│   ├── architectures/                         # Dedicated Architecture Submodules
│   │   │
│   │   ├── hybrid_linkner/                    # Architecture 1: SpanNER + Single-Pass LLM Arbitration + LOF
│   │   │   ├── spanner_model.py               # Single-stage SpanNER neural span classifier
│   │   │   ├── two_stage_spanner_model.py     # Two-stage SpanNER (Binary Boundary + 9-Class Classifier)
│   │   │   ├── novelty_detector.py            # Local Outlier Factor (LOF), kNN & Mahalanobis
│   │   │   ├── prompt_templates.py            # LSF category definitions & prompt builders
│   │   │   └── pipeline.py                    # LinkNERPipeline orchestrator (10-outcome escalation)
│   │   │
│   │   ├── dinasor_dingen/                    # Architecture 2: Reflective Multi-Agent Pipeline
│   │   │   ├── dinasor_agent.py               # Dinasor confidence arbiter & dynamic few-shot retriever
│   │   │   ├── dingen_agent.py                # DinGenerator structured reflection builder
│   │   │   ├── masked_anchors.py              # Anchored sentence masking & challenge classification
│   │   │   └── pipeline.py                    # SpanNER_DinGenPipeline orchestrator
│   │   │
│   │   ├── gating_nn/                         # Architecture 3: Gating NN Dataset & Feature Extraction
│   │   │   ├── dataset_builder.py             # Dual-prompt (v1 with label vs v2 blind) extractor
│   │   │   ├── span_features.py               # Neural feature extractor (MCD, margin, novelty, background prob)
│   │   │   └── prompt_templates.py            # Markdown prompt templates loader & inliner formatter
│   │   │
│   │   ├── ablation_engine/                   # Architecture 4: 6-Stage Parameter Sweep & Pareto Engine
│   │   │   ├── sweep_engine.py                # Operators, weights, uncertainty, novelty, tau sweeps
│   │   │   └── decision_cache.py              # PersistentLLMDecisionCache (0-cost reproducibility)
│   │   │
│   │   └── agentic_guidelines/                # Architecture 5: BERTized ACE & Dynamic Guideline Curation
│   │       ├── bertized_ace.py                # BERTized ACE iterative curation & optimization loop
│   │       ├── bullet_ops.py                  # Atomic bullet CRUD operations (ADD, MODIFY, DELETE, MERGE, SPLIT)
│   │       ├── schema_store.py                # Structured JSON schema storage & validation
│   │       ├── vector_store.py                # ChromaDB vector index for dynamic guideline retrieval
│   │       ├── formatter.py                   # Dynamic guideline serialization & prompt formatting
│   │       ├── manager.py                     # DynamicGuidelinesManager orchestrator
│   │       ├── generator.py                   # Entity extractor from guidelines
│   │       ├── reflector.py                   # Discrepancy identifier vs gold labels
│   │       ├── curator.py                     # Guideline rule synthesizer & pruner
│   │       ├── combined_agent.py              # ReflectorCuratorCombined single-pass agent
│   │       └── pipeline.py                    # AgenticNERPipeline train/test orchestrator
│   │
│   └── prompts/                               # Markdown & Text Prompt Assets (Organized per Architecture)
│       ├── agentic_guidelines/                # Prompts for BERTized ACE & Guideline Curation
│       │   ├── ACE_Generator_v3_bob.txt
│       │   ├── ACE_Reflector_v3_bob.txt
│       │   ├── ACE_Curator_v3_bob.txt
│       │   └── history/                       # Archived earlier prompt versions (v1, v2)
│       ├── dinasor_dingen/                    # Prompts for Dinasor & DinGenerator
│       │   ├── Dinasor_primary_bob.txt
│       │   ├── DinGenerator_v2_bob.txt
│       │   └── history/                       # Archived earlier Dinasor prompts
│       └── gating_nn/                         # Prompts for Gating NN & Single LLM Escalation
│           ├── FINAL_v1_allowed_new_span.md
│           ├── FINAL_v1_with_label.md
│           └── FINAL_v2_no_label.md
│
├── data/                                      # Datasets, splits, guidelines, and vector DB
│   ├── raw/                                   # Original annotated abstracts
│   ├── splits/                                # Train (400), Val (80), Test (200) splits
│   ├── processed/                             # Processed datasets (10-class, binary stage 1, 9-class stage 2)
│   ├── cached/                                # Pre-cached candidate span proposals for fast loader
│   ├── guidelines/                            # Dynamic guidebook JSON, backups, and ChromaDB vector store
│   └── vector_db/                             # Entity embeddings & ChromaDB index
│
├── models/                                    # Checkpoints (Symlinked to SpanNER & RoBERTa models)
│
└── output/                                    # Dedicated output folders per architecture
    ├── linkner/
    ├── dinasor_dingen/
    ├── gating_nn/
    ├── bertized_ace/
    └── ablation_reports/
```

---

## 🔬 Core Innovations

### 1. Two-Stage SpanNER Model
Decouples span proposal from entity classification to drastically reduce false negatives and handle class imbalance:
- **Stage 1 (Binary Boundary Detector)**: Focuses exclusively on span extraction (`Entity` vs `O`), maximizing recall.
- **Stage 2 (9-Class Semantic Classifier)**: Classifies proposed entity spans into fine-grained medical LSF categories.
- **Pre-computed Candidate Caching**: Enables zero-GPU downstream evaluation by caching candidate representations to disk via [CandidateSpanCacheLoader](src/data/cached_loader.py).

### 2. 10-Outcome Escalation Diagnostic Framework
Extends standard evaluation metrics to audit hybrid neural-LLM collaboration across 10 deterministic categories:
- **Covered GT**: Ground truth entities captured by candidate proposal.
- **TP Neural**: Correct neural predictions accepted without escalation.
- **Unreliable Escalation**: Spans gated to the LLM due to high MCD variance or LOF novelty.
- **TP LLM**: Correct entities verified or discovered by the LLM.
- **LLM Corrected Neural**: False neural predictions repaired by the LLM.
- **FP LLM / Hallucinated Span**: Diagnostic tracking of LLM over-predictions.

### 3. BERTized ACE (Agentic Concept Evolution)
A self-improving guideline curation system that optimizes medical entity annotation guidelines against validation errors:
- **Delta-Based Bullet Operations**: Executes atomic CRUD operations (`ADD`, `MODIFY`, `DELETE`, `MERGE`, `SPLIT`) on individual guideline bullet points.
- **Hybrid Search**: Leverages ChromaDB vector indexing and JSON schema hierarchy to inject only relevant guideline rules into prompts.
- **Dynamic Checkpoint Backups**: Automatically snapshots guideline evolution states during training iterations.

---

## 🚀 Quickstart & Usage

### 1. Interactive Notebooks (`notebooks/`)
Execute notebooks in workflow order:
- **`01_extract_entity_embeddings_vectordb.ipynb`**: Embeds entity spans into ChromaDB for novelty scoring.
- **`02_escalation_dataset_preparation_gating_nn.ipynb`**: Prepares training tabular datasets for Gating NN.
- **`03_linkner_pipeline_audit.ipynb`**: Evaluates hybrid LinkNER with full 10-outcome diagnostic auditing.
- **`04_dinasor_dingen_pipeline.ipynb`**: Multi-agent reflection pipeline evaluation.
- **`04_dinasor_dingen_pipeline_graph.ipynb`**: LangGraph state-graph multi-agent pipeline.
- **`05_ablation_sweeps_and_pareto.ipynb`**: 6-stage hyperparameter sweeps and Pareto frontier analysis.
- **`06_agentic_guideline_curation_ner.ipynb`**: Baseline agentic guideline curation.
- **`07_bertized_ace.ipynb`**: BERTized ACE optimization loop with bullet-level CRUD refinement.

### 2. Python Script Imports

```python
from src.common.config import PROJECT_ROOT, DATA_SPLITS
from src.common.label_mapping import canonicalize_label

# Architecture 1: Two-Stage SpanNER & Hybrid LinkNER
from src.architectures.hybrid_linkner import (
    LinkNERPipeline,
    SpanNERModel,
    TwoStageSpanNERModel,
    LOFNoveltyDetector,
)
from src.data.cached_loader import CandidateSpanCacheLoader

# Architecture 2: Dinasor-DinGen Multi-Agent
from src.architectures.dinasor_dingen import SpanNER_DinGenPipeline, Dinasor, DinGenerator

# Architecture 3: Gating NN & Feature Extraction
from src.architectures.gating_nn import GatingNNPipeline, GatingDatasetBuilder

# Architecture 4: Ablation & Sweep Engine
from src.architectures.ablation_engine import AblationEngine

# Architecture 5: BERTized ACE Dynamic Guidelines
from src.architectures.agentic_guidelines import (
    AgenticNERPipeline,
    DynamicGuidelinesManager,
    BERTizedACEOptimizer,
)
```

### 3. CLI Utilities

- **Benchmark SpanNER Variants**:
  ```bash
  python scripts/benchmark_spanner_models.py --split val --stage2-model models/SpanNER_LSF/best_spanner_stage2.pt
  ```

- **Cache Candidate Spans**:
  ```bash
  python scripts/cache_spanner_candidates.py --split 400 --subset train
  ```

- **Inspect Dynamic Guidelines**:
  ```bash
  python scripts/manage_guidelines.py --action stats
  ```

---

## 📦 Large Files & Pretrained Models (Google Drive)

Heavy model checkpoints and the pre-extracted Vector Database are hosted externally on Google Drive:

* 📥 **Google Drive Download Link**: [Download Models & Vector DB Bundle](https://drive.google.com/drive/folders/1yjqGMCM6bqTaKgwEuIA5nW0oZd1bn7fi?usp=drive_link)

### Directory Placement:
1. **Neural Model Checkpoints (`models/`)**:
   ```bash
   models/
   ├── SpanNER_LSF/best_spanner_160train.pt     # Single-stage SpanNER proposal model
   ├── NER_Model/trained_NER_model/             # Fine-tuned RoBERTa token model
   └── Qwen3-Embedding-8B/                      # Qwen 8B embedding weights
   ```

2. **Entity Vector Database (`data/vector_db/`)**:
   ```bash
   data/vector_db/
   ├── lsf_entity_vector_db.json                # Pre-extracted entity embedding vectors
   └── chroma_db/                               # Persistent ChromaDB vector index
   ```
