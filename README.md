# Unified Hybrid ALinkNER: Production-Grade Framework

A unified, modular framework combining neural span representations ([SpanNER](src/architectures/hybrid_linkner/spanner_model.py)), uncertainty-novelty unreliability gating, multi-agent reflection ([Dinasor](src/architectures/dinasor_dingen/dinasor_agent.py)), and dynamic self-curated annotation guidelines.

---

## 📁 Repository Structure

```text
Unified_LSFNER/
├── notebooks/                                 # Consolidated interactive Jupyter Notebooks
│   ├── 01_extract_entity_embeddings_vectordb.ipynb
│   ├── 02_escalation_dataset_preparation_gating_nn.ipynb
│   ├── 03_linkner_pipeline_audit.ipynb
│   ├── 04_dinasor_dingen_pipeline.ipynb
│   ├── 05_ablation_sweeps_and_pareto.ipynb
│   └── 06_agentic_guideline_curation_ner.ipynb
│
├── scripts/                                   # Centralized CLI tools & standalone utilities
│   ├── run_evaluation.py                      # Benchmarking CLI (hybrid LinkNER)
│   ├── manage_guidelines.py                   # Guideline inspection & export CLI
│   ├── generate_subsets.py                    # Dataset subset sampler CLI
│   └── compute_iaa.py                         # Inter-Annotator Agreement (IAA) CLI
│
├── src/
│   ├── common/                                # Shared foundational modules
│   │   ├── config.py                          # Dynamic paths & configuration
│   │   ├── llm_client.py                      # Multi-backend LLM provider (OpenAI, LMStudio, Ollama, Dummy)
│   │   ├── label_mapping.py                   # Canonical 20-class <-> 10-class mappings
│   │   ├── dataset_utils.py                   # BRAT <-> BIO formatters, sentence splitters
│   │   ├── evaluation.py                      # Exact / Partial PRF metrics & IAA scoring
│   │   ├── audit_logger.py                    # 8-outcome escalation diagnostic logger
│   │   └── weave_wrapper.py                   # W&B / Weave tracking integration
│   │
│   ├── architectures/                         # 5 Dedicated Architecture Subfolders (Zero Conflicts)
│   │   │
│   │   ├── hybrid_linkner/                    # Architecture 1: SpanNER + Single-Pass LLM Arbitration + LOF
│   │   │   ├── spanner_model.py               # SpanNER neural span classifier
│   │   │   ├── novelty_detector.py            # Local Outlier Factor (LOF), kNN & Mahalanobis
│   │   │   ├── prompt_templates.py            # LSF category definitions & prompt builders
│   │   │   └── pipeline.py                    # LinkNERPipeline orchestrator
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
│   │   └── agentic_guidelines/                # Architecture 5: Pure LLM Self-Reflective Curation
│   │       ├── generator.py                   # Entity extractor from guidelines
│   │       ├── reflector.py                   # Discrepancy identifier vs gold labels
│   │       ├── curator.py                     # Guideline rule synthesizer & pruner
│   │       ├── combined_agent.py              # ReflectorCuratorCombined single-pass agent
│   │       ├── manager.py                     # DynamicGuidelinesManager (schema & vector store)
│   │       └── pipeline.py                    # AgenticNERPipeline train/test orchestrator
│   │
│   └── prompts/                               # Markdown & Text Prompt Assets (Organized per Architecture)
│       ├── agentic_guidelines/                # Prompts for Agentic Guidelines architecture
│       │   ├── Generator.txt
│       │   ├── Reflector.txt
│       │   ├── Curator.txt
│       │   └── reflector_curator_prompt.txt
│       ├── dinasor_dingen/                    # Prompts for Dinasor & DinGenerator architecture
│       │   ├── Dinasor.txt
│       │   ├── Dinasor_primary.txt
│       │   ├── DinGenerator.txt
│       │   └── DinGenerator_primary.txt
│       └── gating_nn/                         # Prompts for Gating NN & Single LLM Escalation
│           ├── FINAL_v1_with_label.md
│           └── FINAL_v2_no_label.md
│
├── data/                                      # Datasets, splits, guidelines, and vector DB
│   ├── raw/                                   # Original annotated abstracts
│   ├── splits/                                # Train (400), Val (80), Test (200) splits
│   ├── guidelines/                            # Base and dynamic guidelines JSON
│   └── vector_db/                             # Persistent ChromaDB collection & entity embeddings
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

## 🚀 Quickstart & Usage

### 1. Interactive Notebooks (`notebooks/`)
Run any of the numbered notebooks in sequence:
- **`01_extract_entity_embeddings_vectordb.ipynb`**: Embeds all entities into ChromaDB for novelty scoring.
- **`02_escalation_dataset_preparation_gating_nn.ipynb`**: Generates dual-prompt tabular dataset for Gating NN training.
- **`03_linkner_pipeline_audit.ipynb`**: Runs 3-step evaluation of the standard hybrid LinkNER pipeline.
- **`04_dinasor_dingen_pipeline.ipynb`**: Evaluates 4-step progressive multi-agent pipeline with anchored reflection.
- **`05_ablation_sweeps_and_pareto.ipynb`**: Executes 6-stage hyperparameter sweeps, Pareto frontier, and component ablation.
- **`06_agentic_guideline_curation_ner.ipynb`**: Runs pure LLM Generator-Reflector-Curator guideline learning loop.

### 2. Python Script Imports
All architectures can be imported cleanly in custom scripts without conflicts:

```python
import sys, os
from src.common.config import PROJECT_ROOT, DATA_SPLITS
from src.common.label_mapping import canonicalize_label

# Architecture 1: Hybrid LinkNER
from src.architectures.hybrid_linkner import LinkNERPipeline, SpanNERModel, LOFNoveltyDetector

# Architecture 2: Dinasor DinGen Multi-Agent
from src.architectures.dinasor_dingen import SpanNER_DinGenPipeline, Dinasor, DinGenerator

# Architecture 3: Gating NN Single-Pass Pipeline & Dataset Builder
from src.architectures.gating_nn import GatingNNPipeline, GatingDatasetBuilder

# Architecture 4: Ablation & Sweep Engine
from src.architectures.ablation_engine import AblationEngine

# Architecture 5: Dynamic Guideline Learning
from src.architectures.agentic_guidelines import AgenticNERPipeline, DynamicGuidelinesManager

```

---

## 📦 Large Files & Pretrained Models (Google Drive)

Due to GitHub's file size limitations (>100 MB), heavy model checkpoints and the 453 MB pre-extracted Vector Database are hosted externally on Google Drive:

* 📥 **Google Drive Download Link**: [Download Models & Vector DB Bundle](https://drive.google.com/drive/folders/1yjqGMCM6bqTaKgwEuIA5nW0oZd1bn7fi?usp=drive_link)

### Extracted Directory Placement:
After downloading, place or extract the files into the repository as follows:

1. **Neural Model Checkpoints (`models/`)**:
   ```bash
   models/
   ├── SpanNER_LSF/best_spanner_160train.pt     # SpanNER candidate proposal model 
   ├── NER_Model/trained_NER_model/             # Fine-tuned RoBERTa token model 
   └── Qwen3-Embedding-8B/                      # Qwen 8B embedding weights 
   ```

2. **Entity Vector Database (`data/vector_db/`)**:
   ```bash
   data/vector_db/
   ├── lsf_entity_vector_db.json                # Pre-extracted entity embedding vectors
   └── chroma_db/                               # Persistent ChromaDB vector index
   ```

