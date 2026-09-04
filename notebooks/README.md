# 📓 Interactive Research Notebooks Guide

This directory contains the sequential experiments, ablation studies, and evaluation pipelines for the Unified Hybrid ALinkNER framework.

---

## 🚀 Execution Sequence

| # | Notebook | Focus Area | Description |
| :---: | :--- | :--- | :--- |
| **01** | [`01_extract_entity_embeddings_vectordb.ipynb`](01_extract_entity_embeddings_vectordb.ipynb) | Feature Preparation | Extracts entity embeddings using Qwen3-Embedding-8B and populates the local Vector DB (`data/vector_db/`) for Local Outlier Factor (LOF) novelty scoring. |
| **02** | [`02_escalation_dataset_preparation_gating_nn.ipynb`](02_escalation_dataset_preparation_gating_nn.ipynb) | Gating NN Dataset | Prepares the dual-prompt escalation dataset (Prompt v1 with label vs. Prompt v2 blind) with SpanNER uncertainty and novelty metrics. |
| **03** | [`03_linkner_pipeline_audit.ipynb`](03_linkner_pipeline_audit.ipynb) | Hybrid LinkNER Audit | End-to-end evaluation and 8-outcome diagnostic audit of Architecture 1 (SpanNER + LOF Novelty + LLM Escalation). |
| **04a** | [`04_dinasor_dingen_pipeline.ipynb`](04_dinasor_dingen_pipeline.ipynb) | Multi-Agent Reflection | Sequential 4-step progressive multi-agent pipeline with anchored candidate reflection using Dinasor & DinGenerator. |
| **04b** | [`04_dinasor_dingen_pipeline_graph.ipynb`](04_dinasor_dingen_pipeline_graph.ipynb) | LangGraph Orchestration | Graph-based state machine implementation of the Dinasor-DinGenerator architecture with conditional routing and audit nodes. |
| **05** | [`05_ablation_sweeps_and_pareto.ipynb`](05_ablation_sweeps_and_pareto.ipynb) | Ablation & Sweeps | Multi-stage sweeps across gating operators (`PROB_OR`, `WEIGHTED`, `OR`, `AND`), weights, thresholds, and Pareto frontier optimization. |
| **06** | [`06_agentic_guideline_curation_ner.ipynb`](06_agentic_guideline_curation_ner.ipynb) | Guideline Curation | Self-reflective agentic guideline curation loop (Generator -> Reflector -> Curator) evolving `dynamicGuidelines.json`. |
| **07** | [`07_bertized_ace.ipynb`](07_bertized_ace.ipynb) | BERTized ACE Framework | Integration of SpanNER neural proposals with Agentic Curation Engine (ACE) for dynamic boundary refinement and arbitration. |

---

## ⚙️ Prerequisites
Ensure your environment is active and dependencies are installed:
```bash
pip install -r requirements.txt
```
All notebooks automatically resolve the project root and add it to `sys.path`.
