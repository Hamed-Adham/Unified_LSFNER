# Developer Report: Selective Gating & Margin Safeguard Veto Integration in BERTized ACE

**Date**: 2026-09-14  
**Scope**: Integration of SpanNER candidate selective gating and Margin Safeguard Veto into BERTized ACE (`notebooks/07_bertized_ace.ipynb` and `src/architectures/agentic_guidelines/bertized_ace.py`).

---

## 1. Executive Summary & Objective

In standard ACE inference and training, all candidate spans proposed by SpanNER were submitted to the LLM Generator regardless of classifier confidence. In Notebook 04 (`04_dinasor_dingen_pipeline.ipynb`), a selective gating mechanism was demonstrated that filtered out unambiguous, high-confidence spans, routing only ambiguous or novel spans to the LLM agent.

This task ported the selective gating engine into the **BERTized ACE architecture**, adding:
1. **Configurable Dual-Phase Gating**: Independent train (`ENABLE_GATING_TRAIN`) and test (`ENABLE_GATING_TEST`) gating toggles, both active (`True`) by default.
2. **Selective Gating Formulation**: Defaulting to the optimal `prob_or` formula from Notebook 04 ablations:
   `unreliability = 1.0 - (1.0 - novelty) * (1.0 - u_score)`
   with threshold `unreliability_threshold = 0.35`.
3. **Zero-Escalation Fast Path**: When all candidate spans in an abstract are confident, the pipeline skips LLM inference entirely (0 token cost, instant return).
4. **Margin Safeguard Veto**: Active by default (`ENABLE_MARGIN_SAFEGUARD_VETO = True`). If the LLM proposes dropping an escalated candidate to `O` (`Non-LSF`), but SpanNER exhibited high classification confidence (`margin >= 0.50` and `p_o < 0.20` for a non-`O` category), the drop is vetoed and SpanNER's predicted category is restored.
5. **Configurable Background Context**: Confident spans can either be omitted from the prompt (`INCLUDE_CONFIDENT_AS_READONLY = False`, default) or rendered as `[READONLY_CONFIRMED: span (CONFIRMED_SPANNER: label)]` background anchors (`INCLUDE_CONFIDENT_AS_READONLY = True`).

---

## 2. Core Implementations

### A. Gating Engine (`src/architectures/agentic_guidelines/gating.py`)
- `ACEGating` class implementing:
  - `compute_unreliability(u_score, novelty)`: Supports `prob_or`, `or`, `and`, `copula`, and `weighted` modes.
  - `should_escalate(u_score, novelty)`: Boolean decision logic.
  - `filter_candidates(candidate_spans)`: Splits candidate spans into `escalated` and `confident` subsets.
  - `apply_margin_veto(cand, arbitrated_label)`: Safeguards high-margin SpanNER spans against LLM deletion.
- `build_gated_anchored_abstract(abstract, escalated_spans, confident_spans, include_confident)`: Builds targeted span anchors into the raw abstract text for LLM attention.

### B. Pipeline Integration (`src/architectures/agentic_guidelines/bertized_ace.py`)
- `BertizedACEPipeline.__init__`: Exposes gating toggles and hyperparameters (`enable_gating_train`, `enable_gating_test`, `gating_mode`, `unreliability_threshold`, `novelty_threshold`, `w_uncertainty`, `w_novelty`, `include_confident_as_readonly`, `enable_margin_safeguard_veto`, `margin_threshold`, `p_o_threshold`).
- `train_abstract()`:
  - When gating is enabled, filters candidates before Generator execution.
  - Implements fast-path zero-escalation safe-fail return if all spans are confident.
  - Reflector and Curator operate strictly on escalated spans.
- `predict_abstract()`:
  - Filters candidates into escalated vs. confident spans.
  - Evaluates escalated spans via Generator LLM with fresh dynamic guidelines.
  - Applies Margin Safeguard Veto on any candidate dropped by LLM.
  - Preserves confident SpanNER predictions directly.
  - Merges and sorts all predictions by character offsets.
- `train_batch_parallel()`:
  - Records `escalated_count` per abstract to monitor escalation rates and token savings across batches.

### C. Notebook 07 Integration (`notebooks/07_bertized_ace.ipynb`)
- **Cell 2**: Added configuration parameters for gating and margin safeguard veto, with detailed startup log reporting.
- **Cell 5**: Initialized `BertizedACEPipeline` with gating parameters.
- **Cell 6**: Enhanced batch training logging to show candidates count, escalated count, and percentage saved (e.g., `[64.2% saved]`).
- **Cell 8**: Enhanced test inference logging to report escalated vs. confident counts and any triggered margin safeguard vetoes per abstract.

---

## 3. Verification & Validation

1. **Unit Testing (`gating.py`)**:
   - Verified `prob_or` formula computation:
     - `u_score = 0.2`, `novelty = 0.1` -> `unreliability = 0.28 < 0.35` -> No escalation.
     - `u_score = 0.4`, `novelty = 0.1` -> `unreliability = 0.46 >= 0.35` -> Escalated.
   - Verified candidate filtering into escalated vs confident partitions.
   - Verified Margin Safeguard Veto triggers when `margin = 0.70 >= 0.50` and `p_o = 0.05 < 0.20` on LLM drop to `O`.
   - Verified `build_gated_anchored_abstract` for both `include_confident=False` and `include_confident=True`.
2. **End-to-End Pipeline Verification**:
   - Tested `BertizedACEPipeline` with mock confident candidates:
     - `predict_abstract` returned confident predictions with `escalated_to_llm = False` and `source = "SpanNER Confident Baseline (Gating Fast-Path)"`.
     - `train_abstract` completed instantly with `escalated_count = 0` via gating bypass.
3. **AST Knowledge Graph**:
   - Re-indexed codebase with `graphify update .` (48 communities, 949 nodes, 1855 edges).
