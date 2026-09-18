# BERTized ACE Prompt Calibration, Schema Harmonization, and Guideline Cleanup Report

**Date**: 2026-09-14  
**Branch**: `ace_test`  
**Authors/Pair**: Hamed Adham & Antigravity  

---

## 1. Project Context & Current Pipeline Stage

We are currently in the **BERTized Agentic Context Engine (ACE) integration and calibration stage**. 

### Architecture Recap:
1. **Upstream Span Proposal**: A fine-tuned biomedical SpanNER model proposes fixed candidate spans from abstracts, annotated with rich diagnostic signals (`bert_predicted_label`, `second_best_label`, `uncertainty`, `novelty_score`, `margin`).
2. **Anchored Abstract Generation**: Candidate spans are anchored directly within the abstract text via bracketed inline IDs (`[id: X, span: ...]`).
3. **Generator Arbitration (`ACE_Generator_v3`)**: An LLM agent arbitrates each candidate span between the 9 canonical Lifestyle Factor (LSF) categories and `Non-LSF` (`O`), guided by Base Rules, Affirmative Filtering, and the corpus-level `dynamicGuidebook`.
4. **Reflector Diagnosis (`ACE_Reflector_v3`)**: A post-arbitration diagnostic agent categorizes labeling failures using the 10-outcome triad system (`GT_X_BERT_Y_LLM_Z`, `GT_O_BERT_X_LLM_X`, etc.) and derives generalizable semantic reasoning principles (`key_insight`).
5. **Curator Policy Governance (`ACE_Curator_v3`)**: Evaluates candidate insights against genericity and immutability criteria to maintain `dynamicGuidebook.json`.

---

## 2. Issues Addressed & Motivations

During our review of the pipeline inputs, prompt instructions, and production guidelines, four specific deficiencies were identified and addressed:

### Issue 1: Obsolete Sectioned Structure Remnants (`key_insight_section`)
- **Problem**: Earlier iterations of the dynamic guidelines were divided into 12 sub-sections (`ANF`, `CSH`, `ERR`, `NUT`, `SEF`, etc.). The Reflector prompt instructed the model to output a `key_insight_section` abbreviation, and the fallback parsing code retained `"key_insight_section": "ANF"`.
- **Reason for Removal**: The pipeline has transitioned to a flat, unified `dynamicGuidebook.json` indexed by canonical `DG-XXXX` identifiers. Retaining `key_insight_section` introduced dead keys, prompted unnecessary token usage, and risked confusing downstream consumers.

### Issue 2: Empty `bert_info` Objects in `dynamicGuidebook.json`
- **Problem**: All 39 active rules in `data/guidelines/dynamicGuidebook.json` contained an empty dictionary: `"bert_info": {}`.
- **Resolution**: We audited whether the critical diagnostic context was covered elsewhere. In every guideline object, BERT's role is already fully captured by:
  - `bert_label`: The label predicted by BERT on the triggering span.
  - `triad`: The structured failure type and description (e.g., `GT_X_BERT_Y_LLM_Y`).
  - `guideline`: Explicit semantic rules explaining when to confirm BERT vs. when to override it.
  Because historical per-span probability distributions (`prob`, `margin`, `uncertainty`) are neither available across all historical rules nor needed (since the Generator receives real-time metrics for current spans), retaining `"bert_info": {}` added noise and wasted prompt tokens. We removed `bert_info` from `dynamicGuidebook.json`, formatters, test scripts, and prompt definitions.

### Issue 3: LLM Tendency to Blindly Copy or Avoid BERT Predictions
- **Problem**: When shown upstream model predictions, LLMs commonly oscillate between two failure modes:
  1. *Uncritical Anchoring Bias*: Blindly adopting BERT's `predicted_label`, thereby propagating BERT false alarms (`GT_O_BERT_X_LLM_X`) and misclassifications (`GT_X_BERT_Y_LLM_Y`).
  2. *Excessive Avoidance*: Distrusting BERT and over-rejecting valid entities to `Non-LSF` (`O`).
- **Resolution**: Inserted a prominent, concise **Critical Consideration** section right before the BERT Candidate Spans & Evidence section in `ACE_Generator_v3_bob.txt`. It instructs the agent to treat BERT predictions as consultative evidence rather than unquestioned authority, calibrating scrutiny based on uncertainty and margin while enforcing independent contextual reasoning.

### Issue 4: Key Naming Mismatch (`predicted_label` vs `bert_predicted_label`)
- **Problem**: Candidate span objects passed to the Generator used the generic key `"predicted_label"`, while the Generator's required output format specified `"bert_predicted_label"`.
- **Resolution**: Renamed `"predicted_label"` to `"bert_predicted_label"` in the input candidate spans list, test scripts, prompt documentation, and few-shot examples. This aligns input and output schemas and reinforces that this is BERT's upstream suggestion.

---

## 3. Comprehensive Summary of Changes

### A. Prompts
1. **`src/prompts/agentic_guidelines/ACE_Generator_v3_bob.txt`**:
   - Added `### CRITICAL CONSIDERATION ON BERT EVIDENCE (CONSULT, DO NOT BLINDLY COPY)`.
   - Updated references from `predicted_label` to `bert_predicted_label` in evidence descriptions, workflow steps, and few-shot candidate objects.
   - Removed `bert_info` from `### DYNAMICGUIDEBOOK SCHEMA & USAGE` and few-shot guideline examples.
   - Removed legacy prefix section notes (`ANF`, `NUT`), aligning strictly with canonical `DG-XXXX` ID formatting.

2. **`src/prompts/agentic_guidelines/ACE_Reflector_v3_bob.txt`**:
   - Removed `"key_insight_section"` from the target JSON schema and field descriptions.
   - Updated required field count from 8 to 7.
   - Removed `Step 3` sub-section abbreviation list (`ANF`, `CSH`, `ERR`, etc.).
   - Cleaned few-shot examples by removing `"key_insight_section"`.

3. **`src/prompts/agentic_guidelines/ACE_Curator_v3_bob.txt`**:
   - Removed `bert_info` from the `dynamicGuidebook` definition.

### B. Python Pipeline & Utilities
1. **`src/architectures/agentic_guidelines/bertized_ace.py`**:
   - In `BertizedGenerator.run()`, updated candidate span serialization to use `bert_predicted_label`.
   - In `BertizedReflector.run()`, removed `"key_insight_section": "ANF"` from fallback JSON object.

2. **`src/architectures/agentic_guidelines/formatter.py`**:
   - Removed `bert_info` extraction in `to_dinasor_view` and `to_generator_view`.

3. **`src/architectures/agentic_guidelines/bullet_ops.py`**:
   - Removed `"bert_info": op.get("bert_info", {})` when constructing new guideline records.

4. **`babili_test/run_generator_reflector_test.py`**:
   - Updated candidate span list to populate `bert_predicted_label`.
   - Removed `bert_info` from `gen_guidelines_view`.
   - Removed `key_insight_section` from console diagnostic logging.

### C. Data Files
1. **`data/guidelines/dynamicGuidebook.json`**:
   - Stripped `"bert_info": {}` from all 39 entries in the guidebook array, reducing token overhead and cleaning schema structure.

---

## 4. Verification & Validation

1. **JSON Syntax & Integrity**:
   - Confirmed `data/guidelines/dynamicGuidebook.json` loads cleanly with 39 active entries and zero remaining `bert_info` fields.
2. **Prompt Template Consistency**:
   - Verified that `ACE_Generator_v3_bob.txt` contains `bert_predicted_label` and the new critical consideration section.
   - Verified that `ACE_Reflector_v3_bob.txt` contains no traces of `key_insight_section`.
   - Verified that `ACE_Curator_v3_bob.txt` contains no traces of `bert_info`.
3. **AST Knowledge Graph**:
   - Updated AST graph via `graphify` per project rules.
