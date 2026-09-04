# Architecture Prompt Templates

This directory contains prompt templates organized by architecture for the Unified Hybrid ALinkNER project.

## Directory Structure

```text
src/prompts/
├── agentic_guidelines/
│   ├── Generator.txt               # LLM prompt for guideline-conditioned entity generation
│   ├── Reflector.txt               # LLM prompt for error analysis and guideline tagging vs ground truth
│   ├── Curator.txt                 # LLM prompt for guideline rule synthesis, refinement & pruning
│   └── reflector_curator_prompt.txt# Single-pass combined Reflector + Curator prompt
│
├── dinasor_dingen/
│   ├── Dinasor.txt                 # Dinasor agent prompt for uncertainty resolution
│   ├── Dinasor_primary.txt         # Dinasor primary variant prompt
│   ├── DinGenerator.txt            # DinGenerator agent prompt for candidate entity generation
│   └── DinGenerator_primary.txt    # DinGenerator primary variant prompt
│
└── gating_nn/
    ├── FINAL_v1_with_label.md      # Escalation prompt v1 with SpanNER predicted candidate label
    └── FINAL_v2_no_label.md        # Escalation prompt v2 without candidate label (blind classification)
```

## Architecture Details

### 1. Agentic Guidelines (`agentic_guidelines/`)
- **Generator (`Generator.txt`)**: Prompts LLM to extract entities strictly adhering to dynamic guidelines and definitions.
- **Reflector (`Reflector.txt`)**: Analyzes mismatches between Generator extractions and gold annotations, outputting structured error reasons.
- **Curator (`Curator.txt`)**: Generates additions, modifications, or deletions to dynamic guideline rules based on reflection reports.
- **Combined Agent (`reflector_curator_prompt.txt`)**: Consolidates reflection and curation into a single LLM call for accelerated training.

### 2. Dinasor & DinGenerator (`dinasor_dingen/`)
- **Dinasor (`Dinasor.txt` / `Dinasor_primary.txt`)**: Specializes in resolving low-confidence/uncertain mentions flagged by SpanNER.
- **DinGenerator (`DinGenerator.txt` / `DinGenerator_primary.txt`)**: Generates candidate entity extractions guided by hint contexts and boundary markers.

### 3. Gating Neural Network Escalation (`gating_nn/`)
- **Version 1 (`FINAL_v1_with_label.md`)**: Single-call prompt providing the candidate mention with SpanNER's predicted category label for verification or re-classification.
- **Version 2 (`FINAL_v2_no_label.md`)**: Single-call prompt providing the candidate mention without any suggested category label for unbiased category assignment.
