# Research Report: Jev Model Architecture & Integration Assessment for Unified_LSFNER

- **Date:** September 20, 2026
- **Target Project:** `Unified_LSFNER` (Hybrid Neuro-Symbolic & Agentic NER Framework)
- **Subject:** Evaluation of **Jev** (TypeSafe AI) for candidate classification, gating, and noise filtering

---

## 1. Executive Summary & What is Jev?

**Jev** is a non-autoregressive "System One" decision model developed by **TypeSafe AI** (founded by Diogo Almeida, co-inventor of RLHF and InstructGPT at OpenAI, alongside Erik Gafni and Sasha Sheng), officially emerging from stealth in mid-September 2026.

### The "System One" Concept
Named after Daniel Kahneman's cognitive dichotomy (System 1: fast, parallel, reflexive vs. System 2: slow, deliberative, sequential):
- **Traditional LLMs (System 2 / Autoregressive):** Generate outputs token-by-token. Excellent at narrative reasoning, code generation, and unstructured generation, but suffer from high latency (1,000ms to 4,000ms), non-deterministic formatting, parsing failures, and high token costs.
- **Jev (System 1 / Parallel Decision Model):** Processes input context in a **single forward parallel pass**. It does not generate conversational prose or markdown text. Instead, it takes an input state and returns **strictly typed, probabilistic decisions**.

### Key Technical Primitives
Jev operates through three native typed interfaces:
1. **`Choice`**: Selects one option from a dictionary of criteria. Returns the chosen label, the full probability distribution over all categories, and a calibrated confidence score.
2. **`Noul` (Boolean)**: Evaluates a binary proposition (Yes/No), returning a calibrated truth probability.
3. **`Score`**: Rates the input against an ordered ordinal scale (e.g., `["Low", "Medium", "High"]` or numeric rating tiers).

### Training & Economics
- **RLCD (Reinforcement Learning for Calibrated Decisions):** Unlike standard LLMs where softmax probabilities often suffer from miscalibration or extreme overconfidence, Jev is trained specifically so that its confidence metric directly reflects empirical accuracy. A confidence of 0.85 indicates an ~85% empirical correctness probability.
- **Inference Speed:** Under 70ms to 300ms (up to ~190x faster than standard frontier LLM generation).
- **Pricing:** **$0.042 per 1,000,000 input tokens**; output decisions are **$0.00 (free)**. Named after the *Jevons Paradox* (efficiency gains driving higher consumption).

---

## 2. Current Project Architecture Context (`Unified_LSFNER`)

Our codebase is a hybrid framework for Named Entity Recognition on Lifestyle Factors (LSF) and ACE biomedical benchmarks. The current architecture consists of:

1. **Local Neural Core ([spanner_model.py](file:///Users/hmd/Documents/Workspace/Unified_Test/src/architectures/hybrid_linkner/spanner_model.py)):**
   - SpanNER (Span-level BERT/RoBERTa) proposes candidate entity spans from an abstract.
   - Computes local difficulty features: Top-1/Top-2 margins, Monte Carlo Dropout (MCD) variance, predictive entropy, background class probability P(O), and Local Outlier Factor (LOF) / Mahalanobis novelty.
2. **Gating Mechanism ([pipeline.py](file:///Users/hmd/Documents/Workspace/Unified_Test/src/architectures/gating_nn/pipeline.py)):**
   - Logistic Gating Classifier or multi-metric thresholding decides which candidate spans can be resolved locally vs. which need escalation.
3. **LLM Disambiguation / LinkNER Arbitration ([prompt_templates.py](file:///Users/hmd/Documents/Workspace/Unified_Test/src/architectures/hybrid_linkner/prompt_templates.py)):**
   - Escalated spans are formatted into large markdown prompts containing Definitions & Guidelines (D&G) across the 10 canonical LSF categories (Nutrition, Physical_activity, Sleep, Drugs/Substance_use, etc.) plus negative class "O".
   - Autoregressive LLMs (GPT-4o, Claude, Qwen, Ollama) generate text that is regex-parsed to extract final labels.
4. **Agentic Reflection & Guidelines Synthesis ([agentic_guidelines.py](file:///Users/hmd/Documents/Workspace/Unified_Test/src/graphs/agentic_guidelines.py)):**
   - LangGraph-based feedback loop with Reflector and Curator nodes identifying systematic errors and dynamically maintaining guidelines in a Chroma vector store.

---

## 3. How Can Jev Be Used in `Unified_LSFNER`?

There are four primary architectural integration points where Jev aligns with our requirements:

```
                          Input Abstract
                                |
                                v
               [Stage 1: Local SpanNER Neural Core]
                 - High recall span proposal (<10ms)
                 - Computes confidence, entropy, LOF novelty
                                |
              +-----------------+-----------------+
              |                                   |
    [High Confidence / Familiar]         [Uncertain / Novel]
              |                                   |
              v                                   v
       Accept SpanNER                   [Stage 2: Jev "System One"]
       Prediction                       - Affirmative Noise Filter (Noul)
                                        - Category Arbitration (Choice)
                                        - Latency ~150ms | $0.042/M tokens
                                                  |
                               +------------------+------------------+
                               |                                     |
                     [High Calibrated Conf]               [Low Conf or Compound]
                               |                                     |
                               v                                     v
                        Accept Jev Label                   [Stage 3: Full LLM / Agentic]
                                                           - GPT-4o / Claude
                                                           - Span splitting & reflection
```

### Opportunity 1: High-Speed LinkNER Classification (`Choice`)
- **Current Bottleneck:** Autoregressive LLMs generate text line-by-line (`1. Physical_activity\n2. Nutrition...`), requiring string parsers, prompt guardrails, and retries on JSON/formatting failures.
- **Jev Application:**
  - Send the abstract context as `state`.
  - Formulate candidate spans as parallel questions using `Choice`:
    ```python
    questions = {
        f"span_{i}": Choice(
            instructions=f"Classify candidate mention '{span['text']}' into the correct lifestyle factor category.",
            criteria=LSF_CATEGORY_DEFINITIONS
        )
        for i, span in enumerate(candidate_spans)
    }
    ```
  - **Benefits:** No regex parsing, zero markdown hallucination, guaranteed enum outputs, and calibrated probability scores per candidate span.

### Opportunity 2: 3-Tier Cascaded Hybrid Pipeline
Currently, the pipeline has a binary choice: either cheap local SpanNER or expensive frontier LLM.
A 3-tier cascade introduces Jev as an intermediary:
- **Tier 0 (SpanNER):** Ultra-fast local neural inference. If `confidence > 0.90` and `LOF_novelty < 0.20`, keep SpanNER.
- **Tier 1 (Jev System One):** If SpanNER is uncertain or flags moderate novelty, escalate to Jev. Because Jev is calibrated via RLCD:
  - If `jev_confidence >= 0.85`, finalize label immediately.
  - Latency impact: ~150ms.
  - Cost impact: $0.00002 per span.
- **Tier 2 (Frontier LLM / LangGraph Agentic Pipeline):** Escalate to GPT-4o only when Jev confidence is low (< 0.85) or when complex compound span decomposition is required.
- **Impact:** Offloads 80% to 90% of escalated LLM calls away from GPT-4o, reducing aggregate inference cost and total pipeline latency significantly.

### Opportunity 3: Affirmative Noise & Biomarker Filtering (`Noul`)
In biomedical abstracts, SpanNER frequently flags false positive spans (e.g., "fasting blood glucose", "serum cholesterol", "SMOKE study group", "orthopedic surgery").
- Under our LinkNER rules (Rule 3), internal biomarkers, study group names, and hospital surgical procedures must be mapped to "O" (negative).
- Jev's `Noul` primitive can act as a rapid binary gatekeeper:
  ```python
  "is_valid_lifestyle": Noul(
      instructions="Is this candidate mention a voluntary behavioral, dietary, or environmental lifestyle factor, rather than an internal biological metabolite/biomarker, demographic category, or clinical surgical procedure?"
  )
  ```
  If `probability < 0.20`, the span is pruned immediately without further processing.

### Opportunity 4: Semantic Gating Router (`Score`)
Instead of relying solely on shallow logistic regression over scalar features (entropy, margin, LOF score), Jev's `Score` can evaluate semantic difficulty directly:
```python
"difficulty": Score(
    instructions="How ambiguous or context-dependent is this entity classification?",
    criteria=["Direct/Unambiguous", "Moderate Context Required", "Highly Ambiguous/Compound"]
)
```

---

## 4. Where is Jev NOT Useful? (Critical Limitations)

While Jev excels at structured classification and routing, it has clear architectural boundaries that prevent it from replacing generative LLMs entirely:

### 1. Inability to Generate Text or Split Compound Mentions
- **Rule 2 of LinkNER:** If a candidate span is compound (e.g., `"dietary counseling and nicotine patches"` or `"religion and alcohol consumption"`), the system must split the span into sub-spans:
  - `"religion"` -> `Socioeconomic_factors`
  - `"alcohol consumption"` -> `Nutrition`
- **Jev Limitation:** Jev is non-autoregressive and strictly typed. It cannot produce new string tokens or output arbitrary boundary character offsets. Compound span decomposition still requires token-level neural boundary models or generative LLMs.

### 2. Inability to Perform Agentic Reflection & Dynamic Guideline Synthesis
- In our LangGraph agentic pipeline ([agentic_guidelines.py](file:///Users/hmd/Documents/Workspace/Unified_Test/src/graphs/agentic_guidelines.py)), the Reflector node inspects errors, performs diagnostic root-cause analysis, and writes new natural-language guideline bullets.
- **Jev Limitation:** Jev cannot author explanations, summarize trends, or synthesize guidelines.

### 3. Cloud Dependency & Network Latency
- Jev is currently available only via TypeSafe AI's hosted cloud API (`api.typesafe.ai` or LiteLLM proxy).
- For edge deployments or strictly offline environments where local SpanNER and local Ollama are deployed, Jev introduces an external network and API key dependency.

---

## 5. Comparative Evaluation Matrix

| Metric / Dimension | Local SpanNER | TypeSafe Jev | GPT-4o-mini | GPT-4o |
| :--- | :--- | :--- | :--- | :--- |
| **Model Nature** | Dense Token/Span Neural | System 1 Parallel Decision | Autoregressive LLM | Frontier Autoregressive LLM |
| **Inference Latency** | **< 10ms** (GPU/MPS) | **70ms - 300ms** | 800ms - 1,500ms | 1,500ms - 3,500ms |
| **Input Cost / 1M Tokens** | $0.00 (Self-hosted) | **$0.042** | $0.150 | $2.500 |
| **Output Cost / 1M Tokens**| $0.00 (Self-hosted) | **$0.000 (Free)** | $0.600 | $10.000 |
| **Output Format** | Logits / Softmax tensor | Typed `Choice`, `Noul`, `Score` | Unstructured text / JSON | Unstructured text / JSON |
| **Calibration Quality** | Moderate (overconfident) | **High (RLCD Calibrated)** | Low-to-Moderate | Moderate |
| **Generative Text / Splitting** | No | **No** | Yes | Yes |
| **Agentic Reflection** | No | **No** | Fair | **Excellent** |
| **Offline / Edge Execution** | **Yes (Native PyTorch)**| No (Hosted Cloud API) | No (Cloud API) | No (Cloud API) |

### Cost Comparison Formula:
- Savings of Jev over GPT-4o on input tokens:
  `Input Savings = (Cost_GPT4o - Cost_Jev) / Cost_GPT4o = (2.500 - 0.042) / 2.500 = 98.32%`
- Plus 100% elimination of generative output token costs.

---

## 6. Architecture Mockup: Jev Provider Integration

If integrated into `src/common/llm_client.py`, the client wrapper would look as follows:

```python
import os
from typing import Dict, Any, List, Optional
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

class TypeSafeJevClassifier:
    """
    TypeSafe Jev System One Client for fast, typed entity classification.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.client = TypeSafeClient(api_key=self.api_key)

    def classify_spans(
        self,
        abstract_context: str,
        spans: List[str],
        category_definitions: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Classifies multiple spans in a single parallel pass with calibrated probabilities.
        """
        questions = {
            f"span_{i}": Choice(
                instructions=f"Select the exact lifestyle factor category for the mention: '{span}'.",
                criteria=category_definitions
            )
            for i, span in enumerate(spans)
        }

        response = self.client.system_one(
            state=abstract_context,
            questions=questions
        )

        results = []
        for i, span in enumerate(spans):
            ans = response.answers[f"span_{i}"]
            results.append({
                "span": span,
                "predicted_label": ans.choice,
                "confidence": ans.confidence,
                "probabilities": ans.probabilities
            })
        return results
```

---

## 7. Strategic Recommendation

1. **Keep as Architectural Reference for Scaling:**
   As the dataset scales to large abstract corpora (thousands of papers), frontier LLM API costs and generation latencies become the primary bottleneck. Jev represents an ideal Tier-1 drop-in classifier between local SpanNER and frontier reasoning models.
2. **Complementary, Not a Replacement:**
   Jev cannot replace the generative parts of `Unified_LSFNER` (namely, compound mention text splitting and agentic reflection). However, as a deterministic, calibrated decision layer, it bridges the gap between lightweight neural models and heavy generative agents.
