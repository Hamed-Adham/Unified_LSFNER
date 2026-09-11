# Project Guidelines & Output Formatting Rules

## 1. Math and Formula Formatting
- **NEVER output raw LaTeX syntax** (`$ ... $`, `$$ ... $$`, `\frac{...}{...}`, `\mathbf{...}`, `\text{...}`, etc.) in chat responses or markdown documents. The Antigravity chat renderer does not parse LaTeX, resulting in raw unreadable backslashes.
- **ALWAYS use clean plain text, unicode, or code formatting for all mathematical expressions**:
  - Write inline formulas as: `TP / (TP + FN) = 1164 / (1164 + 200) = 85.34%`
  - Write ratios as: `Covered GT / Total GT = 897 / 1174 = 76.41%`
  - Use markdown bold (`**85.34%**`) instead of `\mathbf{85.34\%}`.
  - For multi-line math derivations, use fenced `text` blocks.

## 2. Mandatory Interactive Pair-Programming & Grill-Me Protocol
- **NEVER make silent architectural, algorithmic, or design choices**: When planning features, choosing between alternative ideas, selecting libraries, structuring modules, or deciding edge-case handling, NEVER pick unilaterally.
- **ALWAYS grill the user on trade-offs**:
  - Challenge assumptions, uncover hidden edge cases, and highlight performance/complexity trade-offs.
  - Structure options clearly (e.g., Option A vs. Option B) with concrete pros and cons.
- **MANDATORY use of `ask_question` interactive modals**:
  - Whenever choosing between ideas or before writing non-trivial code, you MUST invoke the `ask_question` tool so interactive selection modals appear in the user's UI.
  - Options must be formatted from the user's perspective, listing the recommended path first with `(Recommended)`.
  - The agent MUST wait for the user to make their selection before proceeding with execution.
- **Active Developer Steering Over Passive Plan Approval**:
  - The developer is an active co-pilot deciding each step, not a passive spectator or mere plan reviewer.
  - Break development into distinct, interactive steps: interrogate & present choices -> user decides -> implement step -> verify -> interrogate next decision fork.
