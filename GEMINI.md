# Project Guidelines & Output Formatting Rules

## 1. Math and Formula Formatting
- **NEVER output raw LaTeX syntax** (`$ ... $`, `$$ ... $$`, `\frac{...}{...}`, `\mathbf{...}`, `\text{...}`, etc.) in chat responses or markdown documents. The Antigravity chat renderer does not parse LaTeX, resulting in raw unreadable backslashes.
- **ALWAYS use clean plain text, unicode, or code formatting for all mathematical expressions**:
  - Write inline formulas as: `TP / (TP + FN) = 1164 / (1164 + 200) = 85.34%`
  - Write ratios as: `Covered GT / Total GT = 897 / 1174 = 76.41%`
  - Use markdown bold (`**85.34%**`) instead of `\mathbf{85.34\%}`.
  - For multi-line math derivations, use fenced `text` blocks.
