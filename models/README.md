# 🧠 Model Checkpoints & Pretrained Weights

This directory holds the neural checkpoints required to run the Unified Hybrid ALinkNER pipelines.
Due to GitHub file size limitations, pretrained model weights are hosted externally on Google Drive.

---

## 📥 Required Checkpoints

| Model Name | Target Path | Description |
| :--- | :--- | :--- |
| **SpanNER 160** | `models/SpanNER_LSF/best_spanner_160train.pt` | Span-level neural proposal and uncertainty model trained on 160 abstracts |
| **NER Model** | `models/NER_Model/trained_NER_model/` | Fine-tuned RoBERTa token classification encoder and weights |
| **Qwen3 Embedding** | `models/Qwen3-Embedding-8B/` | Qwen 8B embedding model for LOF novelty vector extraction |

---

## 🔗 Download Link

* **Google Drive Bundle**: [Download Models Bundle](https://drive.google.com/drive/folders/YOUR_GOOGLE_DRIVE_FOLDER_LINK_HERE)

Once downloaded, extract or symlink the directories to:
```bash
models/
├── SpanNER_LSF/
│   └── best_spanner_160train.pt
├── NER_Model/
│   └── trained_NER_model/
└── Qwen3-Embedding-8B/
```
