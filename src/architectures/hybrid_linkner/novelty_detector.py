"""
anomaly_lof.py - Vector Novelty & Anomaly Detection Module for LinkNER Pipeline

Uses scikit-learn's LocalOutlierFactor and Cosine KNN Distance to calculate:
1. Average KNN Distance Score (individual score per entity, batched execution)
2. LOF Score via scikit-learn LocalOutlierFactor (individual score per entity)
3. Combined Calibrated Novelty Score in [0, 1]
4. Context Representation Lookup: Find what training text/entity an embedding represents (dynamic few-shot).
"""

import os
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor
except ImportError:
    LocalOutlierFactor = None



# ==============================================================================
# 1. Embedder Backends (Local RoBERTa default for offline compatibility)
# ==============================================================================

class LocalTransformersEmbedder:
    """Lazy wrappers around the local RoBERTa checkpoint that SpanNER already
    uses, so novelty scoring works fully offline (MPS or CPU).

    Lazy-loads on first embed() and shares one encoder + tokenizer across all spans.
    """
    def __init__(self, model_path: str = None, device: str = None, batch_size: int = 64):
        self.model_path = model_path or os.path.abspath(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/NER_Model/trained_NER_model"))
        )
        self.device = device
        self.batch_size = batch_size
        self._enc = None
        self._tok = None
        self._dim = None

    def _lazy_load(self):
        if self._enc is not None:
            return
        import torch
        from transformers import AutoTokenizer, RobertaModel
        from transformers import logging
        logging.set_verbosity_error()
        device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._enc = RobertaModel.from_pretrained(self.model_path).to(device).eval()
        self._device = device

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        import torch
        self._lazy_load()
        outs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self._tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self._device)
            with torch.no_grad():
                h = self._enc(**enc).last_hidden_state
            # mean-pool over real tokens, then L2-normalize
            mask = enc["attention_mask"].unsqueeze(-1)
            vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            outs.append(torch.nn.functional.normalize(vec, p=2, dim=1).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    def __repr__(self):
        return f"LocalTransformersEmbedder(path={self.model_path})"


class SentenceTransformerEmbedder:
    """sentence-transformers embedder (e.g. Qwen3-Embedding-8B, BAAI/bge-m3)."""
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        truncate_dim: Optional[int] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        **model_kwargs
    ):
        from sentence_transformers import SentenceTransformer
        import torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        is_local = os.path.isdir(model_name)
        kwargs = dict(model_kwargs)
        if is_local and "local_files_only" not in kwargs:
            kwargs["local_files_only"] = True
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=trust_remote_code, **kwargs)
        if truncate_dim:
            try:
                self.model.truncate_sentence_embeddings(truncate_dim)
            except Exception:
                pass
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 4096), dtype=np.float32)
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    def __repr__(self):
        return f"SentenceTransformerEmbedder(model={self.model_name}, device={self.device})"


class OpenAIAPIEmbedder:
    """OpenAI-compatible API Embedder (supports 9Router Proxy, Ollama, LM Studio, vLLM, OpenAI)."""
    def __init__(
        self,
        model_name: str = "qwen-embedding",
        base_url: str = None,
        api_key: str = None,
        batch_size: int = 32,
        timeout: float = 30.0,
    ):
        import openai
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "qwen-embedding")
        self.base_url = (base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("API_Base_URL") or "http://localhost:20128/v1").rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("NINEROUTER_OPENAI_API_KEY") or os.getenv("API_KEY") or "sk-local-gateway"
        self.batch_size = batch_size
        self.timeout = timeout
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 4096), dtype=np.float32)
        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            try:
                resp = self.client.embeddings.create(input=chunk, model=self.model_name)
            except Exception as e:
                # Provide explicit diagnostic message
                raise RuntimeError(
                    f"OpenAIAPIEmbedder failed requesting '{self.model_name}' from '{self.base_url}': {e}"
                ) from e
            sorted_items = sorted(resp.data, key=lambda x: x.index)
            for item in sorted_items:
                v = np.array(item.embedding, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 1e-9:
                    v = v / norm
                all_vecs.append(v)
        return np.vstack(all_vecs).astype(np.float32)

    def __repr__(self):
        return f"OpenAIAPIEmbedder(model={self.model_name}, endpoint={self.base_url})"


DEFAULT_QWEN_PATH = os.getenv(
    "QWEN_EMBEDDING_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/Qwen3-Embedding-8B"))
)
DEFAULT_BGE_MODEL = os.getenv("BGE_EMBEDDING_MODEL", "BAAI/bge-m3")



class FallbackEmbedder:
    """
    Hierarchical Fallback Embedder:
    0. 9Router / OpenAI API Proxy (when EMBEDDING_PROVIDER=9router or endpoint online)
    1. Primary (Default Local): Qwen 8B (local directory path or HuggingFace Hub name)
    2. Secondary Fallback: BGE-M3 (SentenceTransformers)
    3. Tertiary Fallback: Local RoBERTa (Offline LocalTransformersEmbedder)
    """
    def __init__(
        self,
        qwen_path: Optional[str] = None,
        bge_model: Optional[str] = None,
        local_roberta_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        import torch
        if device is None:
            env_dev = os.getenv("EMBEDDING_DEVICE")
            if env_dev:
                device = env_dev
            elif torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.qwen_path = qwen_path or os.getenv("QWEN_EMBEDDING_PATH", DEFAULT_QWEN_PATH)
        self.bge_model = bge_model or os.getenv("BGE_EMBEDDING_MODEL", DEFAULT_BGE_MODEL)
        self.local_roberta_path = local_roberta_path
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self._active_embedder = None
        self._backend_name = None

    def _lazy_init(self):
        if self._active_embedder is not None:
            return

        # 0. Try 9Router / OpenAI API Proxy if configured
        provider = (os.getenv("EMBEDDING_PROVIDER") or "").strip().lower()
        if provider in ("9router", "ninerouter", "gateway", "openai", "api") or os.getenv("EMBEDDING_MODEL"):
            try:
                emb_model = os.getenv("EMBEDDING_MODEL", "qwen-embedding")
                emb_base = os.getenv("EMBEDDING_BASE_URL") or os.getenv("API_Base_URL", "http://localhost:20128/v1")
                emb_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("NINEROUTER_OPENAI_API_KEY") or os.getenv("API_KEY", "sk-local-gateway")
                if self.verbose:
                    print(f"[Embedder] Initializing 9Router / API Embedder: model='{emb_model}' @ {emb_base}...")
                self._active_embedder = OpenAIAPIEmbedder(
                    model_name=emb_model,
                    base_url=emb_base,
                    api_key=emb_key,
                    batch_size=self.batch_size
                )
                self._backend_name = f"9Router-API ({emb_model} @ {emb_base})"
                if self.verbose:
                    print(f"[Embedder] Successfully connected to API embedder: {self._backend_name}")
                return
            except Exception as e:
                if self.verbose:
                    print(f"[Embedder] 9Router API Embedder failed ({e}). Proceeding to local weights...")

        # 1. Try Primary Default: Qwen 8B
        candidates = []
        if self.qwen_path:
            candidates.append((self.qwen_path, "Qwen 8B (Local Path)" if os.path.exists(self.qwen_path) else "Qwen 8B"))
        rel_qwen = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/Qwen3-Embedding-8B"))
        if os.path.exists(rel_qwen) and rel_qwen != self.qwen_path:
            candidates.append((rel_qwen, "Qwen 8B (Project Local)"))


        for q_target, label in candidates:
            try:
                if self.verbose:
                    print(f"[Embedder] Initializing primary default: {label} ({q_target})...")
                    print(f"           Loading 8-Billion parameter weights (~16 GB) from disk into RAM (takes ~15-30s on first load)...")
                self._active_embedder = SentenceTransformerEmbedder(
                    model_name=q_target,
                    batch_size=self.batch_size,
                    device=self.device,
                    trust_remote_code=True,
                )
                self._backend_name = f"Qwen-8B ({q_target})"
                if self.verbose:
                    print(f"[Embedder] Successfully loaded primary default: {self._backend_name}")
                return
            except Exception as e:
                if self.verbose:
                    print(f"[Embedder] Primary {label} unavailable ({e}). Proceeding to secondary fallback...")

        # 2. Try Secondary Fallback: BGE-M3
        if self.bge_model:
            try:
                if self.verbose:
                    print(f"[Embedder] Initializing secondary fallback: BGE-M3 ({self.bge_model})...")
                self._active_embedder = SentenceTransformerEmbedder(
                    model_name=self.bge_model,
                    batch_size=self.batch_size,
                    device=self.device,
                    trust_remote_code=True,
                )
                self._backend_name = f"BGE-M3 ({self.bge_model})"
                if self.verbose:
                    print(f"[Embedder] Successfully loaded secondary fallback: {self._backend_name}")
                return
            except Exception as e:
                if self.verbose:
                    print(f"[Embedder] Secondary BGE-M3 unavailable ({e}). Proceeding to local offline fallback...")

        # 3. Fallback to Local RoBERTa (100% offline)
        if self.verbose:
            print("[Embedder] Initializing tertiary fallback: Local RoBERTa (offline)...")
        self._active_embedder = LocalTransformersEmbedder(
            model_path=self.local_roberta_path,
            device=self.device,
            batch_size=self.batch_size,
        )
        self._backend_name = f"Local-RoBERTa ({self._active_embedder.model_path})"
        if self.verbose:
            print(f"[Embedder] Successfully loaded offline fallback: {self._backend_name}")

    @property
    def active_embedder(self):
        self._lazy_init()
        return self._active_embedder

    @property
    def active_backend_name(self) -> str:
        self._lazy_init()
        return self._backend_name or "Unknown"

    def embed(self, texts: List[str]) -> np.ndarray:
        self._lazy_init()
        try:
            return self._active_embedder.embed(texts)
        except Exception as e:
            if self.verbose:
                print(f"[Embedder WARNING] Active embedder ({self._backend_name}) failed during embed: {e}")
                print(f"[Embedder] Switching to offline Local RoBERTa fallback...")
            # Fallback to local RoBERTa immediately
            self._active_embedder = LocalTransformersEmbedder(
                model_path=self.local_roberta_path,
                device=self.device,
                batch_size=self.batch_size,
            )
            self._backend_name = f"Local-RoBERTa-Emergency-Fallback ({self._active_embedder.model_path})"
            return self._active_embedder.embed(texts)

    def __repr__(self):
        name = self._backend_name or "lazy (default: 9Router -> Qwen 8B -> BGE-M3 -> Local RoBERTa)"
        return f"FallbackEmbedder(active={name})"


def get_default_embedder(
    preferred: str = "qwen",
    device: Optional[str] = None,
    batch_size: int = 32,
    verbose: bool = False,
) -> Any:
    """
    Returns the configured default embedder with hierarchical fallback:
    9Router / OpenAI API -> Qwen 8B Local -> BGE-M3 -> Local RoBERTa.
    """
    return FallbackEmbedder(device=device, batch_size=batch_size, verbose=verbose)


# ==============================================================================
# 2. Reference Ingestion & Helper Utilities
# ==============================================================================

def extract_embeddings_and_records(
    db: Union[str, Path, dict, list, np.ndarray],
    allowed_abstract_ids: Optional[Union[Set[str], List[str]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extracts a 2D numpy array of embeddings and metadata records, strictly filtering by allowed_abstract_ids if provided."""
    if isinstance(db, (str, Path)):
        with open(db, "r", encoding="utf-8") as f:
            db = json.load(f)

    allowed_set = set(str(x) for x in allowed_abstract_ids) if allowed_abstract_ids is not None else None

    embeddings = []
    records = []
    seen_embeddings = set()

    def add_if_not_exists(emb: Any, rec: Dict[str, Any]) -> None:
        if emb is None:
            return
        emb_arr = np.asarray(emb, dtype=np.float32)
        # Sample 8 points for fast tuple hashing
        if len(emb_arr) >= 8:
            emb_key = tuple(np.round(emb_arr[::len(emb_arr)//8][:8], 4))
        else:
            emb_key = tuple(np.round(emb_arr, 4))
        if emb_key not in seen_embeddings:
            seen_embeddings.add(emb_key)
            embeddings.append(emb_arr)
            records.append(rec)

    if isinstance(db, dict):
        for abstract_id, abstract_data in db.items():
            if allowed_set is not None and str(abstract_id) not in allowed_set:
                continue
            if isinstance(abstract_data, dict):
                for gt in abstract_data.get("ground_truth", []):
                    if "embedding" in gt:
                        add_if_not_exists(
                            gt["embedding"],
                            {
                                "text": gt.get("text"),
                                "label": gt.get("label"),
                                "source": "ground_truth",
                                "abstract_id": abstract_id,
                            },
                        )
                for pred in abstract_data.get("predicted", []):
                    if "embedding" in pred:
                        add_if_not_exists(
                            pred["embedding"],
                            {
                                "text": pred.get("text"),
                                "label": pred.get("label"),
                                "source": "predicted",
                                "confidence": pred.get("confidence"),
                                "abstract_id": abstract_id,
                            },
                        )
            elif isinstance(abstract_data, (list, np.ndarray)):
                add_if_not_exists(abstract_data, {"abstract_id": abstract_id})

    elif isinstance(db, list):
        for idx, item in enumerate(db):
            if isinstance(item, dict) and "embedding" in item:
                rec = {k: v for k, v in item.items() if k != "embedding"}
                rec["index"] = idx
                add_if_not_exists(item["embedding"], rec)
            else:
                add_if_not_exists(item, {"index": idx})

    elif isinstance(db, np.ndarray):
        for i, row in enumerate(db):
            add_if_not_exists(row, {"index": i})

    matrix = np.asarray(embeddings, dtype=np.float32) if embeddings else np.empty((0, 0), dtype=np.float32)
    return matrix, records


def _fit_logistic(d: np.ndarray) -> Tuple[float, float]:
    """Fits (mu, sigma) so logistic mapping spreads over reference distances."""
    mu = float(np.mean(d))
    sigma = float(np.std(d))
    if sigma < 1e-6:
        sigma = 1.0
    return mu, sigma


def _logistic_map(d: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Maps arbitrary distance/LOF score linearly through logistic curve into [0, 1]."""
    z = (d - mu) / max(sigma, 1e-9)
    # Clip z for numerical stability
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


# ==============================================================================
# 3. LOF Novelty Detector Class
# ==============================================================================

class LOFNoveltyDetector:
    """
    Lightweight Vector Novelty & Anomaly Detector with Scikit-Learn LocalOutlierFactor
    and K-Nearest Neighbor Cosine Distance.
    """

    def __init__(
        self,
        vector_db: Union[str, Path, dict, list, np.ndarray],
        k: int = 10,
        alpha: float = 0.5,
        embedder: Any = None,
        allowed_abstract_ids: Optional[Union[Set[str], List[str]]] = None,
    ):
        self.k = k
        self.alpha = alpha
        self.embedder = embedder or get_default_embedder()

        # Extract reference matrix and metadata records (strictly filtered to allowed train doc IDs)
        self.matrix, self.records = extract_embeddings_and_records(
            vector_db,
            allowed_abstract_ids=allowed_abstract_ids
        )
        if self.matrix.size == 0 or len(self.matrix) == 0:
            raise ValueError("Vector database is empty or no valid embeddings found.")

        # Unit-normalize matrix for fast vectorized cosine distance calculations
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        self.norm_matrix = (self.matrix / norms).astype(np.float32)

        # Fit scikit-learn LOF model with novelty=True
        k_neighbors = min(self.k, max(1, len(self.matrix) - 1))
        self.lof_model = LocalOutlierFactor(n_neighbors=k_neighbors, metric="cosine", novelty=True)
        self.lof_model.fit(self.matrix)

        # Fit logistic calibration parameters on reference set
        ref_raw_scores = self._score_combined_raw(self.matrix, alpha=self.alpha)
        self._mu, self._sigma = _fit_logistic(ref_raw_scores)
        self.fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts using the associated embedder."""
        return self.embedder.embed(texts)

    def _prepare_inputs(self, input_embeddings: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 2D unit-normalized vectors [N, D] and 2D original vectors."""
        vecs = np.asarray(input_embeddings, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        norm_vecs = (vecs / norms).astype(np.float32)
        return norm_vecs, vecs

    def score_knn_distance(self, input_embeddings: Union[List[float], np.ndarray]) -> np.ndarray:
        """Method 1: Average cosine distance from the k-nearest neighbors (per entity)."""
        norm_vecs, _ = self._prepare_inputs(input_embeddings)
        # Matrix multiplication: [N, D] x [D, M] -> [N, M] similarities
        similarities = np.dot(norm_vecs, self.norm_matrix.T)
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        
        k_val = min(self.k, distances.shape[1])
        # Partition top-k smallest distances along axis=1
        top_k_distances = np.partition(distances, k_val - 1, axis=1)[:, :k_val]
        return np.mean(top_k_distances, axis=1).astype(np.float32)

    def score_lof(self, input_embeddings: Union[List[float], np.ndarray]) -> np.ndarray:
        """Method 2: LOF novelty score using scikit-learn (per entity)."""
        import warnings
        _, vecs = self._prepare_inputs(input_embeddings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                neg_lof = self.lof_model.score_samples(vecs)
        return (-neg_lof).astype(np.float32)

    def _score_combined_raw(
        self,
        input_embeddings: Union[List[float], np.ndarray],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Uncalibrated linear combination of KNN distance and LOF score."""
        a = self.alpha if alpha is None else alpha
        knn_scores = self.score_knn_distance(input_embeddings)
        lof_scores = self.score_lof(input_embeddings)
        return (a * knn_scores + (1.0 - a) * lof_scores).astype(np.float32)

    def score_combined(
        self,
        input_embeddings: Union[List[float], np.ndarray],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Returns per-entity combined uncalibrated scores."""
        return self._score_combined_raw(input_embeddings, alpha=alpha)

    def novelty(self, input_embeddings: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Calibrated Novelty score in [0, 1] for each individual entity (higher = more novel/rare).
        """
        if isinstance(input_embeddings, list) and len(input_embeddings) == 0:
            return np.array([], dtype=np.float32)
        raw_combined = self._score_combined_raw(input_embeddings, alpha=self.alpha)
        calibrated = _logistic_map(raw_combined, self._mu, self._sigma)
        return calibrated.astype(np.float32)

    def score(
        self,
        input_embedding: Union[List[float], np.ndarray],
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculates all 3 scores for a single input embedding."""
        norm_vec, vec = self._prepare_inputs(input_embedding)
        knn_score = float(self.score_knn_distance(vec)[0])
        lof_score = float(self.score_lof(vec)[0])
        comb_score = float(self._score_combined_raw(vec, alpha=alpha)[0])
        nov_score = float(self.novelty(vec)[0])

        return {
            "knn_distance_score": round(knn_score, 4),
            "lof_score": round(lof_score, 4),
            "combined_score": round(comb_score, 4),
            "novelty_calibrated": round(nov_score, 4),
        }

    def get_representation(
        self,
        input_embedding: Union[List[float], np.ndarray],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Finds what an embedding represents by returning the top-k nearest matching texts and labels.
        """
        norm_vec, _ = self._prepare_inputs(input_embedding)
        similarities = np.dot(self.norm_matrix, norm_vec[0])
        actual_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:actual_k]

        matches = []
        for rank, idx in enumerate(top_indices, 1):
            rec = self.records[idx].copy() if idx < len(self.records) else {}
            sim = float(similarities[idx])
            rec["rank"] = rank
            rec["similarity"] = round(sim, 4)
            rec["distance"] = round(float(np.clip(1.0 - sim, 0.0, 2.0)), 4)
            matches.append(rec)
        return matches

    def get_contrastive_exemplars(
        self,
        input_embedding: Union[List[float], np.ndarray],
        target_classes: List[str],
        top_k_per_class: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the closest training exemplars strictly belonging to the specified target_classes.
        Useful for contrastive disambiguation between competing Top-1 and Top-2 categories.
        """
        norm_vec, _ = self._prepare_inputs(input_embedding)
        similarities = np.dot(self.norm_matrix, norm_vec[0])

        def norm_cat(c: str) -> str:
            c = str(c or "").lower().strip().replace("-", "_").replace(" ", "_")
            if c in ("drugs", "substance_use"):
                return "substance_use"
            if c in ("beauty_and_cleaning", "personal_care_products_and_cosmetic_procedures"):
                return "personal_care"
            if c in ("physical_activity", "physical_activities"):
                return "physical_activities"
            return c

        exemplars = []
        for cls in target_classes:
            target_norm = norm_cat(cls)
            matching_indices = [
                i for i, r in enumerate(self.records)
                if norm_cat(r.get("label", "")) == target_norm or target_norm in norm_cat(r.get("label", ""))
            ]
            if matching_indices:
                cls_sims = similarities[matching_indices]
                best_order = np.argsort(cls_sims)[::-1][:top_k_per_class]
                for rank_idx in best_order:
                    orig_idx = matching_indices[rank_idx]
                    rec = self.records[orig_idx].copy()
                    sim = float(similarities[orig_idx])
                    exemplars.append({
                        "target_class": cls,
                        "matched_term": rec.get("text", ""),
                        "similarity": round(sim, 4),
                        "gold_label": rec.get("label", cls),
                    })
        return exemplars

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


# ==============================================================================
# 4. Builder Function
# ==============================================================================

def build_lof_novelty_detector(
    vector_db_path: str = None,
    dataset_path: str = None,
    embedder: Any = None,
    device: str = None,
    k: int = 10,
    alpha: float = 0.5,
    filter_train_only: bool = True,
) -> LOFNoveltyDetector:
    """
    Constructs and fits a LOFNoveltyDetector using get_default_embedder() (Qwen 8B -> BGE-M3 -> Local RoBERTa) by default.
    Strictly filters reference vectors to training split only (Zero Test/Dev Leakage).
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    embedder = embedder or get_default_embedder(device=device)

    # Resolve dataset path with fallback candidates
    candidate_ds_paths = [
        dataset_path,
        os.path.join(PROJECT_ROOT, "data/processed/spanner_dataset_10class.json"),
        os.path.join(PROJECT_ROOT, "data/processed/spanner_dataset.json"),
        os.path.join(PROJECT_ROOT, "data/Extra/processed/spanner_dataset.json"),
    ]
    ds_path = None
    for p in candidate_ds_paths:
        if p and os.path.exists(p):
            ds_path = p
            break

    train_ids = None
    if filter_train_only and ds_path and os.path.exists(ds_path):
        with open(ds_path, "r", encoding="utf-8") as f:
            ds_data = json.load(f)
        train_ids = set(d["doc_id"] for d in ds_data.get("train", []))

    # Case A: If pre-extracted vector database exists and dimension matches embedder
    candidate_vdb_paths = [
        vector_db_path,
        os.path.join(PROJECT_ROOT, "data/vector_db/lsf_entity_vector_db.json"),
        os.path.join(PROJECT_ROOT, "data/processed/vector_db/lsf_entity_vector_db.json"),
    ]
    vdb_path = None
    for p in candidate_vdb_paths:
        if p and os.path.exists(p):
            vdb_path = p
            break

    if vdb_path and os.path.exists(vdb_path):
        try:
            # Check if vector db dimension matches embedder (and filter strictly to training abstracts)
            detector = LOFNoveltyDetector(
                vector_db=vdb_path,
                k=k,
                alpha=alpha,
                embedder=embedder,
                allowed_abstract_ids=train_ids
            )
            sample_emb = embedder.embed(["test entity"])
            if sample_emb.shape[1] == detector.matrix.shape[1]:
                return detector
        except Exception:
            pass

    # Case B: Build reference embeddings directly from dataset train spans with the active embedder
    if not ds_path or not os.path.exists(ds_path):
        raise FileNotFoundError(f"Neither vector DB ({vdb_path or candidate_vdb_paths[1]}) nor dataset ({ds_path or candidate_ds_paths[1]}) found.")


    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ent_texts = []
    records = []
    seen_texts = set()

    for doc in data.get("train", []):
        abs_id = doc.get("doc_id", "train_doc")
        for s in doc.get("spans", []):
            t = s.get("text", "").strip()
            lbl = s.get("label", "Entity")
            if t and t.lower() not in seen_texts:
                seen_texts.add(t.lower())
                ent_texts.append(t)
                records.append({
                    "text": t,
                    "label": lbl,
                    "abstract_id": abs_id,
                    "source": "ground_truth"
                })

    if not ent_texts:
        raise ValueError(f"No valid entity texts found in {ds_path}")

    # Embed all unique training entities in batch using active embedder
    embeddings = embedder.embed(ent_texts)
    
    # Construct database dictionary
    db = []
    for emb, rec in zip(embeddings, records):
        rec_copy = rec.copy()
        rec_copy["embedding"] = emb
        db.append(rec_copy)

    return LOFNoveltyDetector(vector_db=db, k=k, alpha=alpha, embedder=embedder)
