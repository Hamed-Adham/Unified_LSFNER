import os
import torch
from typing import Dict, List, Optional, Any, Union
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import chromadb
except ImportError:
    chromadb = None


class VectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = str(persist_directory)
        self.backup_directory = str(os.path.join(self.persist_directory, "backup"))
        
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(self.backup_directory, exist_ok=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if SentenceTransformer:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print(f"Initialized SentenceTransformer on device: {device}")
        else:
            self.model = None
            print("Warning: SentenceTransformer not available.")
            
        if chromadb:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self._init_collection()
        else:
            self.chroma_client = None
            self.collection = None
            print("Warning: ChromaDB not available.")

    def _init_collection(self):
        COLLECTION_NAME = "dynamicGuidebook_collection"
        try:
            collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing ChromaDB collection: {COLLECTION_NAME}")
            return collection
        except Exception:
            collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new ChromaDB collection: {COLLECTION_NAME}")
            return collection

    @staticmethod
    def construct_embedding_text(guideline: Dict[str, Any]) -> str:
        """
        Builds the Composite Triad representation:
        Span + Sentence Context + Guideline Text
        """
        span = guideline.get("span", "")
        sentence = guideline.get("span_sentence", "")
        rule = guideline.get("guideline", guideline.get("content", ""))
        
        parts = []
        if span:
            parts.append(f"Span: {span}")
        if sentence:
            parts.append(f"Sentence: {sentence}")
        if rule:
            parts.append(f"Guideline: {rule}")
            
        return " | ".join(parts) if parts else str(guideline)

    def _make_metadata(self, guideline: Dict[str, Any]) -> Dict[str, Any]:
        gid = guideline.get("id", guideline.get("bullet_id", ""))
        ch_types = guideline.get("challenge_types", [])
        if isinstance(ch_types, list):
            ch_str = ",".join(str(c) for c in ch_types)
        else:
            ch_str = str(ch_types or "")

        metrics = guideline.get("usage_metrics", {})
        rule_text = str(guideline.get("guideline", guideline.get("content", "")))
        return {
            "id": str(gid),
            "bullet_id": str(gid),
            "span": str(guideline.get("span", "")),
            "ground_truth_label": str(guideline.get("ground_truth_label", "")),
            "bert_label": str(guideline.get("bert_label", "")),
            "guideline": rule_text,
            "challenge_types": ch_str,
            "helpful": int(metrics.get("helpful", guideline.get("helpful", 0))),
            "harmful": int(metrics.get("harmful", guideline.get("harmful", 0))),
            "usage_count": int(metrics.get("usage_count", guideline.get("usage_count", 0)))
        }

    def sync_with_store(self, store):
        """
        Bi-directional sync between dynamicGuidebook.json and ChromaDB:
        1. Remove ChromaDB entries that no longer exist in JSON
        2. Add embeddings for JSON guidelines missing from ChromaDB
        """
        if not self.collection or not self.model:
            return

        json_guidelines_map = {}
        for g in store.iter_guidelines():
            gid = str(g.get("id", g.get("bullet_id", "")))
            if gid:
                json_guidelines_map[gid] = g

        json_ids = set(json_guidelines_map.keys())

        try:
            chroma_ids = set(self.collection.get(include=[])['ids'])
        except Exception:
            chroma_ids = set()

        # 1. Remove stale ChromaDB entries
        stale_ids = chroma_ids - json_ids
        if stale_ids:
            self.collection.delete(ids=list(stale_ids))
            print(f"Removed {len(stale_ids)} stale entries from ChromaDB.")

        # 2. Add missing embeddings
        missing_ids = json_ids - chroma_ids
        if missing_ids:
            documents, metadatas, ids = [], [], []
            for gid in missing_ids:
                g = json_guidelines_map[gid]
                doc_text = self.construct_embedding_text(g)
                documents.append(doc_text)
                metadatas.append(self._make_metadata(g))
                ids.append(gid)

            embeddings = self.model.encode(documents, batch_size=32).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(ids)} missing embeddings to ChromaDB.")

        if not stale_ids and not missing_ids:
            print("ChromaDB and dynamicGuidebook are in sync.")

    def find_similar(
        self,
        query: Union[str, List[float]],
        category: Optional[str] = None,
        challenge_type: Optional[str] = None,
        n_results: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for similar guidelines based on Composite Triad similarity.
        """
        if not self.collection or not self.model:
            return []

        if isinstance(query, str):
            query_embedding = self.model.encode(query).tolist()
        else:
            query_embedding = query

        where_filter = {}
        if category:
            where_filter["ground_truth_label"] = category
        if challenge_type:
            where_filter["challenge_types"] = challenge_type
            
        where = where_filter if where_filter else None

        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)

        query_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances", "documents"]
        )

        results_list = []
        if query_results["ids"] and query_results["ids"][0]:
            for i in range(len(query_results["ids"][0])):
                gid = query_results["ids"][0][i]
                doc = query_results["documents"][0][i]
                meta = query_results["metadatas"][0][i]
                dist = query_results["distances"][0][i]
                similarity = float(1.0 - dist)
                
                if threshold is not None and similarity < threshold:
                    continue

                results_list.append({
                    "id": gid,
                    "bullet_id": gid,
                    "span": meta.get("span", ""),
                    "ground_truth_label": meta.get("ground_truth_label", ""),
                    "bert_label": meta.get("bert_label", ""),
                    "guideline": meta.get("guideline", ""),
                    "content": meta.get("guideline", ""),
                    "challenge_types": meta.get("challenge_types", "").split(",") if meta.get("challenge_types") else [],
                    "document": doc,
                    "similarity": similarity
                })

        return results_list
