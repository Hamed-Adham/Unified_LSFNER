import os
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import chromadb
except ImportError:
    chromadb = None


class VectorStore:
    def __init__(self, persist_directory):
        self.persist_directory = persist_directory
        self.backup_directory = str(os.path.join(self.persist_directory, "backup"))
        
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(self.backup_directory, exist_ok=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"Initialized SentenceTransformer on device: {device}")
        
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._init_collection()

    def _init_collection(self):
        COLLECTION_NAME = "dynamicGuidelines_bullets"
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

    def _make_metadata(self, bullet_id, section_name, supercategory, helpful=0, harmful=0, usage_count=0, modification_count=0):
        return {
            "bullet_id": bullet_id, "section_name": section_name, "supercategory": supercategory,
            "helpful": helpful, "harmful": harmful, "usage_count": usage_count, "modification_count": modification_count
        }

    def sync_with_store(self, store):
        """
        Bi-directional sync between dynamicGuidelines.json and ChromaDB:
        1. Remove ChromaDB entries that no longer exist in JSON
        2. Add embeddings for JSON bullets missing from ChromaDB
        """
        # Get all bullet IDs from JSON
        json_bullet_ids = set()
        json_bullets_map = {}  # bullet_id -> (bullet, supercategory, section_name)
        for bullet, supercategory, section_name in store.iter_bullets():
            bullet_id = bullet["bullet_id"]
            json_bullet_ids.add(bullet_id)
            json_bullets_map[bullet_id] = (bullet, supercategory, section_name)
        
        # Get all bullet IDs from ChromaDB
        try:
            chroma_ids = set(self.collection.get(include=[])['ids'])
        except Exception:
            chroma_ids = set()
        
        # 1. Remove stale ChromaDB entries (in ChromaDB but not in JSON)
        stale_ids = chroma_ids - json_bullet_ids
        if stale_ids:
            self.collection.delete(ids=list(stale_ids))
            print(f"Removed {len(stale_ids)} stale entries from ChromaDB.")
        
        # 2. Add missing embeddings (in JSON but not in ChromaDB)
        missing_ids = json_bullet_ids - chroma_ids
        if missing_ids:
            contents, metadatas, ids = [], [], []
            for bullet_id in missing_ids:
                bullet, supercategory, section_name = json_bullets_map[bullet_id]
                contents.append(bullet["content"])
                metadatas.append(self._make_metadata(
                    bullet_id, section_name, supercategory,
                    bullet.get("helpful", 0), bullet.get("harmful", 0), bullet.get("usage_count", 0)
                ))
                ids.append(bullet_id)
            
            self.collection.add(
                embeddings=self.model.encode(contents, batch_size=32).tolist(),
                documents=contents, metadatas=metadatas, ids=ids
            )
            print(f"Added {len(ids)} missing embeddings to ChromaDB.")
        
        if not stale_ids and not missing_ids:
            print("ChromaDB and dynamicGuidelines.json are in sync.")

    def find_similar(self, query, section_name=None, n_results=5, threshold=None):
        """
        Query ChromaDB for similar bullets.
        query: can be a string (text) or a precomputed embedding (list/numpy array).
        section_name: optional filter for a specific section.
        n_results: number of results to retrieve.
        threshold: if set, will also check if the top result meets the similarity threshold,
                   returning a dict with 'is_similar', 'similarity', 'similar_bullet', etc.
        """
        if isinstance(query, str):
            query_embedding = self.model.encode(query).tolist()
        else:
            query_embedding = query

        where_filter = {"section_name": section_name} if section_name else None
        
        count = self.collection.count()
        if count == 0:
            if threshold is not None:
                return {"is_similar": False, "similarity": 0.0, "similar_bullet": None}
            return []
            
        n_results = min(n_results, count)
        
        query_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "distances", "documents"]
        )
        
        results_list = []
        if query_results["ids"] and query_results["ids"][0]:
            for i in range(len(query_results["ids"][0])):
                bullet_id = query_results["ids"][0][i]
                doc = query_results["documents"][0][i]
                meta = query_results["metadatas"][0][i]
                dist = query_results["distances"][0][i]
                similarity = float(1 - dist)
                results_list.append({
                    "bullet_id": bullet_id,
                    "content": doc,
                    "section_name": meta.get("section_name"),
                    "supercategory": meta.get("supercategory"),
                    "helpful": meta.get("helpful", 0),
                    "harmful": meta.get("harmful", 0),
                    "usage_count": meta.get("usage_count", 0),
                    "modification_count": meta.get("modification_count", 0),
                    "similarity": similarity
                })
        
        if threshold is not None:
            is_similar = False
            top_similarity = 0.0
            similar_bullet = None
            if results_list:
                top_similarity = results_list[0]["similarity"]
                if top_similarity >= threshold:
                    is_similar = True
                    similar_bullet = {
                        "bullet_id": results_list[0]["bullet_id"],
                        "content": results_list[0]["content"]
                    }
            return {
                "is_similar": is_similar,
                "similarity": top_similarity,
                "similar_bullet": similar_bullet
            }
            
        return results_list
