import os
import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional, Union

class BulletOps:
    def __init__(self, store, vectors):
        self.store = store
        self.vectors = vectors
        
        self.bullets_added_since_pruning = 0
        self.bullets_before_pruning = 50
        self.pruning_score_threshold = 5 
        self.processing_history = {}

    def add_guideline(self, guideline_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds a new guideline adhering to the rich Dynamic Guidebook schema.
        Syncs into ChromaDB with Composite Triad embedding.
        """
        gid = self.store.add_or_update(guideline_dict)
        
        # Sync with ChromaDB
        if self.vectors and self.vectors.collection:
            try:
                doc_text = self.vectors.construct_embedding_text(guideline_dict)
                meta = self.vectors._make_metadata(guideline_dict)
                emb = self.vectors.model.encode(doc_text).tolist() if self.vectors.model else None
                if emb:
                    self.vectors.collection.add(
                        ids=[gid],
                        embeddings=[emb],
                        documents=[doc_text],
                        metadatas=[meta]
                    )
            except Exception as e:
                print(f"Warning: Failed to add guideline {gid} to ChromaDB: {e}")
                
        self.store.save()
        self.bullets_added_since_pruning += 1
        return {"success": True, "id": gid, "guideline": guideline_dict}

    def modify_guideline(
        self,
        guideline_id: str,
        new_rule_text: str,
        reason: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Updates rule content of an existing guideline while preserving modification history."""
        g = self.store.get_by_id(guideline_id)
        if not g:
            return {"success": False, "message": f"Guideline {guideline_id} not found."}

        old_rule = g.get("guideline", g.get("content", ""))
        g.setdefault("history", []).append({
            "previous_guideline": old_rule,
            "modified_in_file": file_name or "unknown",
            "reason": reason or "Refinement based on Reflection analysis",
            "timestamp": datetime.now().isoformat()
        })

        g["guideline"] = new_rule_text
        if "content" in g:
            g["content"] = new_rule_text

        metrics = g.setdefault("usage_metrics", {
            "helpful": 0, "harmful": 0, "neutral": 0, "usage_count": 0, "modification_count": 0
        })
        metrics["modification_count"] = metrics.get("modification_count", 0) + 1

        # Re-embed in ChromaDB
        if self.vectors and self.vectors.collection:
            try:
                doc_text = self.vectors.construct_embedding_text(g)
                meta = self.vectors._make_metadata(g)
                emb = self.vectors.model.encode(doc_text).tolist() if self.vectors.model else None
                if emb:
                    self.vectors.collection.update(
                        ids=[guideline_id],
                        embeddings=[emb],
                        documents=[doc_text],
                        metadatas=[meta]
                    )
            except Exception as e:
                print(f"Warning: Failed to update guideline {guideline_id} in ChromaDB: {e}")

        self.store.save()
        return {"success": True, "id": guideline_id, "modification_count": metrics["modification_count"]}

    def update_bullet_metrics(
        self,
        bullet_ids: Union[str, List[str]],
        *,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
        neutral_delta: int = 0,
        usage_delta: int = 0
    ) -> Dict[str, Any]:
        """Unified method for updating guideline metrics in JSON and ChromaDB."""
        if not bullet_ids:
            return {"success": True, "updated_count": 0}

        from collections import Counter
        if isinstance(bullet_ids, str):
            id_counts = {bullet_ids: 1}
        elif isinstance(bullet_ids, dict):
            id_counts = {str(k): int(v) for k, v in bullet_ids.items() if k and v}
        elif isinstance(bullet_ids, (list, tuple)):
            flat_ids = []
            for item in bullet_ids:
                if isinstance(item, (list, tuple)):
                    flat_ids.extend([str(x) for x in item if x])
                elif item:
                    flat_ids.append(str(item))
            id_counts = Counter(flat_ids)
        else:
            id_counts = {str(bullet_ids): 1}

        updated_count = 0
        updated_metadatas = []
        updated_ids = []

        for gid, count in id_counts.items():
            g = self.store.get_by_id(gid)
            if g:
                metrics = g.setdefault("usage_metrics", {
                    "helpful": 0, "harmful": 0, "neutral": 0, "usage_count": 0, "modification_count": 0
                })
                if helpful_delta:
                    metrics["helpful"] = metrics.get("helpful", 0) + (helpful_delta * count)
                if harmful_delta:
                    metrics["harmful"] = metrics.get("harmful", 0) + (harmful_delta * count)
                if neutral_delta:
                    metrics["neutral"] = metrics.get("neutral", 0) + (neutral_delta * count)
                if usage_delta:
                    metrics["usage_count"] = metrics.get("usage_count", 0) + (usage_delta * count)

                updated_count += count
                if self.vectors:
                    updated_ids.append(gid)
                    updated_metadatas.append(self.vectors._make_metadata(g))

        if updated_ids and self.vectors and self.vectors.collection:
            try:
                self.vectors.collection.update(
                    ids=updated_ids,
                    metadatas=updated_metadatas
                )
            except Exception as e:
                print(f"Warning: Failed to update ChromaDB metrics: {e}")

        if updated_count > 0:
            self.store.save()

        return {"success": True, "updated_count": updated_count}

    def process_curator_operations(
        self,
        curator_output: Dict[str, Any],
        similarity_threshold: float = 0.8,
        file_name: Optional[str] = None,
        abstract: Optional[str] = None,
        reflection: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parses Curator output operations (ADD, MODIFY, PRUNE) and executes them.
        """
        operations = curator_output.get("operations", [])
        results = {"added": [], "modified": [], "rejected": [], "errors": []}

        for op in operations:
            op_type = op.get("type", "").upper()

            if op_type == "ADD":
                # Build rich guideline payload
                rule_text = op.get("guideline") or op.get("content") or ""
                span = op.get("span") or op.get("erroneous_token") or ""
                gt_label = op.get("ground_truth_label") or op.get("gt_label") or ""
                if gt_label.lower() in ("annotation_fundamentals", "context_scope_handling", "entity_relationship_rules", "general", "general_ner"):
                    gt_label = "O"
                bert_label = op.get("bert_label") or ""
                llm_label = op.get("llm_label") or ""
                
                # Check for semantic duplicate in vector store before adding
                if self.vectors and rule_text:
                    similar = self.vectors.find_similar(rule_text, n_results=1, threshold=similarity_threshold)
                    if similar:
                        results["rejected"].append({
                            "rule": rule_text,
                            "reason": f"Semantically similar to existing guideline {similar[0]['id']} (sim={similar[0]['similarity']:.2f})"
                        })
                        continue

                similar_spans = op.get("similar_spans", [])
                for s in similar_spans:
                    if isinstance(s, dict):
                        lbl_val = s.get("ground_truth_label", s.get("label", ""))
                        if lbl_val.lower() in ("annotation_fundamentals", "context_scope_handling", "entity_relationship_rules", "general", "general_ner"):
                            lbl_val = "O"
                        s["ground_truth_label"] = lbl_val
                        if "label" in s:
                            del s["label"]

                new_guideline = {
                    "id": op.get("id") or self.store.next_id(),
                    "span": span,
                    "span_sentence": op.get("span_sentence") or op.get("sentence") or "",
                    "ground_truth_label": gt_label,
                    "bert_label": bert_label,
                    "llm_label": llm_label,
                    "triad": op.get("triad", {
                        "type": "CURATOR_SYNTHESIZED",
                        "description": op.get("reason", "Curator generated guideline")
                    }),
                    "guideline": rule_text,
                    "challenge_types": op.get("challenge_types", ["DISAMBIGUATION_RULE"]),
                    "similar_spans": similar_spans,
                    "usage_metrics": {
                        "helpful": 0, "harmful": 0, "neutral": 0, "usage_count": 0, "modification_count": 0
                    },
                    "history": []
                }
                add_res = self.add_guideline(new_guideline)
                results["added"].append(add_res["id"])

            elif op_type == "MODIFY":
                gid = op.get("id") or op.get("bullet_id")
                new_rule = op.get("guideline") or op.get("content") or op.get("new_content")
                if gid and new_rule:
                    mod_res = self.modify_guideline(gid, new_rule, reason=op.get("reason"), file_name=file_name)
                    if mod_res["success"]:
                        results["modified"].append(gid)
                    else:
                        results["errors"].append(mod_res)
                else:
                    results["errors"].append({"operation": op, "message": "Missing ID or guideline text for MODIFY"})

        return results

    # Legacy compatibility wrappers
    def add_bullet(self, category, content, file_name=None, root_cause="", error_examples=None):
        return self.add_guideline({
            "span": error_examples[0].get("erroneous_token", "") if error_examples else "",
            "span_sentence": error_examples[0].get("sentence", "") if error_examples else "",
            "ground_truth_label": category,
            "guideline": content,
            "triad": {"type": "LEGACY_ADD", "description": root_cause}
        })

    def modify_bullet(self, bullet_id, new_content, file_name=None, reason=None, error_examples=None):
        return self.modify_guideline(bullet_id, new_content, reason=reason, file_name=file_name)
