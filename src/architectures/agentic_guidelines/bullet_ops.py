import os
import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter

class BulletOps:
    def __init__(self, store, vectors):
        self.store = store
        self.vectors = vectors
        
        self.bullets_added_since_pruning = 0
        self.bullets_before_pruning = 50
        self.pruning_score_threshold = 5 
        self.processing_history = {}

    @staticmethod
    def filter_and_cap_error_examples_static(error_examples, max_limit=4):
        """
        Filters and caps error examples to max_limit (default 4).
        Tries to ensure representation of unique error types (e.g. FN, FP, BIO_ERR, CLSS_ERR).
        Groups/deduplicates by sentence (case-insensitive strip).
        Prioritizes recency (newer items are at the end of the input list).
        """
        if not error_examples:
            return []
            
        # Deduplicate by sentence to avoid exact duplicate sentences
        seen_sentences = set()
        unique_examples = []
        # Loop backwards (from most recent to oldest) to preserve the most recent ones first
        for ex in reversed(error_examples):
            sent_clean = ex.get("sentence", "").strip().lower()
            if sent_clean and sent_clean not in seen_sentences:
                seen_sentences.add(sent_clean)
                unique_examples.append(ex)
                
        # Now we have unique examples, sorted from most recent to oldest.
        # Group by error type
        by_type = {}
        for ex in unique_examples:
            err_type = ex.get("error_type", "unknown").upper()
            by_type.setdefault(err_type, []).append(ex)
            
        selected = []
        # 1. Take the most recent example from each error type
        sorted_types = sorted(by_type.keys())
        for etype in sorted_types:
            if len(selected) >= max_limit:
                break
            selected.append(by_type[etype].pop(0))
            
        # 2. Fill the remaining slots using remaining examples, ordered by recency
        remaining = []
        for etype, items in by_type.items():
            remaining.extend(items)
            
        # Sort remaining by their index in the unique_examples list
        remaining.sort(key=lambda x: unique_examples.index(x))
        
        for ex in remaining:
            if len(selected) >= max_limit:
                break
            selected.append(ex)
            
        # Finally, sort the selected examples back in chronological order (oldest to newest)
        selected.reverse()
        return selected

    def filter_and_cap_error_examples(self, error_examples, max_limit=4):
        return self.filter_and_cap_error_examples_static(error_examples, max_limit)

    def parse_reflection_for_errors(self, abstract, reflection, file_name):
        """
        Parses the error_identification markdown table from reflection,
        splits the abstract into sentences, and matches erroneous tokens
        to their containing sentences.
        """
        if not abstract or not reflection:
            return []
            
        table = reflection.get("error_identification", "")
        if not table or not isinstance(table, str):
            return []
            
        # 1. Split abstract into sentences
        sentences = re.split(r'(?<=[\.\!\?])\s+', abstract.strip())
        
        # 2. Parse tokens and error types from markdown table
        tokens_and_types = []
        for line in table.split("\n"):
            if not line.strip() or not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                token = parts[1]
                err_type = parts[2]
                if token and not token.startswith("-") and token != "Erroneous Token" and token != "Token":
                    tokens_and_types.append({"token": token, "type": err_type})
                    
        # 3. Match tokens to sentences
        error_examples = []
        for item in tokens_and_types:
            token = item["token"]
            matched_sentence = None
            for sentence in sentences:
                pattern = r'\b' + re.escape(token) + r'\b'
                if re.search(pattern, sentence, re.IGNORECASE):
                    matched_sentence = sentence
                    break
            
            if not matched_sentence:
                for sentence in sentences:
                    if token.lower() in sentence.lower():
                        matched_sentence = sentence
                        break
                        
            if matched_sentence:
                error_examples.append({
                    "file_name": file_name or "unknown",
                    "erroneous_token": token,
                    "error_type": item["type"],
                    "sentence": matched_sentence
                })
        return self.filter_and_cap_error_examples(error_examples)

    def modify_bullet(self, bullet_id, new_content, file_name=None, reason=None, error_examples=None):
        """Updates bullet content while preserving history. Uses O(1) prefix-based section lookup."""
        section_name = self.store.get_section_for_bullet_id(bullet_id)
        if not section_name:
            return {"success": False, "message": f"Unknown prefix in bullet ID: {bullet_id}"}
        
        supercategory = self.store.section_to_supercategory[section_name]
        bullets = self.store.dynamicGuidelines["dynamicGuidelines_sections"][supercategory][section_name]
        
        bullet = next((b for b in bullets if b.get("bullet_id") == bullet_id), None)
        if not bullet:
            return {"success": False, "message": f"Bullet ID {bullet_id} not found in section {section_name}"}
        
        bullet.setdefault("modification_count", 0)
        bullet.setdefault("content_history", [])
        bullet.setdefault("history", [])
        
        # Update JSON structure
        bullet["content_history"].append(bullet["content"])
        bullet["history"].append({
            "previous_content": bullet["content"],
            "modified_in_file": file_name or "unknown",
            "reason_for_modification": reason or "Modified by Curator based on Reflection analysis"
        })
        
        bullet["content"] = new_content
        bullet["modification_count"] += 1
        
        if error_examples:
            dc = bullet.setdefault("diagnostic_context", {})
            existing_errs = dc.setdefault("error_examples", [])
            merged = existing_errs + error_examples
            dc["error_examples"] = self.filter_and_cap_error_examples(merged)
        
        # Update ChromaDB (Embeddings MUST be recalculated)
        embedding = self.vectors.model.encode(new_content).tolist()
        self.vectors.collection.update(
            ids=[bullet_id],
            embeddings=[embedding],
            documents=[new_content],
            metadatas=[self.vectors._make_metadata(
                bullet_id, section_name, supercategory,
                bullet.get("helpful", 0), bullet.get("harmful", 0),
                bullet.get("usage_count", 0), bullet["modification_count"]
            )]
        )
        
        self.store.save()
        return {"success": True, "bullet_id": bullet_id, "modification_count": bullet["modification_count"]}

    def update_bullet_metrics(self, bullet_ids, *, helpful_delta=0, harmful_delta=0, usage_delta=0):
        """Unified method for updating bullet metrics in JSON and ChromaDB."""
        if not bullet_ids:
            return {"success": True, "updated_count": 0}
            
        if isinstance(bullet_ids, str):
            bullet_ids_set = {bullet_ids}
        else:
            bullet_ids_set = set(bullet_ids)
            
        updated_bullets, updated_count = [], 0
        
        for bullet, supercategory, section_name in self.store.iter_bullets():
            bullet_id = bullet.get("bullet_id")
            if bullet_id in bullet_ids_set:
                if helpful_delta:
                    bullet["helpful"] = bullet.get("helpful", 0) + helpful_delta
                if harmful_delta:
                    bullet["harmful"] = bullet.get("harmful", 0) + harmful_delta
                if usage_delta:
                    bullet["usage_count"] = bullet.get("usage_count", 0) + usage_delta
                    
                updated_count += 1
                updated_bullets.append({
                    "id": bullet_id,
                    "metadata": self.vectors._make_metadata(
                        bullet_id, section_name, supercategory,
                        bullet.get("helpful", 0), bullet.get("harmful", 0),
                        bullet.get("usage_count", 0), bullet.get("modification_count", 0)
                    )
                })
                
        if updated_bullets:
            try:
                self.vectors.collection.update(
                    ids=[b["id"] for b in updated_bullets],
                    metadatas=[b["metadata"] for b in updated_bullets]
                )
            except Exception as e:
                print(f"Warning: Failed to update ChromaDB metrics: {e}")
            self.store.save()
            
        return {"success": True, "updated_count": updated_count}

    def process_curator_operations(self, curator_output, similarity_threshold=0.8, file_name=None, abstract=None, reflection=None):
        operations = curator_output.get("operations", [])
        results = {"added": [], "modified": [], "rejected": [], "errors": []}
        op_counts = Counter(op.get("type", "UNKNOWN") for op in operations)
        
        root_cause = reflection.get("root_cause_analysis", "") if reflection else ""
        error_examples = []
        if abstract and reflection:
            error_examples = self.parse_reflection_for_errors(abstract, reflection, file_name)
        
        add_ops = [op for op in operations if op.get("type") == "ADD" and op.get("section") and op.get("content")]
        modify_ops = [op for op in operations if op.get("type") == "MODIFY" and op.get("bullet_id") and (op.get("content") or op.get("new_content"))]
        
        for op in operations:
            if op.get("type") not in ["ADD", "MODIFY"]: 
                results["errors"].append({"operation": op, "message": f"Operation type '{op.get('type')}' not supported."})
            elif op.get("type") == "ADD" and not (op.get("section") and op.get("content")):
                results["errors"].append({"operation": op, "message": "Missing section or content for ADD"})
            elif op.get("type") == "MODIFY" and not (op.get("bullet_id") and (op.get("content") or op.get("new_content"))):
                results["errors"].append({"operation": op, "message": "Missing bullet_id or content for MODIFY"})

        for op in modify_ops:
            new_content = op.get("content") or op.get("new_content")
            reason = op.get("reason") or curator_output.get("reasoning") or "Modified by Curator based on Reflection analysis"
            mod_res = self.modify_bullet(op["bullet_id"], new_content, file_name=file_name, reason=reason, error_examples=error_examples)
            if mod_res["success"]:
                results["modified"].append({
                    "bullet_id": op["bullet_id"], 
                    "new_content": new_content, 
                    "modification_count": mod_res["modification_count"]
                })
            else:
                results["errors"].append({"operation": op, "message": mod_res["message"]})

        def finalize(res_dict):
            history_key = file_name or f"unknown_{len(self.processing_history)}"
            self.processing_history[history_key] = {
                "n_operation": len(operations), "operation_types": dict(op_counts),
                "total_success": len(res_dict["added"]) + len(res_dict["modified"]), 
                "total_rejected": len(res_dict["rejected"]) + len(res_dict["errors"]),
                "added_bullets": res_dict["added"], 
                "modified_bullets": res_dict["modified"], 
                "rejected_bullets": res_dict["rejected"] + res_dict["errors"]
            }
            return res_dict

        if not add_ops:
            return finalize(results)

        embeddings = self.vectors.model.encode([op["content"] for op in add_ops], batch_size=32).tolist()
        chroma_add_data = {"embeddings": [], "documents": [], "metadatas": [], "ids": []}

        for i, op in enumerate(add_ops):
            section_name, content, embedding = op["section"], op["content"], embeddings[i]
            supercategory, section_abbr, canonical_name = self.store.resolve_section(section_name)
            
            if not supercategory:
                results["errors"].append({"operation": op, "message": "Section does not exist"})
                continue

            sim_check = self.vectors.find_similar(embedding, section_name=canonical_name, n_results=1, threshold=similarity_threshold)
            if sim_check["is_similar"]:
                results["rejected"].append({
                    "section": canonical_name, "content": content, "reason": "Too similar",
                    "similar_bullet": sim_check["similar_bullet"]
                })
                continue
            
            new_id = self.store.next_id(section_abbr)
            
            diagnostic_context = {
                "created_from_file": file_name or "unknown",
                "root_cause_of_creation": root_cause,
                "error_examples": error_examples
            }
            
            self.store.dynamicGuidelines["dynamicGuidelines_sections"][supercategory][canonical_name].append({
                "bullet_id": new_id, 
                "helpful": 0, 
                "harmful": 0, 
                "usage_count": 0, 
                "modification_count": 0,
                "content_history": [],
                "content": content,
                "history": [],
                "diagnostic_context": diagnostic_context
            })
            
            chroma_add_data["embeddings"].append(embedding)
            chroma_add_data["documents"].append(content)
            chroma_add_data["ids"].append(new_id)
            chroma_add_data["metadatas"].append(self.vectors._make_metadata(new_id, canonical_name, supercategory))
            
            results["added"].append({"bullet_id": new_id, "section": canonical_name, "content": content, "similarity": sim_check["similarity"]})

        if chroma_add_data["ids"]:
            self.vectors.collection.add(**chroma_add_data)
            self.maybe_prune(len(chroma_add_data["ids"]))

        self.store.save()
        return finalize(results)

    def maybe_prune(self, added_count=1):
        self.bullets_added_since_pruning += added_count
        if self.bullets_added_since_pruning >= self.bullets_before_pruning:
            pruning_res = self.score_based_pruning(score_threshold=self.pruning_score_threshold)
            if pruning_res.get("removed_bullets"):
                print(f"Pruning removed {len(pruning_res['removed_bullets'])} bullet(s).")
            self.bullets_added_since_pruning = 0

    def score_based_pruning(self, section_name=None, score_threshold=None):
        if section_name:
            _, _, section_name = self.store.resolve_section(section_name)
        threshold = score_threshold if score_threshold is not None else self.pruning_score_threshold
        all_bullets = self.vectors.collection.get(
            where={"section_name": section_name} if section_name else None,
            include=["metadatas", "documents"]
        )
        
        removed_ids, removed_info = [], []
        for i, bullet_id in enumerate(all_bullets["ids"]):
            meta = all_bullets["metadatas"][i]
            score = meta.get("harmful", 0) - meta.get("helpful", 0)
            if score >= threshold:
                removed_ids.append(bullet_id)
                removed_info.append({
                    "removed_bullet": {
                        "bullet_id": bullet_id,
                        "content": all_bullets["documents"][i],
                        "helpful": meta.get("helpful"),
                        "harmful": meta.get("harmful")
                    },
                    "section": meta["section_name"],
                    "supercategory": meta["supercategory"],
                    "score": score
                })
        
        if not removed_ids:
            return {"message": f"No bullets with score >= {threshold}"}

        self.vectors.collection.delete(ids=removed_ids)
        
        # Remove from in-memory structure
        ids_set = set(removed_ids)
        for cat in self.store.dynamicGuidelines["dynamicGuidelines_sections"].values():
            for sec, bullets in cat.items():
                cat[sec] = [b for b in bullets if b.get("bullet_id") not in ids_set]

        self.store.save()
        return {"removed_bullets": removed_info, "count": len(removed_info), "threshold": threshold}

    def generate_curator_report(self, output_path=None):
        if not self.processing_history:
            return {"message": "No processing history."}
            
        if not output_path:
            base_dir = Path(self.store.dynamicGuidelines_file_path).parent.parent.parent
            output_path = str(base_dir / "output" / "bullet_tracking" / "curator_report.json")

        total_rep = {
            "total_files_processed": len(self.processing_history),
            "n_operation": 0, "operation_types": Counter(), "total_success": 0, "total_rejected": 0,
            "total_modified": 0, "new_bullets_per_section": Counter(), "new_bullets_per_supercategory": Counter()
        }

        for stats in self.processing_history.values():
            total_rep["n_operation"] += stats["n_operation"]
            total_rep["operation_types"].update(stats["operation_types"])
            total_rep["total_success"] += stats["total_success"]
            total_rep["total_rejected"] += stats["total_rejected"]
            total_rep["total_modified"] += len(stats.get("modified_bullets", []))
            
            for added in stats.get("added_bullets", []):
                section = added["section"]
                supercat, _, _ = self.store.resolve_section(section)
                total_rep["new_bullets_per_section"][section] += 1
                if supercat:
                    total_rep["new_bullets_per_supercategory"][supercat] += 1

        report = {"files": self.processing_history, "total_report": total_rep}
        report["total_report"]["operation_types"] = dict(total_rep["operation_types"])
        report["total_report"]["new_bullets_per_section"] = dict(total_rep["new_bullets_per_section"])
        report["total_report"]["new_bullets_per_supercategory"] = dict(total_rep["new_bullets_per_supercategory"])

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Curator report saved to {output_path}")
        except Exception as e:
            print(f"Failed to save report: {e}")
        return report
