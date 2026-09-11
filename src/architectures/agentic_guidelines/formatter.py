import json
from typing import Dict, List, Any, Optional

class Formatter:
    def __init__(self, vectors=None):
        self.vectors = vectors

    @staticmethod
    def to_dinasor_view(g: Dict[str, Any]) -> Dict[str, Any]:
        """
        Projects a dynamic guideline into the Dinasor view.
        Hides ground truth, LLM error, triad, and internal metrics for blind arbitration.
        """
        view = {
            "id": g.get("id", g.get("bullet_id", "")),
            "span": g.get("span", ""),
            "span_sentence": g.get("span_sentence", ""),
            "bert_label": g.get("bert_label", ""),
            "challenge_types": g.get("challenge_types", []),
            "guideline": g.get("guideline", g.get("content", ""))
        }
        if "bert_info" in g:
            view["bert_info"] = g["bert_info"]
        return view

    @staticmethod
    def to_generator_view(g: Dict[str, Any]) -> Dict[str, Any]:
        """
        Projects a dynamic guideline into the Generator ACE view.
        Includes triad, bert_info, and similar_spans for deep guideline calibration.
        """
        view = {
            "id": g.get("id", g.get("bullet_id", "")),
            "span": g.get("span", ""),
            "span_sentence": g.get("span_sentence", ""),
            "bert_label": g.get("bert_label", ""),
            "ground_truth_label": g.get("ground_truth_label", ""),
            "llm_label": g.get("llm_label", ""),
            "triad": g.get("triad", {}),
            "guideline": g.get("guideline", g.get("content", ""))
        }
        if "bert_info" in g:
            view["bert_info"] = g["bert_info"]
        if "similar_spans" in g:
            view["similar_spans"] = g["similar_spans"]
        return view

    @staticmethod
    def to_curator_view(g: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full record view for Curator reflection & maintenance.
        """
        return g

    def export(self, store, format: str = "raw", query_text: Optional[str] = None, n_similar: int = 3) -> Any:
        """
        Universal export dispatcher supporting both modern JSON views and legacy markdown formats.
        """
        if format == "dinasor":
            guidelines = self._get_selected_guidelines(store, query_text=query_text, n_results=n_similar)
            return [self.to_dinasor_view(g) for g in guidelines]
            
        elif format == "dinasor_json":
            views = self.export(store, format="dinasor", query_text=query_text, n_similar=n_similar)
            return json.dumps(views, indent=2, ensure_ascii=False)

        elif format == "generator":
            guidelines = self._get_selected_guidelines(store, query_text=query_text, n_results=n_similar)
            return [self.to_generator_view(g) for g in guidelines]

        elif format == "generator_json":
            views = self.export(store, format="generator", query_text=query_text, n_similar=n_similar)
            return json.dumps(views, indent=2, ensure_ascii=False)

        elif format in ("raw", "guidebook"):
            return store.guidebook

        elif format == "guidebook_json":
            return json.dumps(store.guidebook, indent=2, ensure_ascii=False)

        elif format == "md_generator":
            return self._to_md_generator(store)

        elif format == "md_curator":
            return self._to_md_curator(store, query_text=query_text, n_similar=n_similar)

        elif format == "flat":
            return self._to_flat(store)

        else:
            raise ValueError(f"Unknown format: {format}")

    def _get_selected_guidelines(self, store, query_text: Optional[str] = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieves either top-k similar guidelines via vector search or all guidelines."""
        all_guidelines = list(store.iter_guidelines())
        if not all_guidelines:
            return []

        if query_text and self.vectors and self.vectors.collection and self.vectors.collection.count() > 0:
            try:
                similar_entries = self.vectors.find_similar(query_text, n_results=min(n_results, len(all_guidelines)))
                matched_ids = {item["id"] for item in similar_entries}
                selected = [g for g in all_guidelines if g.get("id") in matched_ids]
                if selected:
                    return selected
            except Exception as e:
                print(f"Warning: vector similarity retrieval error: {e}")

        return all_guidelines[:n_results] if query_text else all_guidelines

    def _to_md_generator(self, store) -> str:
        """Legacy markdown projection for generator."""
        lines = []
        for g in store.iter_guidelines():
            gid = g.get("id", g.get("bullet_id", "DG"))
            rule = g.get("guideline", g.get("content", ""))
            span = g.get("span", "")
            cat = g.get("ground_truth_label") or g.get("bert_label") or "General"
            
            header = f"- **{gid}** [{cat}]"
            if span:
                header += f" (Target: '{span}')"
            lines.append(f"{header}: {rule}")
            
        return "\n".join(lines) if lines else "No dynamicGuidelines available"

    def _to_md_curator(self, store, query_text: Optional[str] = None, n_similar: int = 3) -> str:
        """Legacy markdown projection for curator."""
        lines = []
        for g in self._get_selected_guidelines(store, query_text=query_text, n_results=n_similar):
            gid = g.get("id", g.get("bullet_id", "DG"))
            rule = g.get("guideline", g.get("content", ""))
            metrics = g.get("usage_metrics", {})
            helpful = metrics.get("helpful", g.get("helpful", 0))
            harmful = metrics.get("harmful", g.get("harmful", 0))
            usage = metrics.get("usage_count", g.get("usage_count", 0))
            
            lines.append(f"### Rule {gid}")
            if g.get("span"):
                lines.append(f"- **Trigger Span**: {g.get('span')}")
            if g.get("ground_truth_label"):
                lines.append(f"- **Ground Truth**: {g.get('ground_truth_label')} | **BERT**: {g.get('bert_label')}")
            lines.append(f"- **Guideline**: {rule}")
            lines.append(f"- **Metrics**: usage={usage}, helpful={helpful}, harmful={harmful}")
            
            if "triad" in g and g["triad"]:
                lines.append(f"- **Triad**: {g['triad'].get('type', '')} - {g['triad'].get('description', '')}")
            if "similar_spans" in g and g["similar_spans"]:
                spans_str = ", ".join(s.get("span", "") for s in g["similar_spans"])
                lines.append(f"- **Similar Spans**: {spans_str}")
                
        return "\n\n".join(lines) if lines else "No dynamicGuidelines available"

    def _to_flat(self, store) -> Dict[str, List[str]]:
        """Legacy flat dictionary grouped by category."""
        flat = {}
        for g in store.iter_guidelines():
            cat = g.get("ground_truth_label") or g.get("bert_label") or "General"
            gid = g.get("id", g.get("bullet_id", "DG"))
            rule = g.get("guideline", g.get("content", ""))
            flat.setdefault(cat, []).append(f"{gid}: {rule}")
        return flat
