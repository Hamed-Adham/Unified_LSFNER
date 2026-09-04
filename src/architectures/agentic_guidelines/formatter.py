class Formatter:
    def __init__(self, vectors):
        self.vectors = vectors

    def export(self, store, format="raw", query_text=None, n_similar=3):
        if format == "raw":
            return self._to_raw(store)
        elif format == "flat":
            return self._to_flat(store)
        elif format == "md_generator":
            return self._to_md_generator(store)
        elif format == "md_curator":
            return self._to_md_curator(store, query_text, n_similar)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_raw(self, store):
        return store.dynamicGuidelines

    def _to_flat(self, store):
        flat = {section_name: [] for data in store.DYNAMICGUIDELINES_CONFIG.values() for section_name in data["sections"]}
        for bullet, _, section_name in store.iter_bullets():
            bullet_id = bullet.get("bullet_id", "")
            content = bullet.get("content", "")
            if bullet_id:
                flat[section_name].append(f"{bullet_id}: {content}")
            else:
                flat[section_name].append(content)
        return flat

    def _to_md_generator(self, store):
        md_lines = []
        for supercategory, sections in store.dynamicGuidelines.get("dynamicGuidelines_sections", {}).items():
            md_lines.append(f"# {supercategory}")
            for section_name, bullets in sections.items():
                if bullets:
                    md_lines.append(f"## {section_name}")
                    for b in bullets:
                        bullet_id = b.get("bullet_id", "")
                        content = b.get("content", "")
                        if bullet_id:
                            md_lines.append(f"- **{bullet_id}**: {content}")
                        else:
                            md_lines.append(f"- {content}")
        return "\n".join(md_lines)

    def _to_md_curator(self, store, query_text=None, n_similar=3):
        similar_ids = set()
        count = self.vectors.collection.count()
        if query_text and count > 0:
            try:
                similar_bullets = self.vectors.find_similar(query_text, n_results=min(n_similar, count))
                similar_ids = {b["bullet_id"] for b in similar_bullets}
            except Exception as e:
                print(f"Error querying similar bullets for curator optimization: {e}")

        md_lines = []
        for supercategory, sections in store.dynamicGuidelines.get("dynamicGuidelines_sections", {}).items():
            md_lines.append(f"# {supercategory}")
            for section_name, bullets in sections.items():
                if bullets:
                    md_lines.append(f"## {section_name}")
                    for b in bullets:
                        bullet_id = b.get("bullet_id")
                        md_lines.append(f"### Rule {bullet_id}")
                        md_lines.append(f"- **Content**: {b.get('content', '')}")
                        
                        # Basic Metrics
                        metrics = f"usage={b.get('usage_count', 0)}, helpful={b.get('helpful', 0)}, harmful={b.get('harmful', 0)}"
                        md_lines.append(f"- **Metrics**: {metrics}")
                        
                        # Include detailed fields ONLY if the rule is similar or query_text is None
                        if query_text is None or bullet_id in similar_ids:
                            # History
                            history = b.get("history", [])
                            if not history and b.get("content_history"):
                                history = [{
                                    "previous_content": h,
                                    "modified_in_file": "unknown",
                                    "reason_for_modification": "Legacy update"
                                } for h in b.get("content_history", [])]
                                
                            if history:
                                md_lines.append("- **History**:")
                                for h in history:
                                    prev = h.get("previous_content", "")
                                    reason = h.get("reason_for_modification", "Modified by Curator")
                                    mod_file = h.get("modified_in_file", "unknown")
                                    md_lines.append(f"  - Modified in {mod_file}: \"{prev}\" -> Reason: {reason}")
                            
                            # Diagnostic context
                            dc = b.get("diagnostic_context", {})
                            if dc:
                                md_lines.append("- **Diagnostic Context**:")
                                root_cause = dc.get("root_cause_of_creation", "")
                                if root_cause:
                                    md_lines.append(f"  - **Root Cause**: {root_cause}")
                                errs = dc.get("error_examples", [])
                                if errs:
                                    md_lines.append("  - **Error Examples**:")
                                    for err in errs:
                                        etype = err.get("error_type")
                                        token = err.get("erroneous_token")
                                        fn = err.get("file_name", "unknown")
                                        sent = err.get("sentence", "")
                                        md_lines.append(f"    - [{etype}] Token: \"{token}\" in file {fn} | Sentence: \"{sent}\"")
        return "\n".join(md_lines)
