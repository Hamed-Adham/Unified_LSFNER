#!/usr/bin/env python3
"""
CLI utility to inspect, export, backup, and summarize Dynamic Guidelines.
Usage:
    python scripts/manage_guidelines.py --stats
    python scripts/manage_guidelines.py --export-md output/bertized_ace/guidelines.md
    python scripts/manage_guidelines.py --export-csv output/bertized_ace/guidelines.csv
    python scripts/manage_guidelines.py --snapshot my_checkpoint
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

# Resolve workspace root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.agentic_guidelines.manager import DynamicGuidebookManager, DynamicGuidelinesManager

DEFAULT_GUIDELINES_PATH = (
    PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidebook.json"
    if (PROJECT_ROOT / "data" / "guidelines" / "dynamicGuidebook.json").exists()
    else PROJECT_ROOT / "output" / "bertized_ace" / "dynamicGuidelines.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Manage, export, and inspect Dynamic Guidebook records.")
    parser.add_argument(
        "--guidebook-path",
        "--guidelines-path",
        dest="guidelines_path",
        type=str,
        default=str(DEFAULT_GUIDELINES_PATH),
        help="Path to dynamicGuidebook.json file"
    )
    parser.add_argument("--stats", action="store_true", help="Print summary statistics of all active guidebook records")
    parser.add_argument("--export-md", type=str, help="Export guidebook to human-readable Markdown file")
    parser.add_argument("--export-csv", type=str, help="Export guidebook to tabular CSV file")
    parser.add_argument("--snapshot", type=str, help="Create a labeled backup snapshot in the backup/ directory")
    return parser.parse_args()


def main():
    args = parse_args()
    guidelines_file = Path(args.guidelines_path)

    if not guidelines_file.exists():
        print(f"⚠️ Warning: Guidebook file not found at: {guidelines_file}")
        print("Initializing new empty manager to inspect schema...")

    mgr = DynamicGuidebookManager(guidebook_file_path=str(guidelines_file))

    # 1. Print Stats
    if args.stats or (not args.export_md and not args.export_csv and not args.snapshot):
        total_records = 0
        category_counts = {}
        total_helpful = 0
        total_harmful = 0
        total_usage = 0

        for bullet, supercategory, section_name in mgr.store.iter_bullets():
            total_records += 1
            category_counts[section_name] = category_counts.get(section_name, 0) + 1
            metrics = bullet.get("usage_metrics", {})
            total_helpful += metrics.get("helpful", bullet.get("helpful", 0))
            total_harmful += metrics.get("harmful", bullet.get("harmful", 0))
            total_usage += metrics.get("usage_count", bullet.get("usage_count", 0))

        print("=" * 80)
        print("📚 DYNAMIC GUIDEBOOK REPOSITORY STATISTICS")
        print("=" * 80)
        print(f"  • Source File         : {guidelines_file}")
        print(f"  • Total Active Records: {total_records}")
        print(f"  • Total Usages        : {total_usage}")
        print(f"  • Helpful Markings    : {total_helpful} (Green)")
        print(f"  • Harmful Markings    : {total_harmful} (Red / Pruning Candidates)")
        print("-" * 80)
        print(f"{'Category / Label':<45} | {'Active Records':<15}")
        print("-" * 80)
        for sec, cnt in sorted(category_counts.items()):
            print(f"{sec:<45} | {cnt:<15}")
        print("=" * 80)

    # 2. Export Markdown
    if args.export_md:
        out_path = Path(args.export_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = mgr.get_dynamicGuidelines_for_generator_md()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"✅ Exported Markdown guidelines to: {out_path}")

    # 3. Export CSV
    if args.export_csv:
        out_path = Path(args.export_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "section", "supercategory", "guideline", "usage_count", "helpful", "harmful", "modifications"])
            for bullet, supercategory, section_name in mgr.store.iter_bullets():
                metrics = bullet.get("usage_metrics", {})
                writer.writerow([
                    bullet.get("id", bullet.get("bullet_id", "")),
                    section_name,
                    supercategory,
                    bullet.get("guideline", bullet.get("content", "")),
                    metrics.get("usage_count", bullet.get("usage_count", 0)),
                    metrics.get("helpful", bullet.get("helpful", 0)),
                    metrics.get("harmful", bullet.get("harmful", 0)),
                    metrics.get("modification_count", bullet.get("modification_count", 0))
                ])
        print(f"✅ Exported Tabular CSV guidelines to: {out_path}")

    # 4. Create Snapshot
    if args.snapshot:
        backup_dir = guidelines_file.parent / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_label = args.snapshot.strip().replace(" ", "_")
        snapshot_file = backup_dir / f"dynamicGuidelines_{clean_label}_{ts}.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(mgr.store.dynamicGuidelines, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved snapshot backup to: {snapshot_file}")


if __name__ == "__main__":
    main()
