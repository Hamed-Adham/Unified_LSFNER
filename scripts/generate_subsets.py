#!/usr/bin/env python3
"""
CLI utility to generate balanced annotation subsets with density stratification.
Bridges to src.common.unified_subset_generator.
Usage:
    python scripts/generate_subsets.py --source data/new/400Abstracts/train --sizes 40 20 10 --output data/processed/subsets
    python scripts/generate_subsets.py --help
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.unified_subset_generator import orchestrate_unified

def parse_args():
    parser = argparse.ArgumentParser(description="Generate balanced entity subsets from BRAT annotations.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/new/400Abstracts/train",
        help="Source directory containing .txt and .ann files"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[40, 20, 10],
        help="List of subset sizes to generate (e.g. --sizes 40 20 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/subsets",
        help="Base output directory where subsets will be written"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    source = os.path.join(PROJECT_ROOT, args.source) if not os.path.isabs(args.source) else args.source
    output = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    
    if not os.path.exists(source):
        print(f"❌ Error: Source directory does not exist: {source}")
        sys.exit(1)
        
    print(f"📦 Generating subsets from {source} into {output}...")
    print(f"   Subset sizes: {args.sizes}")
    orchestrate_unified(source, args.sizes, output)
    print("✅ All subsets generated successfully.")

if __name__ == "__main__":
    main()
