#!/usr/bin/env python3
"""
CLI entry point to run Hybrid LinkNER evaluation & benchmarking.
Bridges directly to src.architectures.hybrid_linkner.run_evaluation.
Usage:
    python scripts/run_evaluation.py --dataset subset_1 --threshold 0.35
    python scripts/run_evaluation.py --help
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.hybrid_linkner.run_evaluation import main

if __name__ == "__main__":
    main()
