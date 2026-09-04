#!/usr/bin/env python3
"""
CLI utility to compute Inter-Annotator Agreement (IAA) on standoff annotations.
Bridges to src.common.iaa.
Usage:
    python scripts/compute_iaa.py directory_with_gold directory_with_test --exact
    python scripts/compute_iaa.py directory_with_gold directory_with_test --overlap
    python scripts/compute_iaa.py --help
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.iaa import main

if __name__ == "__main__":
    main(sys.argv[1:])
