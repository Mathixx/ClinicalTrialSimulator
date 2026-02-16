#!/usr/bin/env python3
"""
Launch the full stack from repo root without installing the package.
Usage: python scripts/launch.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "servers"))

from clinical_trial_simulator.launch import main

if __name__ == "__main__":
    main()
