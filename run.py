#!/usr/bin/env python3
"""
Zeno Workshop — run.py
Usage:
  python run.py                        # prompts for project path
  python run.py ./test_project         # loads test_project directly
  python run.py /path/to/your/project  # any project folder
  python run.py --hud ./test_project   # desktop HUD prototype
"""
import sys

from zeno.hud import run_hud
from zeno.main import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hud":
        project_path = sys.argv[2] if len(sys.argv) > 2 else None
        run_hud(project_path)
    else:
        main()
