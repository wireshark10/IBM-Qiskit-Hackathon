"""Assemble track3_dynamics_under_noise.ipynb from the section modules.

    python scripts/build_notebook.py [sections...]      e.g.  beginner intermediate advanced
"""
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SECTIONS = sys.argv[1:] or ["beginner", "intermediate", "advanced", "closing"]
cells = []
for s in SECTIONS:
    mod = __import__(f"nb_{s}")
    cells += mod.CELLS
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = HERE.parent / "track3_dynamics_under_noise.ipynb"
nbformat.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
