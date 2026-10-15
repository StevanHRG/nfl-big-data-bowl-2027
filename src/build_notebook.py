"""Monta o notebook público do Kaggle a partir dos módulos em src/ (uma única fonte de verdade)."""
from pathlib import Path

import nbformat as nbf

SRC = Path(__file__).parent
OUT = SRC.parent / "notebook" / "bend-you-can-measure.ipynb"
OUT.parent.mkdir(exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def module(name):
    return code(f"%%writefile {name}\n" + (SRC / name).read_text(encoding="utf8"))


cells = [
    md("# Bend You Can Measure — reproducible code\n\n"
       "Companion notebook for the Big Data Bowl 2027 writeup *Bend You Can Measure: A Combine Turn-Force Score "
       "That Predicts NFL Pass-Rush Pressure*.\n\n"
       "Every number and figure in the writeup is produced here. The notebook writes small modules to disk and runs them "
       "in order:\n\n"
       "1. `data.py` / `features.py` — loading and the per-frame centripetal-acceleration (turn force) feature\n"
       "2. `validate_dl.py` — robustness checks and in-game turn force on every DL pass rush\n"
       "3. `final_metric.py` — the Combine Bend Score (CBS) and effect sizes in pressures per 100 rushes\n"
       "4. `nested_test.py` — nested, leave-one-draft-class-out validation with permutations (DL and OL)\n"
       "5. `composite_test.py` — includes the WR \"brake-and-burst\" idea that did **not** survive\n"
       "6. `figures.py` — the six figures"),
    code("import importlib.util, subprocess, sys\n"
         "for pkg in ['polars', 'statsmodels', 'scikit-learn']:\n"
         "    mod = 'sklearn' if pkg == 'scikit-learn' else pkg\n"
         "    if importlib.util.find_spec(mod) is None:\n"
         "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)"),
    md("## Modules"),
    *[module(n) for n in ["data.py", "features.py", "dl_viability.py", "validate_dl.py", "final_metric.py",
                          "nested_test.py", "composite_test.py", "sensitivity.py", "figures.py"]],
    md("## 1. Robustness and in-game turn force"),
    code("%run validate_dl.py"),
    md("## 2. Combine Bend Score and effect sizes"),
    code("%run final_metric.py"),
    md("## 2b. Sensitivity to arbitrary choices"),
    code("%run sensitivity.py"),
    md("## 3. Nested validation (takes ~15 min: 1,000 permutations)"),
    code("%run nested_test.py"),
    md("## 4. Ideas that failed (reported for transparency)"),
    code("%run dl_viability.py\n%run composite_test.py"),
    md("## 5. Figures"),
    code("%run figures.py\nfrom IPython.display import Image, display\nimport glob\n"
         "for f in sorted(glob.glob('figures/*.png')):\n    display(Image(f))"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
