"""
Run the full study end-to-end, in order.

    py src/run_all.py

Each phase writes its artifacts to outputs/. Safe to re-run. The Streamlit app
reads those artifacts, so once this finishes you can launch the app.
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

PHASES = [
    ("Phase 1  EDA + separability", "eda.py"),
    ("Phase 2  Logistic baseline", "baseline.py"),
    ("Phase 3  Imbalance experiment", "experiment.py"),
    ("Phase 4  Bootstrap + significance", "stats_tests.py"),
    ("Phase 5  Cost analysis", "cost_analysis.py"),
    ("Phase 6  SHAP explainability", "explain.py"),
]


def main():
    for label, script in PHASES:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        runpy.run_path(str(SRC / script), run_name="__main__")
    print("\n" + "=" * 70)
    print("Done. Launch the app with:  streamlit run app/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
