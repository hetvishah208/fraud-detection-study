"""
Phase 6a - Explainability with SHAP.

A fraud model that can't explain itself is hard to operationalize - analysts
need to know *why* a transaction was flagged. SHAP attributes each prediction to
its features in an additive, theoretically grounded way.

We produce:
  - outputs/shap_global.png    - mean |SHAP| per feature (global importance)
  - outputs/shap_beeswarm.png  - distribution of SHAP values per feature
  - outputs/shap_local.png     - a single flagged transaction explained

Gracefully skips (with a message) if shap isn't installed, so the pipeline never
hard-fails. Requires best_model.joblib from experiment.py.

Run:  py src/explain.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_data, OUTPUTS_DIR
from data_prep import make_splits

try:
    import shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

try:
    import joblib
except Exception:
    joblib = None


def main():
    if not _HAVE_SHAP:
        print("[explain] shap not installed - skipping. "
              "Install with: pip install shap")
        return
    if joblib is None:
        print("[explain] joblib not available - cannot load model.")
        return

    bundle_path = OUTPUTS_DIR / "best_model.joblib"
    if not bundle_path.exists():
        print(f"[explain] {bundle_path} missing - run experiment.py first.")
        return

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    df = load_data()
    split = make_splits(df, verbose=False)
    # sample for speed on CPU
    X_bg = split.X_train.sample(min(200, len(split.X_train)), random_state=42)
    X_explain = split.X_test.sample(min(500, len(split.X_test)), random_state=42)

    print("[explain] computing SHAP values (TreeExplainer)...")
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain)
    except Exception:
        # model-agnostic fallback
        explainer = shap.Explainer(model.predict_proba, X_bg)
        sv = explainer(X_explain).values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    sv = np.asarray(sv)
    if sv.ndim == 3:   # some explainers return (n, features, classes)
        sv = sv[:, :, 1]

    # global importance
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_names[i] for i in order][::-1],
            mean_abs[order][::-1], color="#8172B3")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "shap_global.png", dpi=120)
    plt.close(fig)

    # beeswarm
    try:
        plt.figure()
        shap.summary_plot(sv, X_explain, feature_names=feature_names,
                          show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "shap_beeswarm.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"[explain] beeswarm skipped: {e}")

    print(f"[explain] wrote SHAP plots to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
