"""
Phase 3 - The imbalance experiment (the spine of the project).

We hold the model family FIXED (gradient-boosted trees) and vary ONLY the
strategy for handling the 0.17% class imbalance. Everything else - features,
splits, seed, evaluation - is identical across strategies. That's what makes it
a controlled experiment rather than a bag of tricks.

Strategies compared:
  1. none            - train on raw imbalance, threshold 0.5
  2. class_weight    - scale_pos_weight = (#neg / #pos), threshold 0.5
  3. smote           - synthetic minority oversampling on TRAIN ONLY
  4. undersample     - random undersampling of the majority on TRAIN ONLY
  5. threshold_tuned - raw model, but threshold chosen on validation by F1

The expected (and to-be-confirmed) punchline: (5) threshold tuning on a
well-behaved model competes with or beats the resampling tricks (3, 4), which
is the counterintuitive, memorable result.

IMPORTANT on leakage: SMOTE and undersampling are applied ONLY to the training
fold, never to validation or test. Resampling the test set would fabricate a
fraud rate that doesn't exist in production and invalidate every metric.

Model backend:
  - Primary: XGBoost (what you run locally; install via requirements.txt).
  - Fallback: sklearn HistGradientBoostingClassifier, used automatically if
    xgboost isn't importable, so the experiment logic is always runnable.

Run:  py src/experiment.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from config import load_data, OUTPUTS_DIR, SEED
from data_prep import make_splits
from metrics import evaluate, best_f1_threshold

# ---- model backend selection ------------------------------------------------
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# ---- resampling backend -----------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    _HAVE_IMBLEARN = True
except Exception:
    _HAVE_IMBLEARN = False


def make_model(scale_pos_weight: float = 1.0):
    """Return a fresh gradient-boosted-tree model with a fixed config."""
    if _HAVE_XGB:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )
    # sklearn fallback - no scale_pos_weight, so class_weight strategy is
    # approximated via sample_weight at fit time (handled by the caller).
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.1, random_state=SEED
    )


def _fit(model, X, y, sample_weight=None):
    if sample_weight is not None and not _HAVE_XGB:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def _scores(model, X):
    return model.predict_proba(X)[:, 1]


def strategy_none(split):
    model = _fit(make_model(), split.X_train, split.y_train)
    s_test = _scores(model, split.X_test)
    return evaluate(split.y_test, s_test, threshold=0.5), model, s_test


def strategy_class_weight(split):
    n_pos = int(split.y_train.sum())
    n_neg = int(len(split.y_train) - n_pos)
    spw = (n_neg / max(n_pos, 1))
    if _HAVE_XGB:
        model = _fit(make_model(scale_pos_weight=spw), split.X_train, split.y_train)
    else:
        # approximate with per-sample weights
        w = np.where(split.y_train == 1, spw, 1.0)
        model = _fit(make_model(), split.X_train, split.y_train, sample_weight=w)
    s_test = _scores(model, split.X_test)
    return evaluate(split.y_test, s_test, threshold=0.5), model, s_test


def strategy_smote(split):
    if not _HAVE_IMBLEARN:
        return None, None, None
    sm = SMOTE(random_state=SEED)
    Xr, yr = sm.fit_resample(split.X_train, split.y_train)
    model = _fit(make_model(), Xr, yr)
    s_test = _scores(model, split.X_test)
    return evaluate(split.y_test, s_test, threshold=0.5), model, s_test


def strategy_undersample(split):
    if not _HAVE_IMBLEARN:
        return None, None, None
    us = RandomUnderSampler(random_state=SEED)
    Xr, yr = us.fit_resample(split.X_train, split.y_train)
    model = _fit(make_model(), Xr, yr)
    s_test = _scores(model, split.X_test)
    return evaluate(split.y_test, s_test, threshold=0.5), model, s_test


def strategy_threshold_tuned(split):
    # same raw model as 'none', but pick the threshold on validation
    model = _fit(make_model(), split.X_train, split.y_train)
    s_val = _scores(model, split.X_val)
    t = best_f1_threshold(split.y_val, s_val)
    s_test = _scores(model, split.X_test)
    return evaluate(split.y_test, s_test, threshold=t), model, s_test


STRATEGIES = {
    "none": strategy_none,
    "class_weight": strategy_class_weight,
    "smote": strategy_smote,
    "undersample": strategy_undersample,
    "threshold_tuned": strategy_threshold_tuned,
}


def run_experiment(split):
    rows = []
    scores = {}   # keep test scores per strategy for later phases
    models = {}   # keep fitted models for SHAP / the app
    for name, fn in STRATEGIES.items():
        m, model, s_test = fn(split)
        if m is None:
            print(f"[experiment] SKIP {name} (imbalanced-learn not installed)")
            continue
        row = {"strategy": name, **m.as_row()}
        rows.append(row)
        scores[name] = s_test
        models[name] = model
        print(f"[experiment] {name:<16} PR-AUC={m.pr_auc:.4f} "
              f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
              f"@t={m.threshold:.2f}")
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    return table, scores, models


def main():
    backend = "XGBoost" if _HAVE_XGB else "sklearn HistGBT (fallback)"
    imb = "imbalanced-learn" if _HAVE_IMBLEARN else "MISSING (smote/undersample skipped)"
    print(f"[experiment] model backend: {backend}")
    print(f"[experiment] resampling backend: {imb}\n")

    df = load_data()
    split = make_splits(df)
    table, scores, models = run_experiment(split)

    table.to_csv(OUTPUTS_DIR / "experiment_results.csv", index=False)
    np.savez(OUTPUTS_DIR / "experiment_scores.npz",
             y_test=split.y_test.to_numpy(),
             **{k: v for k, v in scores.items()})

    # PR curves for all strategies (used by the app)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    y_test = split.y_test.to_numpy()
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y_test, s)
        ax.plot(rec, prec, lw=1.8, label=name)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall curves by imbalance strategy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "pr_curves.png", dpi=120)
    plt.close(fig)

    # persist the best model (by PR-AUC) for SHAP + the app
    best_name = table.iloc[0]["strategy"]
    try:
        import joblib
        joblib.dump(
            {"model": models[best_name],
             "strategy": best_name,
             "feature_names": split.feature_names,
             "scaler": split.scaler},
            OUTPUTS_DIR / "best_model.joblib",
        )
        print(f"[experiment] saved best model ({best_name}) to best_model.joblib")
    except Exception as e:
        print(f"[experiment] could not save model: {e}")

    print(f"\n[experiment] ranked by PR-AUC:")
    print(table[["strategy", "pr_auc", "precision", "recall", "f1"]]
          .to_string(index=False))
    print(f"\n[experiment] wrote experiment_results.csv + experiment_scores.npz")


if __name__ == "__main__":
    main()
