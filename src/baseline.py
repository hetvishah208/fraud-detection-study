"""
Phase 2b - Logistic regression baseline.

The baseline exists so every later improvement is measured against something
real. Logistic regression is a deliberate choice: it's fast, calibrated-ish,
and if a linear model already separates the classes well, that tells you
something about the problem. We report PR-AUC as the headline.

Run:  py src/baseline.py
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

from config import load_data, OUTPUTS_DIR
from data_prep import make_splits
from metrics import evaluate, best_f1_threshold


def run_baseline(split, class_weight=None):
    clf = LogisticRegression(
        max_iter=2000, class_weight=class_weight
    )
    clf.fit(split.X_train, split.y_train)
    val_scores = clf.predict_proba(split.X_val)[:, 1]
    test_scores = clf.predict_proba(split.X_test)[:, 1]
    # choose threshold on validation, report on test (no test leakage)
    t = best_f1_threshold(split.y_val, val_scores)
    m = evaluate(split.y_test, test_scores, threshold=t)
    return clf, m


def main():
    df = load_data()
    split = make_splits(df)

    print("\n[baseline] plain logistic regression")
    _, m_plain = run_baseline(split, class_weight=None)
    print(f"  PR-AUC={m_plain.pr_auc:.4f}  ROC-AUC={m_plain.roc_auc:.4f}  "
          f"P={m_plain.precision:.3f} R={m_plain.recall:.3f} "
          f"F1={m_plain.f1:.3f} @t={m_plain.threshold:.2f}")

    print("[baseline] logistic regression with balanced class weights")
    _, m_bal = run_baseline(split, class_weight="balanced")
    print(f"  PR-AUC={m_bal.pr_auc:.4f}  ROC-AUC={m_bal.roc_auc:.4f}  "
          f"P={m_bal.precision:.3f} R={m_bal.recall:.3f} "
          f"F1={m_bal.f1:.3f} @t={m_bal.threshold:.2f}")

    out = {"plain": m_plain.as_row(), "balanced": m_bal.as_row()}
    with open(OUTPUTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[baseline] wrote {OUTPUTS_DIR/'baseline_metrics.json'}")
    print("[baseline] note: PR-AUC is the metric to watch, not ROC-AUC - on "
          "extreme imbalance ROC-AUC looks rosy even for weak models.")


if __name__ == "__main__":
    main()
