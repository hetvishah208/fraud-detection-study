"""
Phase 4 - Statistical honesty: bootstrap CIs + significance test.

A point estimate like "PR-AUC = 0.82" hides whether that number is solid or
luck. Two things fix that:

  1. Bootstrap confidence intervals. We resample the test set with replacement
     many times, recompute PR-AUC each time, and report the 2.5th-97.5th
     percentile band. Now every headline number is an interval, e.g.
     "PR-AUC 0.82, 95% CI [0.79, 0.85]".

  2. Paired significance test between the two best strategies. Using the SAME
     bootstrap resamples for both models, we look at the distribution of the
     PR-AUC *difference*. If that difference's 95% CI excludes 0, the winner is
     significantly better; if it straddles 0, the gap is noise and we say so.
     Pairing (same resample indices for both models) controls for test-set
     variance and is the statistically correct way to compare them.

Reads experiment_scores.npz produced by Phase 3.
Run:  py src/stats_tests.py
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score

from config import OUTPUTS_DIR

N_BOOT = 2000
SEED = 42


def _load_scores():
    data = np.load(OUTPUTS_DIR / "experiment_scores.npz")
    y = data["y_test"].astype(int)
    scores = {k: data[k] for k in data.files if k != "y_test"}
    return y, scores


def bootstrap_pr_auc(y, score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    point = average_precision_score(y, score)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)        # resample with replacement
        # guard against a resample with only one class
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            vals[b] = np.nan
            continue
        vals[b] = average_precision_score(y[idx], score[idx])
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def paired_diff_test(y, score_a, score_b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the PR-AUC difference (A - B) using shared resamples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            diffs[b] = np.nan
            continue
        a = average_precision_score(y[idx], score_a[idx])
        bb = average_precision_score(y[idx], score_b[idx])
        diffs[b] = a - bb
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: proportion of resamples on the wrong side of 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
    }


def main():
    y, scores = _load_scores()

    print("[stats] bootstrap PR-AUC 95% CIs:")
    cis = {}
    for name, s in scores.items():
        ci = bootstrap_pr_auc(y, s)
        cis[name] = ci
        print(f"  {name:<16} {ci['point']:.4f}  "
              f"95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")

    # compare the top 2 strategies by point PR-AUC
    ranked = sorted(cis.items(), key=lambda kv: kv[1]["point"], reverse=True)
    out = {"cis": cis}
    if len(ranked) >= 2:
        a_name, b_name = ranked[0][0], ranked[1][0]
        test = paired_diff_test(y, scores[a_name], scores[b_name])
        out["comparison"] = {"a": a_name, "b": b_name, **test}
        verdict = ("SIGNIFICANT" if test["significant"]
                   else "NOT significant (gap is within noise)")
        print(f"\n[stats] {a_name} vs {b_name}:")
        print(f"  mean PR-AUC diff={test['mean_diff']:+.4f}  "
              f"95% CI [{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]  "
              f"p={test['p_value']:.3f}  -> {verdict}")

    with open(OUTPUTS_DIR / "stats_tests.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[stats] wrote {OUTPUTS_DIR/'stats_tests.json'}")


if __name__ == "__main__":
    main()
