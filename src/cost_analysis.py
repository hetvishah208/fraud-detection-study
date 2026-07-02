"""
Phase 5 - Cost analysis (the payoff).

This is the section that turns a model into a business decision. Accuracy and
even PR-AUC don't tell an operator where to set the decision threshold. Dollars
do.

Cost model per transaction:
  - True negative  (legit, passed):        $0
  - True positive  (fraud, caught):        $0   (we stopped it)
  - False negative (fraud, missed):        transaction Amount + fixed overhead
  - False positive (legit, flagged):       fixed false-alarm cost

We sweep the decision threshold across its full range, compute total expected
dollar cost on the test set at each, and the minimum of that curve is the
recommended operating point. Then we do sensitivity analysis: re-run the sweep
under different cost assumptions and show how the optimal threshold moves -
because the honest message of the project is "the right threshold depends on the
cost structure, and here's exactly how."

Uses the test scores saved by Phase 3 and the real Amount column from the data.
Run:  py src/cost_analysis.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    load_data, OUTPUTS_DIR, COST_FALSE_NEGATIVE_FIXED, COST_FALSE_POSITIVE,
    TARGET, SEED, TEST_SIZE,
)
from data_prep import make_splits


def total_cost(y_true, y_score, threshold, amounts,
               c_fn_fixed, c_fp):
    """Total dollar cost at a given threshold.

    FN cost = transaction amount lost + fixed overhead.
    FP cost = fixed false-alarm cost.
    """
    y_pred = (y_score >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_cost = (amounts[fn_mask].sum() + c_fn_fixed * fn_mask.sum())
    fp_cost = c_fp * fp_mask.sum()
    return float(fn_cost + fp_cost), int(fn_mask.sum()), int(fp_mask.sum())


def sweep(y_true, y_score, amounts, c_fn_fixed, c_fp, grid=None):
    if grid is None:
        grid = np.linspace(0.001, 0.999, 200)
    costs, fns, fps = [], [], []
    for t in grid:
        c, fn, fp = total_cost(y_true, y_score, t, amounts, c_fn_fixed, c_fp)
        costs.append(c); fns.append(fn); fps.append(fp)
    costs = np.array(costs)
    best_i = int(np.argmin(costs))
    return {
        "grid": grid, "costs": costs, "fns": np.array(fns), "fps": np.array(fps),
        "best_threshold": float(grid[best_i]),
        "best_cost": float(costs[best_i]),
        "best_fn": int(fns[best_i]), "best_fp": int(fps[best_i]),
    }


def get_test_amounts(df):
    """Recover the test-set Amount values aligned to the saved test scores.

    We re-run the identical split (same seed) so row order matches the
    y_test/scores saved in Phase 3.
    """
    # NOTE: Amount is scaled inside make_splits. We need the ORIGINAL amounts,
    # so we re-derive them from the raw df using the same split indices.
    split = make_splits(df, verbose=False)
    test_index = split.X_test.index
    raw_amounts = df.loc[test_index, "Amount"].to_numpy()
    return raw_amounts, split.y_test.to_numpy()


def main():
    df = load_data()
    data = np.load(OUTPUTS_DIR / "experiment_scores.npz")
    y_test = data["y_test"].astype(int)

    amounts, y_check = get_test_amounts(df)
    assert np.array_equal(y_check, y_test), \
        "test label order mismatch - re-run experiment.py and cost_analysis.py together"

    # pick the best strategy by PR-AUC from Phase 3 results
    results = pd.read_csv(OUTPUTS_DIR / "experiment_results.csv")
    best_strategy = results.sort_values("pr_auc", ascending=False).iloc[0]["strategy"]
    y_score = data[best_strategy]
    print(f"[cost] using best strategy by PR-AUC: {best_strategy}")

    # --- main sweep at the documented cost assumptions ---
    res = sweep(y_test, y_score, amounts,
                COST_FALSE_NEGATIVE_FIXED, COST_FALSE_POSITIVE)
    print(f"[cost] optimal threshold = {res['best_threshold']:.3f}")
    print(f"[cost] min total cost     = ${res['best_cost']:,.0f}  "
          f"(missed {res['best_fn']} fraud, {res['best_fp']} false alarms)")

    # baseline: cost if we naively used t=0.5
    c_half, fn_half, fp_half = total_cost(
        y_test, y_score, 0.5, amounts,
        COST_FALSE_NEGATIVE_FIXED, COST_FALSE_POSITIVE)
    saving = c_half - res["best_cost"]
    print(f"[cost] cost at naive t=0.50 = ${c_half:,.0f}  -> "
          f"optimizing saves ${saving:,.0f}")

    # --- cost curve plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["grid"], res["costs"], color="#C44E52", lw=2)
    ax.axvline(res["best_threshold"], ls="--", color="#333",
               label=f"optimal t={res['best_threshold']:.3f}")
    ax.axvline(0.5, ls=":", color="#888", label="naive t=0.50")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("total cost ($)")
    ax.set_title(f"Total cost vs threshold ({best_strategy})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cost_curve.png", dpi=120)
    plt.close(fig)

    # --- sensitivity analysis: vary FN fixed cost, watch optimal t move ---
    multipliers = [0.5, 1.0, 2.0, 4.0, 8.0]
    sens = []
    for mult in multipliers:
        r = sweep(y_test, y_score, amounts,
                  COST_FALSE_NEGATIVE_FIXED * mult, COST_FALSE_POSITIVE)
        sens.append({"fn_cost_mult": mult,
                     "optimal_threshold": r["best_threshold"],
                     "min_cost": r["best_cost"],
                     "missed_fraud": r["best_fn"],
                     "false_alarms": r["best_fp"]})
    sens_df = pd.DataFrame(sens)
    print("\n[cost] sensitivity - as the cost of a missed fraud rises, the "
          "optimal threshold should fall (catch more, tolerate more alarms):")
    print(sens_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sens_df["fn_cost_mult"], sens_df["optimal_threshold"],
            marker="o", color="#4C72B0")
    ax.set_xlabel("false-negative cost multiplier")
    ax.set_ylabel("optimal decision threshold")
    ax.set_title("Sensitivity: optimal threshold vs cost of a missed fraud")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cost_sensitivity.png", dpi=120)
    plt.close(fig)

    out = {
        "best_strategy": str(best_strategy),
        "optimal_threshold": res["best_threshold"],
        "min_cost": res["best_cost"],
        "naive_cost": c_half,
        "saving_vs_naive": saving,
        "missed_fraud_at_optimal": res["best_fn"],
        "false_alarms_at_optimal": res["best_fp"],
        "sensitivity": sens,
        "cost_assumptions": {
            "false_negative_fixed": COST_FALSE_NEGATIVE_FIXED,
            "false_positive": COST_FALSE_POSITIVE,
            "false_negative_also_includes": "transaction Amount",
        },
    }
    with open(OUTPUTS_DIR / "cost_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[cost] wrote cost_analysis.json + cost_curve.png + cost_sensitivity.png")


if __name__ == "__main__":
    main()
