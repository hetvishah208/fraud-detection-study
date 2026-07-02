"""
Phase 1 - Exploratory data analysis + feature separability.

This is the "understand the data before modeling" phase. The centerpiece is a
per-feature separability ranking: for every feature, how cleanly does it pull
fraud apart from non-fraud? We use two complementary, defensible metrics:

  1. |Cohen's d|  - absolute standardized mean difference between the two
     classes. Scale-free, interpretable ("how many pooled std-devs apart are
     the class means").
  2. KS statistic - Kolmogorov-Smirnov distance between the two class
     distributions. Catches separation that isn't just a mean shift.

Output:
  - outputs/separability.csv         (the ranked table)
  - outputs/separability_top.png     (bar chart of the most separable features)
  - outputs/dist_<feat>.png          (overlaid fraud vs non-fraud distributions)
  - outputs/class_balance.png        (the imbalance, visualized)
  - outputs/amount_by_class.png      (Amount distribution by class, log scale)

Run:  py src/eda.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
from scipy import stats

from config import load_data, OUTPUTS_DIR, TARGET


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute standardized mean difference between groups a and b."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return abs(a.mean() - b.mean()) / pooled


def separability_table(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c != TARGET]
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]

    rows = []
    for f in feats:
        fa = fraud[f].to_numpy()
        la = legit[f].to_numpy()
        d = cohens_d(fa, la)
        ks = stats.ks_2samp(fa, la).statistic
        rows.append({"feature": f, "cohens_d": d, "ks_stat": ks})

    out = pd.DataFrame(rows)
    # rank by the average of the two min-max-normalized signals so neither
    # metric dominates purely because of its scale
    for col in ["cohens_d", "ks_stat"]:
        lo, hi = out[col].min(), out[col].max()
        out[col + "_norm"] = 0.0 if hi == lo else (out[col] - lo) / (hi - lo)
    out["separability_score"] = out[["cohens_d_norm", "ks_stat_norm"]].mean(axis=1)
    out = out.sort_values("separability_score", ascending=False).reset_index(drop=True)
    return out


def plot_class_balance(df: pd.DataFrame):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Legit (0)", "Fraud (1)"], counts.values,
                  color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Class imbalance: fraud = {df[TARGET].mean()*100:.3f}% of rows")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "class_balance.png", dpi=120)
    plt.close(fig)


def plot_amount_by_class(df: pd.DataFrame):
    if "Amount" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color, label in [(0, "#4C72B0", "Legit"), (1, "#C44E52", "Fraud")]:
        vals = df.loc[df[TARGET] == cls, "Amount"]
        vals = vals[vals > 0]
        ax.hist(np.log1p(vals), bins=50, alpha=0.55, color=color,
                label=label, density=True)
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("density")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "amount_by_class.png", dpi=120)
    plt.close(fig)


def plot_top_separability(table: pd.DataFrame, k: int = 12):
    top = table.head(k)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"][::-1], top["separability_score"][::-1],
            color="#55A868")
    ax.set_xlabel("separability score (0-1)")
    ax.set_title(f"Top {k} features separating fraud from non-fraud")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "separability_top.png", dpi=120)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, features, prefix="dist"):
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]
    for f in features:
        fig, ax = plt.subplots(figsize=(6, 4))
        lo = np.percentile(df[f], 1)
        hi = np.percentile(df[f], 99)
        bins = np.linspace(lo, hi, 60)
        ax.hist(legit[f], bins=bins, alpha=0.5, density=True,
                color="#4C72B0", label="Legit")
        ax.hist(fraud[f], bins=bins, alpha=0.5, density=True,
                color="#C44E52", label="Fraud")
        ax.set_title(f"{f}: fraud vs non-fraud")
        ax.set_xlabel(f); ax.set_ylabel("density"); ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / f"{prefix}_{f}.png", dpi=120)
        plt.close(fig)


def main():
    df = load_data()

    print("\n[eda] computing feature separability...")
    table = separability_table(df)
    table.to_csv(OUTPUTS_DIR / "separability.csv", index=False)

    print("[eda] top 10 most separable features:")
    print(table[["feature", "cohens_d", "ks_stat", "separability_score"]]
          .head(10).to_string(index=False))

    plot_class_balance(df)
    plot_amount_by_class(df)
    plot_top_separability(table)
    plot_feature_distributions(df, table["feature"].head(4).tolist())

    print(f"\n[eda] wrote separability.csv + plots to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
