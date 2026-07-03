"""
Fraud detection cost-explorer - Streamlit app.

The signature interaction: two sliders for the dollar cost of a missed fraud and
the dollar cost of a false alarm. As you drag them, the optimal decision
threshold, the confusion matrix, and the total cost update live. The whole point
of the project - "the right threshold depends on the cost structure" - becomes
something you feel, not just read.

Tabs:
  1. Cost Explorer   - the live interactive centerpiece
  2. The Experiment  - imbalance-strategy comparison table + PR curves
  3. Data & Signal   - separability story and class imbalance
  4. How it works    - plain-language method writeup

Runs against outputs/ artifacts produced by the src/ pipeline. If artifacts are
missing it tells you exactly which script to run.

Local run:  streamlit run app/streamlit_app.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="Fraud Cost Explorer", layout="wide",
                   initial_sidebar_state="collapsed")

# ---- minimal, disciplined styling; one accent color -------------------------
ACCENT = "#C44E52"
st.markdown(f"""
<style>
  .stApp {{ }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
  .small {{ color:#6b7280; font-size:0.9rem; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_scores():
    f = OUT / "experiment_scores.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {"y_test": d["y_test"].astype(int),
            "scores": {k: d[k] for k in d.files if k != "y_test"}}


@st.cache_data
def load_amounts():
    """Original (unscaled) test amounts aligned to y_test, via the same split."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from config import load_data
    from data_prep import make_splits
    df = load_data(verbose=False)
    split = make_splits(df, verbose=False)
    return df.loc[split.X_test.index, "Amount"].to_numpy(), split.y_test.to_numpy()


@st.cache_data
def load_json(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data
def load_csv(name):
    f = OUT / name
    return pd.read_csv(f) if f.exists() else None


def confusion_at(y, score, t):
    pred = (score >= t).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def total_cost(y, score, t, amounts, c_fn_fixed, c_fp):
    pred = (score >= t).astype(int)
    fn_mask = (y == 1) & (pred == 0)
    fp_mask = (y == 0) & (pred == 1)
    return float(amounts[fn_mask].sum() + c_fn_fixed * fn_mask.sum()
                 + c_fp * fp_mask.sum())


def optimal_threshold(y, score, amounts, c_fn_fixed, c_fp, grid=None):
    if grid is None:
        grid = np.linspace(0.001, 0.999, 200)
    costs = [total_cost(y, score, t, amounts, c_fn_fixed, c_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i]), grid, np.array(costs)


# ---- header -----------------------------------------------------------------
st.title("Fraud detection: a cost-optimization study")
st.markdown(
    "<span class='small'>Most fraud models stop at accuracy. The real question "
    "is where to set the decision threshold once you admit that catching fraud "
    "and annoying real customers both cost money. Drag the costs below and watch "
    "the optimal threshold move.</span>", unsafe_allow_html=True)

data = load_scores()
if data is None:
    st.error("No experiment artifacts found. Run the pipeline first:\n\n"
             "`py src/experiment.py`  then refresh.")
    st.stop()

results = load_csv("experiment_results.csv")
best_strategy = (results.sort_values('pr_auc', ascending=False).iloc[0]['strategy']
                 if results is not None else list(data['scores'])[0])

try:
    amounts, y_check = load_amounts()
    y = data["y_test"]
    if not np.array_equal(y_check, y):
        amounts = None
except Exception:
    amounts = None

tab1, tab2, tab3, tab4 = st.tabs(
    ["Cost Explorer", "The Experiment", "Data & Signal", "How it works"])

# =============================== TAB 1 =======================================
with tab1:
    y = data["y_test"]
    score = data["scores"][best_strategy]

    if amounts is None:
        st.warning("Using a flat per-fraud cost (real transaction amounts "
                   "unavailable in this environment).")
        amt = np.full_like(y, 0.0, dtype=float)
    else:
        amt = amounts

    c1, c2 = st.columns(2)
    with c1:
        c_fn = st.slider("Cost of a MISSED fraud ($, fixed overhead)",
                         0, 1000, 100, step=10, key="fn_cost",
                         help="On top of the transaction amount you lose.")
    with c2:
        c_fp = st.slider("Cost of a FALSE ALARM ($)",
                         0, 200, 10, step=5, key="fp_cost",
                         help="Operational + customer-friction cost of "
                              "flagging a legit transaction.")

    t_opt, cost_opt, grid, costs = optimal_threshold(y, score, amt, c_fn, c_fp)
    tp, fp, fn, tn = confusion_at(y, score, t_opt)
    cost_naive = total_cost(y, score, 0.5, amt, c_fn, c_fp)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimal threshold", f"{t_opt:.3f}")
    m2.metric("Total cost at optimum", f"${cost_opt:,.0f}")
    m3.metric("vs naive t=0.50", f"${cost_naive - cost_opt:,.0f} saved")
    recall = tp / max(tp + fn, 1)
    m4.metric("Fraud caught (recall)", f"{recall*100:.1f}%")

    g1, g2 = st.columns([3, 2])
    with g1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(grid, costs, color=ACCENT, lw=2)
        ax.axvline(t_opt, ls="--", color="#333", label=f"optimal {t_opt:.3f}")
        ax.axvline(0.5, ls=":", color="#999", label="naive 0.50")
        ax.set_xlabel("decision threshold"); ax.set_ylabel("total cost ($)")
        ax.set_ylim(0, max(costs) * 1.15)
        ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with g2:
        cm = pd.DataFrame([[tn, fp], [fn, tp]],
                          index=["Actual legit", "Actual fraud"],
                          columns=["Pred legit", "Pred fraud"])
        st.markdown("**Confusion matrix at the optimal threshold**")
        st.dataframe(cm, use_container_width=True)
        st.caption(f"Caught {tp} fraud, missed {fn}. "
                   f"Flagged {fp} legit customers.")

# =============================== TAB 2 =======================================
with tab2:
    st.subheader("Imbalance strategy comparison")
    st.markdown(
        "<span class='small'>Same gradient-boosted model, same data, same seed. "
        "Only the imbalance strategy changes. PR-AUC is the headline because "
        "ROC-AUC flatters models on a 0.17%-positive problem.</span>",
        unsafe_allow_html=True)
    if results is not None:
        show = results[["strategy", "pr_auc", "roc_auc", "precision",
                        "recall", "f1", "threshold"]].copy()
        for c in ["pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"]:
            show[c] = show[c].map(lambda v: f"{v:.4f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

    stats = load_json("stats_tests.json")
    if stats and "comparison" in stats:
        cmp = stats["comparison"]
        verdict = ("a statistically significant difference"
                   if cmp["significant"]
                   else "no significant difference — the gap is within noise")
        st.markdown(
            f"**{cmp['a']}** vs **{cmp['b']}**: mean PR-AUC difference "
            f"{cmp['mean_diff']:+.4f}, 95% CI "
            f"[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}] — {verdict}.")

    pr = OUT / "pr_curves.png"
    if pr.exists():
        st.image(str(pr), caption="Precision-recall curves by strategy")

# =============================== TAB 3 =======================================
with tab3:
    st.subheader("Which signals actually separate fraud?")
    sep = load_csv("separability.csv")
    if sep is not None:
        top = sep.head(12)[["feature", "cohens_d", "ks_stat",
                            "separability_score"]].copy()
        for c in ["cohens_d", "ks_stat", "separability_score"]:
            top[c] = top[c].map(lambda v: f"{v:.3f}")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.dataframe(top, use_container_width=True, hide_index=True)
        with cc2:
            img = OUT / "separability_top.png"
            if img.exists():
                st.image(str(img))
    b = OUT / "class_balance.png"
    if b.exists():
        st.image(str(b), caption="The imbalance: fraud is a tiny fraction of rows")

# =============================== TAB 4 =======================================
with tab4:
    st.markdown("""
### The thesis
Accuracy is a trap on a 99.8%-legit dataset — predict "never fraud" and you're
99.8% accurate and useless. The job is to **rank** transactions by risk and then
**choose an operating threshold that minimizes total dollar cost**, given that a
missed fraud and a false alarm cost different amounts.

### Method
1. **Separability analysis** — rank features by how cleanly they pull fraud
   apart from legit (Cohen's d + KS statistic).
2. **Leak-free splits** — stratified train/val/test so the fraud rate is
   identical across all three; scaler fit on train only.
3. **Controlled imbalance experiment** — one model family (gradient-boosted
   trees), vary only the imbalance strategy: none, class weights, SMOTE,
   undersampling, threshold tuning.
4. **Statistical honesty** — bootstrap 95% CIs on every metric; paired
   bootstrap test on the top two strategies.
5. **Cost analysis** — explicit dollar cost matrix, threshold sweep, optimal
   operating point, and sensitivity to the cost assumptions.
6. **Explainability** — SHAP for global and per-transaction drivers.

### Why no deep learning
On tabular data, gradient-boosted trees beat neural nets in the large majority
of cases, train in seconds on a CPU, and hand you feature importances for free.
Reaching for a neural net here would be the wrong call — and knowing that is the
point.
""")
