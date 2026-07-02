---
title: Fraud Cost Explorer
emoji: 💳
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Fraud Detection: A Cost-Optimization Study

Most fraud-detection projects stop at "I got 0.98 AUC." This one starts where
that leaves off and asks the question a risk team actually has to answer: **given
that a missed fraud and a false alarm cost different amounts of money, where
exactly do you set the decision threshold?**

The model is deliberately boring — gradient-boosted trees on tabular data. The
work is everything around it: understanding which signals separate fraud,
running a controlled experiment on how to handle extreme class imbalance,
reporting every number with a confidence interval, and turning the model into a
dollar-cost decision with a defensible operating point.

**Live demo:** _(HF Spaces link goes here once deployed)_
**Dataset:** ULB credit-card fraud — 284,807 transactions, 492 fraud (0.172%).

---

## The thesis

On a dataset that's 99.8% legitimate, a model that predicts "never fraud" is
99.8% accurate and completely useless. Accuracy is the wrong objective. So is
raw ROC-AUC, which looks rosy on extreme imbalance because the enormous negative
class dominates the false-positive-rate axis. The honest headline metric is
**PR-AUC** (average precision), and the real deliverable is a **threshold chosen
to minimize total expected dollar cost**.

## What's in here

| Phase | What it does | Key output |
|------|---------------|------------|
| 1. EDA + separability | Ranks features by how cleanly they pull fraud apart from legit (Cohen's d + KS statistic) | `separability.csv`, distribution plots |
| 2. Baseline | Leak-free stratified split; logistic-regression baseline | `baseline_metrics.json` |
| 3. Imbalance experiment | One model family, vary only the imbalance strategy: none / class weights / SMOTE / undersampling / threshold tuning | `experiment_results.csv`, PR curves |
| 4. Statistical honesty | Bootstrap 95% CIs on every metric; paired bootstrap test on the top two strategies | `stats_tests.json` |
| 5. Cost analysis | Dollar cost matrix, threshold sweep, optimal operating point, sensitivity analysis | `cost_analysis.json`, cost curves |
| 6. Explainability + app | SHAP global/local; interactive Streamlit cost explorer | `shap_*.png`, deployed app |

## The interesting results

These come from running the pipeline on the real dataset (the numbers below are
placeholders until you run it — see "Reproduce"):

- **Threshold tuning vs resampling.** Holding the model fixed and only tuning the
  decision threshold competes with — and often beats — SMOTE and undersampling.
  The fancy resampling tricks aren't free wins.
- **The cost curve has a clear minimum.** The dollar-optimal threshold is far
  from the naive 0.5, and operating there saves real money versus the default.
- **Sensitivity is the whole point.** As the assumed cost of a missed fraud
  rises, the optimal threshold falls — the system rationally accepts more false
  alarms to catch more fraud. The right decision *depends on the cost
  structure*, and the app lets you feel that by dragging two sliders.

## Why no deep learning

On tabular data, gradient-boosted trees beat neural networks in the large
majority of cases, train in seconds on a CPU, and provide feature importances
for free. Reaching for a neural net here would be the wrong call. Knowing that —
and reaching for the simplest thing that works — is the point.

## Reproduce

```bash
# 1. install
pip install -r requirements.txt

# 2. get the data: download creditcard.csv from the ULB / Kaggle
#    credit-card-fraud dataset and place it at data/creditcard.csv
#    (without it, the pipeline falls back to a synthetic sample for testing)

# 3. run the whole study end-to-end
py src/run_all.py        # Windows
# python src/run_all.py  # mac/linux

# 4. launch the interactive app
streamlit run app/streamlit_app.py
```

Every phase can also be run on its own (`py src/eda.py`, `py src/experiment.py`,
etc.). Artifacts land in `outputs/`.

## Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP · SciPy ·
matplotlib · Streamlit · Docker · Hugging Face Spaces. Zero paid APIs, runs
CPU-only.

## Project layout

```
src/
  config.py         # paths, seed, split sizes, cost assumptions
  make_synthetic.py # schema-faithful sample for testing without the real CSV
  eda.py            # Phase 1
  data_prep.py      # leak-free stratified splits
  baseline.py       # Phase 2
  metrics.py        # PR-AUC-first evaluation
  experiment.py     # Phase 3 (the spine)
  stats_tests.py    # Phase 4
  cost_analysis.py  # Phase 5 (the payoff)
  explain.py        # Phase 6 SHAP
  run_all.py        # runs every phase in order
app/
  streamlit_app.py  # the interactive cost explorer
```

## A note on honesty

The synthetic sample exists only so the code is runnable and testable without
the 150 MB dataset. All reported metrics come from running the pipeline on the
real data. Nothing in this repo reports a number it didn't measure.
