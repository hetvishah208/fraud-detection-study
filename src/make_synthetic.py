"""
Generate a small, schema-faithful synthetic sample that matches the ULB
credit-card-fraud dataset (creditcard.csv) so the pipeline can be smoke-tested
without the real 150MB file.

Schema: Time, V1..V28, Amount, Class  (Class=1 is fraud, ~0.17% prevalence)

This is ONLY for testing the code path. Real numbers for the resume come from
running the pipeline on the real creditcard.csv.
"""
import numpy as np
import pandas as pd


def make_synthetic(n=20000, fraud_rate=0.0173, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    # 28 anonymized PCA-style features. We make a handful genuinely separable
    # between classes so the separability analysis has something real to find,
    # and leave the rest as pure noise (mirrors reality: only some V's matter).
    signal_features = [3, 9, 10, 11, 13, 16, 17]  # 1-indexed V columns w/ signal

    def make_block(n_rows, is_fraud):
        cols = {}
        for i in range(1, 29):
            if i in signal_features:
                # shift the fraud distribution for signal features
                shift = rng.uniform(1.2, 2.8) * (-1 if is_fraud else 0)
                cols[f"V{i}"] = rng.normal(loc=shift, scale=1.0, size=n_rows)
            else:
                cols[f"V{i}"] = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        # Amount: right-skewed; fraud tends to smaller, testing-style amounts
        if is_fraud:
            cols["Amount"] = rng.gamma(shape=1.2, scale=40.0, size=n_rows)
        else:
            cols["Amount"] = rng.gamma(shape=1.5, scale=60.0, size=n_rows)
        cols["Time"] = rng.uniform(0, 172792, size=n_rows)  # ~2 days in seconds
        cols["Class"] = np.full(n_rows, 1 if is_fraud else 0, dtype=int)
        return pd.DataFrame(cols)

    df = pd.concat(
        [make_block(n_legit, False), make_block(n_fraud, True)],
        ignore_index=True,
    )
    # reorder to canonical column order
    ordered = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = df[ordered].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = make_synthetic()
    out = "data/creditcard_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}  fraud={int(df.Class.sum())} "
          f"({df.Class.mean()*100:.3f}%)")
