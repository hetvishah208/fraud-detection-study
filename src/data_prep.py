"""
Phase 2a - Preprocessing & leak-free stratified splits.

Why stratify: at 0.17% prevalence, an ordinary random split can hand you a test
set with almost no fraud, making every downstream metric noise. Stratifying on
the label keeps the fraud rate identical across train/val/test.

Why scale only Amount and Time: V1..V28 are already PCA outputs (centered,
comparable scale). Amount and Time are raw and on wildly different scales, so we
standardize them. Critically, the scaler is fit on TRAIN ONLY and then applied
to val/test - fitting on the full data before splitting would leak test
information into training.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SEED, TEST_SIZE, VAL_SIZE, TARGET, RAW_NUMERIC_TO_SCALE


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: list


def make_splits(df: pd.DataFrame, verbose: bool = True) -> SplitData:
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # First carve off the test set.
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    # Then carve validation out of the remaining training portion.
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=val_relative,
        stratify=y_tr_full, random_state=SEED
    )

    # Scale raw numeric columns, fit on train only.
    to_scale = [c for c in RAW_NUMERIC_TO_SCALE if c in X.columns]
    scaler = StandardScaler()
    if to_scale:
        scaler.fit(X_train[to_scale])
        for part in (X_train, X_val, X_test):
            part.loc[:, to_scale] = scaler.transform(part[to_scale])

    if verbose:
        def rate(s): return f"{s.mean()*100:.3f}%"
        print(f"[split] train={len(y_train):,} (fraud {rate(y_train)})  "
              f"val={len(y_val):,} (fraud {rate(y_val)})  "
              f"test={len(y_test):,} (fraud {rate(y_test)})")
        print(f"[split] scaled columns: {to_scale}")

    return SplitData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        scaler=scaler, feature_names=list(X.columns),
    )
