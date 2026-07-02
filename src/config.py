"""
Shared configuration and data loading for the fraud detection study.

Single source of truth for paths, the random seed, the split sizes, and the
cost assumptions used in the cost analysis. Every other module imports from here
so that one change (e.g. a different cost matrix) propagates everywhere.
"""
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Real dataset (you download this) vs synthetic (auto-generated for testing).
REAL_CSV = DATA_DIR / "creditcard.csv"
SYNTHETIC_CSV = DATA_DIR / "creditcard_synthetic.csv"

# ---------------------------------------------------------------------------
# Reproducibility & splits
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20      # held-out test set
VAL_SIZE = 0.20       # validation carved from the remaining train portion

# ---------------------------------------------------------------------------
# Cost assumptions (dollars). Documented and defensible, not arbitrary.
#   - A missed fraud (false negative) costs the average fraudulent amount that
#     gets charged back, plus a fixed handling/chargeback overhead.
#   - A false alarm (false positive) costs the operational + customer-friction
#     cost of reviewing/declining a legitimate transaction.
# These are STARTING values. The whole point of the cost analysis is to show
# how the optimal decision threshold moves as these change, so they are knobs,
# not constants carved in stone.
# ---------------------------------------------------------------------------
COST_FALSE_NEGATIVE_FIXED = 100.0   # overhead per missed fraud, on top of $ amount
COST_FALSE_POSITIVE = 10.0          # cost of bothering a legit customer
# When the transaction Amount is available we add it to the FN cost (you lose
# the actual money). This is handled in the cost module.

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGET = "Class"
RAW_NUMERIC_TO_SCALE = ["Amount", "Time"]  # V1..V28 are already PCA-standardized


def load_data(prefer_real: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Load the dataset. Uses the real CSV if present, else the synthetic one.

    Returns the dataframe with canonical columns. Drops exact duplicate rows
    (the real ULB set contains ~1800 of them) and reports what it did.
    """
    if prefer_real and REAL_CSV.exists():
        path = REAL_CSV
        source = "REAL"
    elif SYNTHETIC_CSV.exists():
        path = SYNTHETIC_CSV
        source = "SYNTHETIC"
    else:
        raise FileNotFoundError(
            f"No data found. Put the real file at {REAL_CSV} "
            f"or run: py src/make_synthetic.py"
        )

    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    if verbose:
        frauds = int(df[TARGET].sum())
        print(f"[load_data] source={source}  path={path.name}")
        print(f"[load_data] rows={len(df):,}  (dropped {n_dupes:,} dup rows)")
        print(f"[load_data] fraud={frauds:,}  prevalence={df[TARGET].mean()*100:.3f}%")
    return df


def using_real_data() -> bool:
    return REAL_CSV.exists()
