"""Synthetic stress scenarios, not claims of realistic temporal credit drift."""
import numpy as np

from sentinelml.settings import FEATURES, TARGET

SCENARIOS = ["stable", "age_shift", "repayment_shift", "fractional_repayment_shift", "missingness", "label_shift"]


def simulate(pool, scenario="stable", size=500, strength=1.0, seed=42):
    if scenario not in SCENARIOS:
        raise ValueError(f"Choose a scenario from {SCENARIOS}")
    if not 1 <= size <= len(pool):
        raise ValueError(f"Batch size must be between 1 and {len(pool)}")
    if not 0 <= strength <= 3:
        raise ValueError("Strength must be between 0 and 3")
    rng = np.random.default_rng(seed)
    sample = pool.iloc[rng.choice(len(pool), size, replace=False)].copy()
    x = sample[FEATURES].astype(float)
    y = sample[TARGET].astype(int).to_numpy(copy=True)
    if scenario == "age_shift":
        x["AGE"] = (x.AGE + 10 * strength).clip(18, 120)
    elif scenario == "repayment_shift":
        for col in ["PAY_0", "PAY_2", "PAY_3"]:
            x[col] = (x[col] + round(2 * strength)).clip(-2, 9)
    elif scenario == "fractional_repayment_shift":
        if strength > 1:
            raise ValueError("Fractional repayment shift strength must be at most 1")
        # PAY status is integer-valued. A shared Bernoulli mask gives an expected
        # increment of +strength per feature without inventing invalid decimals.
        mask = rng.random(size) < strength
        for col in ["PAY_0", "PAY_2", "PAY_3"]:
            x.loc[mask, col] = (x.loc[mask, col] + 1).clip(-2, 9)
    elif scenario == "missingness":
        rows = rng.choice(size, int(size * min(.3 * strength, 1)), replace=False)
        x.iloc[rows, x.columns.get_indexer(["PAY_0", "LIMIT_BAL"])] = np.nan
    elif scenario == "label_shift":
        # Synthetic label corruption demonstrates a blind spot: X does not change.
        flip = rng.random(size) < min(.3 * strength, 1)
        y[flip] = 1 - y[flip]
    return x.reset_index(drop=True), y, sample.ID.astype(int).tolist()
