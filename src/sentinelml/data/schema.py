import numpy as np
import pandas as pd

from sentinelml.settings import FEATURES


def validate(frame):
    """Reject structural/nonnumeric input; allow missing values for imputation."""
    if frame.empty:
        raise ValueError("Batch must contain at least one row")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES) - {"ID"})
    if missing or extra:
        raise ValueError(f"Missing features: {missing}; unexpected columns: {extra}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate columns are not allowed")
    try:
        result = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Features must be numeric or null") from exc
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite feature values are not allowed")
    return result


def quality(frame):
    invalid = {}
    for column, allowed in {"SEX": [1, 2], "EDUCATION": [1, 2, 3, 4], "MARRIAGE": [1, 2, 3]}.items():
        invalid[column] = float((frame[column].notna() & ~frame[column].isin(allowed)).mean())
    for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        invalid[column] = float((frame[column].notna() & ~frame[column].isin(range(-2, 10))).mean())
    for column in ["LIMIT_BAL", *[f"PAY_AMT{i}" for i in range(1, 7)]]:
        invalid[column] = float((frame[column] < 0).mean())
    invalid["AGE"] = float(((frame.AGE < 18) | (frame.AGE > 120)).mean())
    return {"missing": frame.isna().mean().to_dict(), "invalid": invalid,
            "duplicate_feature_rows": float(frame.duplicated().mean())}
