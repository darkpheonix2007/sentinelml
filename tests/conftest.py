import numpy as np
import pandas as pd
import pytest

from sentinelml.settings import FEATURES, TARGET, config
from sentinelml.modeling import pipeline, metrics
from sentinelml.storage import Store
from sentinelml.service import Monitor


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    result = pd.DataFrame({col: rng.normal(100, 20, 300) for col in FEATURES})
    result["SEX"] = rng.choice([1, 2], len(result))
    result["EDUCATION"] = rng.choice([1, 2, 3, 4], len(result))
    result["MARRIAGE"] = rng.choice([1, 2, 3], len(result))
    result["AGE"] = rng.integers(21, 70, len(result))
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        result[col] = rng.integers(-2, 4, len(result))
    return result


@pytest.fixture
def bundle(frame):
    y = (frame.PAY_0 > 0).astype(int)
    model = pipeline("logistic")
    model.fit(frame, y)
    return {"model": model, "reference": frame.copy(), "thresholds": {"250": {f: .2 for f in FEATURES}},
            "null_scores": {"250": {f: np.linspace(0, .2, 500).tolist() for f in FEATURES}},
            "baseline": metrics(y, model.predict_proba(frame)[:, 1]),
            "metadata": {"version": "test-v1", "config": config()}}


@pytest.fixture
def monitor(bundle, tmp_path):
    return Monitor(bundle, Store(tmp_path / "test.sqlite"))
