import numpy as np
import pytest

from sentinelml.data.schema import validate
from sentinelml.modeling import partition, metrics
from sentinelml.monitoring import (inspect_batch, feature_scores, histogram, bh_rejections,
                                   corrected_rejections, strong_single_batch_override)
from sentinelml.settings import TARGET
from sentinelml.simulation import simulate


def test_identical_distributions_no_alert(frame, bundle):
    report, probability = inspect_batch(bundle, frame)
    assert not report["drift_alert"]
    assert not report["alerts"]
    assert len(probability) == len(frame)
    assert all(row["score"] == 0 for row in report["features"].values())


def test_shift_is_detected(frame, bundle):
    frame["AGE"] += 50
    report, _ = inspect_batch(bundle, frame)
    assert report["features"]["AGE"]["drifted"]
    assert any(a["feature"] == "AGE" for a in report["alerts"])


def test_bh_step_up_and_persistence(frame, bundle):
    assert bh_rejections({"a": .001, "b": .03, "c": .3}, .05) == {"a", "b"}
    pvalues = {"a": .001, "b": .03, "c": .3}
    assert corrected_rejections(pvalues, .05, "bonferroni") == {"a"}
    assert corrected_rejections(pvalues, .05, "by") == {"a"}
    shifted = frame.copy()
    shifted["AGE"] += 50
    first, _ = inspect_batch(bundle, shifted)
    assert first["features"]["AGE"]["drifted"]
    assert any(a["feature"] == "AGE" and a["severity"] == "warning" for a in first["alerts"])
    second, _ = inspect_batch(bundle, shifted, first["flag_streaks"])
    assert any(a["feature"] == "AGE" and a["severity"] == "critical" for a in second["alerts"])
    cleared, _ = inspect_batch(bundle, frame, second["flag_streaks"])
    assert cleared["flag_streaks"] == {}


def test_strong_single_batch_override_needs_three_floor_pvalues():
    floor = 1 / 501
    pvalues = {"a": floor, "b": floor, "c": floor, "d": .2}
    assert strong_single_batch_override(pvalues, {"a", "b", "c"}, 500, 3)
    assert not strong_single_batch_override(pvalues, {"a", "b"}, 500, 3)
    assert not strong_single_batch_override({"a": floor * 2, "b": floor, "c": floor},
                                            {"a", "b", "c"}, 500, 3)


def test_strong_override_raises_critical_on_first_batch(frame, bundle):
    bundle["metadata"]["config"]["strong_override_enabled"] = True
    shifted = frame.copy()
    for name in ["PAY_0", "PAY_2", "PAY_3"]:
        shifted[name] += 5
    report, _ = inspect_batch(bundle, shifted)
    assert report["single_batch_override"]
    assert report["drift_alert"]
    assert len([a for a in report["alerts"] if a["category"] == "feature_drift"
                and a["severity"] == "critical" and a["single_batch_override"]]) >= 3


def test_strong_override_is_not_enabled_in_live_default(frame, bundle):
    shifted = frame.copy()
    for name in ["PAY_0", "PAY_2", "PAY_3"]:
        shifted[name] += 5
    report, _ = inspect_batch(bundle, shifted)
    assert not report["single_batch_override"]
    assert not any(a["category"] == "feature_drift" and a["severity"] == "critical"
                   for a in report["alerts"])


def test_categorical_frequency_table_preserves_counts():
    import pandas as pd
    reference = pd.Series([1, 1, 3, None], name="COW")
    current = pd.Series([1, 2, 2, None], name="COW")
    assert histogram(reference, current, categorical=["COW"]) == ([2, 0, 1], [1, 2, 0], ["1.0", "2.0", "3.0"])


def test_small_batch_insufficient(frame, bundle):
    frame["AGE"] += 50
    report, _ = inspect_batch(bundle, frame.head(3))
    assert not report["sufficient_evidence"]
    assert not report["features"]["AGE"]["drifted"]


def test_missing_and_unknown_categories(frame, bundle):
    frame["AGE"] = np.nan
    frame["SEX"] = 99
    report, p = inspect_batch(bundle, frame)
    assert np.isfinite(p).all()
    assert report["features"]["SEX"]["unseen_rate"] == 1
    assert any(a["category"] == "missingness" for a in report["alerts"])


@pytest.mark.parametrize("problem", ["missing", "infinity", "text", "target", "empty"])
def test_invalid_input(frame, problem):
    if problem == "missing":
        frame = frame.drop(columns="AGE")
    elif problem == "infinity":
        frame["AGE"] = np.inf
    elif problem == "text":
        frame["AGE"] = "bad"
    elif problem == "target":
        frame[TARGET] = 0
    else:
        frame = frame.head(0)
    with pytest.raises(ValueError):
        validate(frame)


def test_constant_features_finite(frame):
    frame["AGE"] = 20
    other = frame.copy()
    other["AGE"] = 21
    report = feature_scores(frame, other)
    assert np.isfinite(report["AGE"]["score"])
    assert report["AGE"]["score"] > 0


def test_partitions_disjoint_and_reproducible(frame):
    frame[TARGET] = np.arange(len(frame)) % 2
    frame["ID"] = np.arange(len(frame))
    parts = partition(frame)
    all_ids = [i for part in parts.values() for i in part.ID]
    assert len(all_ids) == len(set(all_ids)) == len(frame)
    again = partition(frame)
    assert all(parts[k].ID.tolist() == again[k].ID.tolist() for k in parts)


def test_simulation_preserves_inputs_for_label_shift(frame):
    frame[TARGET] = np.arange(len(frame)) % 2
    frame["ID"] = np.arange(len(frame))
    x, y, ids = simulate(frame, "stable", 100, seed=12)
    x2, y2, ids2 = simulate(frame, "label_shift", 100, seed=12)
    assert x.equals(x2)
    assert ids == ids2
    assert (y != y2).any()


def test_fractional_repayment_shift_keeps_integer_statuses(frame):
    frame[TARGET] = np.arange(len(frame)) % 2
    frame["ID"] = np.arange(len(frame))
    stable, _, ids = simulate(frame, "stable", 100, seed=13)
    shifted, _, shifted_ids = simulate(frame, "fractional_repayment_shift", 100, .25, seed=13)
    assert ids == shifted_ids
    for column in ["PAY_0", "PAY_2", "PAY_3"]:
        difference = shifted[column] - stable[column]
        assert difference.isin([0, 1]).all()
        assert difference.sum() > 0
    assert shifted["PAY_0"].between(-2, 9).all()


def test_single_class_metrics():
    result = metrics([0, 0], [.2, .3])
    assert result["roc_auc"] is None
    assert result["confusion_matrix"] == [[2, 0], [0, 0]]
