"""Network-free checks for the optional ACS benchmark."""

import pandas as pd

from sentinelml.acs_validation import FEATURES, TARGET, partition_train_year, prepare_acs


def test_prepare_acs_uses_common_schema_and_adult_filter():
    raw = pd.DataFrame({name: [2, 3, 4, 5] for name in FEATURES})
    raw["AGEP"] = [16, 42, 39, 28]
    raw["WKHP"] = [40, 30, 0, 35]
    raw["PINCP"] = [1000, 60000, 80000, 12000]
    raw["PWGTP"] = [1, 1, 1, 1]
    raw["RELP"] = [1, 2, 3, 4]
    x, y = prepare_acs(raw)
    assert list(x.columns) == FEATURES
    assert x["AGEP"].tolist() == [42, 28]
    assert y.tolist() == [1, 0]
    reduced, labels = prepare_acs(raw, [name for name in FEATURES if name != "OCCP"])
    assert "OCCP" not in reduced
    assert labels.tolist() == y.tolist()


def test_acs_train_year_partitions_do_not_overlap():
    x = pd.DataFrame({name: [i % 5 + 1 for i in range(200)] for name in FEATURES})
    x["AGEP"] = range(20, 220)
    y = pd.Series([0, 1] * 100)
    parts = partition_train_year(x, y)
    assert sum(len(part) for part in parts.values()) == 200
    ids = [set(part["AGEP"]) for part in parts.values()]
    assert all(not left & right for i, left in enumerate(ids) for right in ids[i + 1:])
    assert all(TARGET in part for part in parts.values())
