"""Optional, CPU-only ACSIncome-derived validation on real year/state shifts.

This deliberately uses a nine-feature common schema: RELP was removed from ACS
after 2018, so silently using the stock ten-feature task in 2021 is invalid.
"""

import argparse
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sentinelml.modeling import metrics
from sentinelml.monitoring import (calibrate_size, corrected_rejections,
                                   empirical_pvalue, feature_scores, strong_single_batch_override)
from sentinelml.performance import degradation_evidence
from sentinelml.settings import ROOT, ARTIFACTS, config, save_json


FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "WKHP", "SEX", "RAC1P"]
CATEGORICAL = ["COW", "SCHL", "MAR", "OCCP", "POBP", "SEX", "RAC1P"]
TARGET = "income_over_50000"
METHODS = ["bh", "bonferroni", "by", "bh_override"]


def prepare_acs(raw, features=FEATURES):
    """Apply the stock ACSIncome adult/working filter on a common feature set."""
    needed = set(features + ["PINCP", "PWGTP"])
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise ValueError(f"ACS year lacks required common-schema columns: {missing}")
    selected = raw.loc[(raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)].copy()
    x = selected[features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    if x.isna().mean().max() > .25:
        raise ValueError("More than 25% of a common feature is missing/non-numeric; inspect year schema")
    y = (selected.PINCP > 50000).astype(int).reset_index(drop=True)
    if len(x) != len(y) or y.nunique() != 2:
        raise ValueError("ACS year has no usable binary labelled sample")
    return x, y


def load_acs_year(year, state, density, download, features=FEATURES):
    try:
        from folktables import ACSDataSource
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies: pip install -e ".[acs]"') from exc
    source = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person",
                           root_dir=str(ROOT / "data/raw/acs"))
    raw = source.get_data(states=[state], density=density, random_seed=42, download=download)
    x, y = prepare_acs(raw, features)
    return x, y, {"year": year, "state": state, "raw_rows_sampled": len(raw),
                  "task_rows": len(x), "density": density}


def partition_train_year(x, y):
    data = x.assign(**{TARGET: y})
    train, rest = train_test_split(data, train_size=.4, stratify=data[TARGET], random_state=42)
    baseline, rest = train_test_split(rest, train_size=1/6, stratify=rest[TARGET], random_state=43)
    reference, rest = train_test_split(rest, train_size=.3, stratify=rest[TARGET], random_state=44)
    calibration, stable = train_test_split(rest, train_size=3/7, stratify=rest[TARGET], random_state=45)
    parts = {"train": train, "baseline": baseline, "reference": reference,
             "calibration": calibration, "stable": stable}
    if sum(len(p) for p in parts.values()) != len(data):
        raise AssertionError("Training-year partition lost rows")
    return parts


def xgboost_pipeline(features=FEATURES):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies: pip install -e ".[acs]"') from exc
    categorical = [name for name in CATEGORICAL if name in features]
    preprocess = ColumnTransformer([
        ("numerical", SimpleImputer(strategy="median"), [name for name in ["AGEP", "WKHP"] if name in features]),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], sparse_threshold=1.0)
    model = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=.08,
                          subsample=.8, colsample_bytree=.8, tree_method="hist",
                          n_jobs=2, eval_metric="logloss", random_state=42)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate_stream(x, y, model, reference, null, baseline, batch_size, max_batches, seed,
                    features=FEATURES):
    """Use non-overlapping rows within this one stream; carry alert state forward."""
    cfg = config()
    categorical = [name for name in CATEGORICAL if name in features]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    batches = min(len(order) // batch_size, max_batches)
    streaks = {method: {} for method in METHODS}
    rows = []
    for index in range(batches):
        take = order[index * batch_size:(index + 1) * batch_size]
        current_x, current_y = x.iloc[take], y.iloc[take]
        scores = feature_scores(reference, current_x, details=False,
                                features=features, categorical=categorical)
        pvalues = {name: empirical_pvalue(row["score"], null[name]) for name, row in scores.items()}
        rejected = {method: corrected_rejections(pvalues, cfg["feature_fdr"], method)
                    for method in ["bh", "bonferroni", "by"]}
        rejected["bh_override"] = rejected["bh"]
        override = strong_single_batch_override(pvalues, rejected["bh"],
            len(next(iter(null.values()))), cfg.get("strong_override_min_features", 3))
        row = {"batch": index + 1, "rows": len(take), "positive_labels": int(current_y.sum()),
               "strong_override": override}
        for method in METHODS:
            streaks[method] = {name: streaks[method].get(name, 0) + 1 for name in rejected[method]}
            row[f"{method}_flag"] = bool(rejected[method])
            row[f"{method}_alert"] = (any(count >= cfg["persistence_batches"] for count in streaks[method].values())
                                      or (method == "bh_override" and override))
        probability = model.predict_proba(current_x)[:, 1]
        measured = metrics(current_y, probability)
        evidence = degradation_evidence(current_y.to_numpy(), probability, baseline,
                                        cfg["minimum_positive_labels"], 200, seed + index)
        row.update({"accuracy": float(accuracy_score(current_y, probability >= .5)),
                    "f1": measured["f1"], "recall": measured["recall"],
                    "roc_auc": measured["roc_auc"], "degraded": evidence["degraded"]})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_stream(table):
    if table.empty:
        return {"batches": 0, "reason": "Too few rows for a full batch"}
    eligible = table[table.degraded.notna()]
    result = {"batches": len(table), "rows": int(table.rows.sum()),
              "mean_accuracy": float(table.accuracy.mean()),
              "mean_f1": float(table.f1.mean()), "mean_recall": float(table.recall.mean()),
              "mean_roc_auc": float(table.roc_auc.mean()),
              "degradation_evaluated_batches": len(eligible)}
    for method in METHODS:
        result[method] = {"flag_batches": int(table[f"{method}_flag"].sum()),
                          "actionable_batches": int(table[f"{method}_alert"].sum()),
                          "drift_without_confirmed_degradation": int((~eligible.degraded.astype(bool) & eligible[f"{method}_alert"]).sum()),
                          "degradation_without_drift_alert": int((eligible.degraded.astype(bool) & ~eligible[f"{method}_alert"]).sum())}
    return result


def run(train_state="CA", years=(2015, 2016, 2017, 2018), compare_state=None,
        compare_year=2018, density=.5, batch_size=500, max_batches=200,
        calibration_draws=1000, download=False, exclude_occupation=False):
    if not 0 < density <= 1 or batch_size < 100 or max_batches < 1 or calibration_draws < 500:
        raise ValueError("Invalid density, batch size, batch limit or calibration count")
    features = [name for name in FEATURES if not (exclude_occupation and name == "OCCP")]
    categorical = [name for name in CATEGORICAL if name in features]
    train_x, train_y, provenance = load_acs_year(2014, train_state, density, download, features)
    parts = partition_train_year(train_x, train_y)
    if min(len(parts["reference"]), len(parts["calibration"]), len(parts["stable"])) < batch_size:
        raise ValueError("Increase density or choose a larger state: a partition is smaller than one batch")
    model = xgboost_pipeline(features)
    model.fit(parts["train"][features], parts["train"][TARGET])
    baseline_p = model.predict_proba(parts["baseline"][features])[:, 1]
    baseline = metrics(parts["baseline"][TARGET], baseline_p)
    reference = parts["reference"][features]
    null = calibrate_size(reference, parts["calibration"][features], batch_size,
                          calibration_draws, features=features, categorical=categorical, seed=42)
    output = ARTIFACTS / ("acs_validation_no_occupation" if exclude_occupation else "acs_validation")
    output.mkdir(parents=True, exist_ok=True)
    stable = evaluate_stream(parts["stable"][features], parts["stable"][TARGET], model,
                             reference, null, baseline, batch_size, max_batches, 100, features)
    stable.to_csv(output / f"{train_state}_2014_stable_batches.csv", index=False)
    summary = {"task": "ACSIncome-derived common-schema task (RELP excluded in every year)",
               "train_year": 2014, "train_state": train_state, "train_source": provenance,
               "partition_sizes": {k: len(v) for k, v in parts.items()},
               "features": features, "categorical": categorical,
               "occupation_excluded": exclude_occupation,
               "model": "XGBoost CPU histogram trees", "folktables_version": version("folktables"),
               "xgboost_version": version("xgboost"), "density": density,
               "batch_size": batch_size, "max_batches_per_stream": max_batches,
               "calibration_draws": calibration_draws, "baseline": baseline,
               "streams": {f"{train_state}-2014-heldout": summarize_stream(stable)}}
    save_json(output / "summary.json", summary)
    targets = [(year, train_state) for year in years]
    if compare_state:
        targets.append((compare_year, compare_state))
    for year, state in targets:
        current_x, current_y, source = load_acs_year(year, state, density, download, features)
        frame = evaluate_stream(current_x, current_y, model, reference, null, baseline,
                                batch_size, max_batches, year * 100 + sum(map(ord, state)), features)
        frame.to_csv(output / f"{state}_{year}_batches.csv", index=False)
        summary["streams"][f"{state}-{year}"] = {**summarize_stream(frame), "source": source}
        save_json(output / "summary.json", summary)
        print(f"Completed ACS {state} {year}: {len(frame)} disjoint batches", flush=True)
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Optional ACSIncome-derived temporal/geographic validation")
    parser.add_argument("--train-state", default="CA")
    parser.add_argument("--years", type=int, nargs="+", default=[2015, 2016, 2017, 2018])
    parser.add_argument("--compare-state")
    parser.add_argument("--compare-year", type=int, default=2018)
    parser.add_argument("--density", type=float, default=.5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--calibration-draws", type=int, default=1000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--exclude-occupation", action="store_true",
                        help="Sensitivity run: remove OCCP in all years to avoid the 2018 codebook break")
    args = parser.parse_args()
    summary = run(args.train_state, args.years, args.compare_state, args.compare_year,
                  args.density, args.batch_size, args.max_batches, args.calibration_draws,
                  args.download, args.exclude_occupation)
    print(json.dumps({"task": summary["task"], "partition_sizes": summary["partition_sizes"],
                      "streams": summary["streams"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
