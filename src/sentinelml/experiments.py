"""Repeated experiments and an explicit, auditable candidate promotion gate."""
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import recall_score

from sentinelml.modeling import load_bundle, metrics, pipeline
from sentinelml.monitoring import (inspect_batch, feature_scores, empirical_pvalue,
                                   corrected_rejections, calibrate_size, strong_single_batch_override)
from sentinelml.performance import degradation_evidence
from sentinelml.settings import ARTIFACTS, FEATURES, TARGET, config, save_json
from sentinelml.simulation import simulate


def wilson(successes, total):
    if total == 0:
        return [None, None]
    z = float(norm.ppf(.975))
    p = successes / total
    mid = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [float(mid-half), float(mid+half)]


def benchmark_alerts(batches=500, size=500):
    """Paired legacy/new alert measurements on independent seeded batch sequences."""
    if batches < 2 or size < 100:
        raise ValueError("Use at least two batches of at least 100 rows")
    bundle = load_bundle()
    if "null_scores" not in bundle:
        raise ValueError("Run recalibrate-monitor first")
    rows = []
    for scenario in ["stable", "repayment_shift"]:
        streaks = {}  # start each scenario as a fresh monitoring stream
        for index in range(batches):
            seed = 30000 + index  # same source rows in the two scenarios
            x, _, source_ids = simulate(bundle["parts"]["simulation"], scenario, size, 1.0, seed)
            report, _ = inspect_batch(bundle, x, streaks)
            streaks = report["flag_streaks"]
            quality_warning = any(a["category"] in {"missingness", "unseen_category", "invalid_values"}
                                  for a in report["alerts"])
            legacy = bool(report["raw_feature_exceedances"] or quality_warning)
            actionable = any(a["category"] == "feature_drift" and a["severity"] == "critical"
                             for a in report["alerts"])
            warning = any(a["severity"] == "warning" for a in report["alerts"])
            rows.append({"scenario": scenario, "batch_number": index + 1, "seed": seed,
                         "source_ids": json.dumps(source_ids), "legacy_alert": legacy,
                         "corrected_feature_flag": bool(report["flagged_features"]),
                         "warning": warning, "actionable_alert": actionable})
            if (index + 1) % 100 == 0:
                print(f"{scenario}: {index + 1}/{batches}", flush=True)
    table = pd.DataFrame(rows)
    summary = {"model_version": bundle["metadata"]["version"], "batch_size": size,
               "batches_per_scenario": batches, "correction": "BH empirical p-values, q=" + str(bundle["metadata"]["config"]["feature_fdr"]),
               "persistence_batches": bundle["metadata"]["config"]["persistence_batches"],
               "null_draws_per_size": bundle["metadata"]["config"]["calibration_repetitions"]}
    for scenario, prefix in [("stable", "false_alert"), ("repayment_shift", "detection")]:
        block = table[table.scenario == scenario]
        for name, column in [("before", "legacy_alert"), ("after", "actionable_alert")]:
            flagged = block[column].to_numpy(dtype=bool)
            summary[f"{prefix}_rate_{name}"] = float(flagged.mean())
            summary[f"{prefix}_count_{name}"] = int(flagged.sum())
            summary[f"{prefix}_first_batch_{name}"] = int(np.flatnonzero(flagged)[0] + 1) if flagged.any() else None
        summary[f"{prefix}_warning_rate_after"] = float(block.warning.mean())
    output = ARTIFACTS / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "alert_rule_batches.csv", index=False)
    save_json(output / "alert_rule_comparison.json", summary)
    return summary


def benchmark_shift_sweep(repetitions=100, size=500, stable_batches=500, null_draws=None):
    """Paired weak-shift episode sweep and feature-rule correction baselines."""
    if repetitions < 2 or stable_batches < 2 or size < 100:
        raise ValueError("Use at least two episodes/stable batches and at least 100 rows per batch")
    bundle = load_bundle()
    if "null_scores" not in bundle:
        raise ValueError("Run recalibrate-monitor first")
    if size > len(bundle["parts"]["simulation"]):
        raise ValueError("Batch size exceeds the simulation pool")
    cfg = bundle["metadata"]["config"]
    calibrated_size = min(bundle["null_scores"], key=lambda n: abs(int(n) - size))
    if null_draws is not None and null_draws < 500:
        raise ValueError("Use at least 500 null draws for the comparison")
    null = (calibrate_size(bundle["reference"], bundle["parts"]["calibration"][FEATURES],
                           size, null_draws, seed=cfg["seed"])
            if null_draws is not None else bundle["null_scores"][calibrated_size])
    actual_draws = len(next(iter(null.values())))
    methods = ["bh", "bonferroni", "by", "bh_override"]

    def flags(x):
        scores = feature_scores(bundle["reference"], x, details=False)
        pvalues = {name: empirical_pvalue(row["score"], null[name])
                   for name, row in scores.items()}
        corrected = {method: corrected_rejections(pvalues, cfg["feature_fdr"], method)
                     for method in ["bh", "bonferroni", "by"]}
        corrected["bh_override"] = corrected["bh"]
        corrected["_override"] = strong_single_batch_override(
            pvalues, corrected["bh"], actual_draws, cfg.get("strong_override_min_features", 3))
        return corrected

    def advance(previous, current):
        streak = {name: previous.get(name, 0) + 1 for name in current}
        return streak, any(count >= cfg["persistence_batches"] for count in streak.values())

    pool = bundle["parts"]["simulation"]
    stable_counts = {method: 0 for method in methods}
    stable_flags = {method: 0 for method in methods}
    stable_override_count = 0
    streaks = {method: {} for method in methods}
    for index in range(stable_batches):
        x, _, _ = simulate(pool, "stable", size, 0, 50000 + index)
        current = flags(x)
        stable_override_count += current["_override"]
        for method in methods:
            stable_flags[method] += bool(current[method])
            streaks[method], fired = advance(streaks[method], current[method])
            if method == "bh_override":
                fired = fired or current["_override"]
            stable_counts[method] += fired
        if (index + 1) % 100 == 0:
            print(f"sweep stable: {index + 1}/{stable_batches}", flush=True)

    episode_rows = []
    for repeat in range(repetitions):
        for strength in [.05, .10, .25, .5, 1.0]:
            # Each event begins clean: persistence alone cannot detect a one-batch event.
            state = {method: {} for method in methods}
            steps = {method: [] for method in methods}
            warnings = {method: [] for method in methods}
            overrides = []
            for position in range(3):
                seed = 60000 + repeat * 4 + 1 + position
                x, _, _ = simulate(pool, "fractional_repayment_shift", size, strength, seed)
                current = flags(x)
                overrides.append(current["_override"])
                for method in methods:
                    warnings[method].append(bool(current[method]))
                    state[method], fired = advance(state[method], current[method])
                    if method == "bh_override":
                        fired = fired or current["_override"]
                    steps[method].append(fired)
            for duration in [1, 2, 3]:
                row = {"repeat": repeat, "strength": strength, "duration": duration,
                       "initial_alert_state": "clean", "override_triggered": any(overrides[:duration]),
                       "shift_seeds": json.dumps([60000 + repeat * 4 + 1 + i for i in range(duration)])}
                for method in methods:
                    observed = steps[method][:duration]
                    row[f"{method}_detected"] = any(observed)
                    row[f"{method}_delay"] = observed.index(True) + 1 if any(observed) else None
                    row[f"{method}_warning"] = any(warnings[method][:duration])
                episode_rows.append(row)
        if (repeat + 1) % 20 == 0:
            print(f"sweep episodes: {repeat + 1}/{repetitions}", flush=True)

    episodes = pd.DataFrame(episode_rows)
    summary = {"model_version": bundle["metadata"]["version"], "batch_size": size,
               "stable_batches": stable_batches, "episodes_per_setting": repetitions,
               "calibrated_batch_size": int(calibrated_size), "null_draws": actual_draws,
               "q": cfg["feature_fdr"], "persistence_batches": cfg["persistence_batches"],
               "strong_override_min_features": cfg.get("strong_override_min_features", 3),
               "stable_override_count": stable_override_count,
               "shift_definition": "Each PAY_0/PAY_2/PAY_3 row gets a shared +1 integer increment with probability strength, clipped at 9; expected pre-clipping increment equals strength.",
               "stable": {method: {"flag_count": stable_flags[method], "flag_rate": stable_flags[method] / stable_batches,
                                    "actionable_count": stable_counts[method],
                                    "actionable_rate": stable_counts[method] / stable_batches,
                                    "wilson95_if_independent": wilson(stable_counts[method], stable_batches)}
                          for method in methods}, "groups": []}
    for (strength, duration), group in episodes.groupby(["strength", "duration"]):
        entry = {"strength": float(strength), "duration": int(duration), "episodes": len(group)}
        for method in methods:
            detected = int(group[f"{method}_detected"].sum())
            delays = group[f"{method}_delay"].dropna()
            entry[method] = {"detected": detected, "rate": detected / len(group),
                             "mean_delay_when_detected": float(delays.mean()) if len(delays) else None,
                             "warning_episodes": int(group[f"{method}_warning"].sum())}
        summary["groups"].append(entry)
    output = ARTIFACTS / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{actual_draws}" if null_draws is not None else ""
    episodes.to_csv(output / f"shift_sweep_episodes{suffix}.csv", index=False)
    save_json(output / f"shift_sweep_summary{suffix}.json", summary)
    return summary


def study(repetitions=30):
    if repetitions < 2:
        raise ValueError("Use at least two repetitions")
    bundle = load_bundle()
    rows = []
    for size in [100, 250, 500, 1000]:
        for scenario in ["stable", "age_shift", "repayment_shift", "missingness", "label_shift"]:
            for strength in ([0] if scenario == "stable" else [.5, 1.0, 2.0]):
                streaks = {}
                for repeat in range(repetitions):
                    seed = 10000 + repeat  # distinct from calibration draws; paired across scenarios
                    x, y, source_ids = simulate(bundle["parts"]["simulation"], scenario, size, strength, seed)
                    report, probability = inspect_batch(bundle, x, streaks)
                    streaks = report["flag_streaks"]
                    score = metrics(y, probability)
                    cfg = bundle["metadata"]["config"]
                    evidence = degradation_evidence(y, probability, bundle["baseline"],
                        cfg["minimum_positive_labels"], cfg["bootstrap_repetitions"], seed)
                    degradation = evidence["degraded"]
                    actionable = any(a["severity"] == "critical" for a in report["alerts"])
                    rows.append({"scenario": scenario, "strength": strength, "size": size, "seed": seed,
                                 "model_version": bundle["metadata"]["version"],
                                 "source_ids": json.dumps(source_ids), "drift_fraction": report["drift_fraction"],
                                 "feature_drift_alert": report["drift_alert"],
                                 "any_monitor_alert": actionable, "degradation": degradation,
                                 **{k: score[k] for k in ["precision", "recall", "f1", "roc_auc", "average_precision"]}})
                print(f"Completed {scenario}, strength={strength}, batch={size}", flush=True)
    table = pd.DataFrame(rows)
    output = ARTIFACTS / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "batches.csv", index=False)
    summaries = []
    for (scenario, strength, size), group in table.groupby(["scenario", "strength", "size"]):
        count = int(group.any_monitor_alert.sum())
        summaries.append({"scenario": scenario, "strength": strength, "size": int(size),
                          "runs": len(group), "monitor_alert_rate": count / len(group),
                          "alert_rate_wilson95": wilson(count, len(group)),
                          "feature_drift_alert_rate": float(group.feature_drift_alert.mean()),
                          "degradation_rate": float(group.degradation.dropna().mean()) if group.degradation.notna().any() else None,
                          "degradation_evaluated": int(group.degradation.notna().sum()),
                          "mean_f1": float(group.f1.mean())})
    stable = table[table.scenario == "stable"]
    eligible = table[table.degradation.notna()]
    actual, predicted = eligible.degradation.astype(bool), eligible.any_monitor_alert.astype(bool)
    tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
    fn, tn = int((actual & ~predicted).sum()), int((~actual & ~predicted).sum())
    summary = {"model_version": bundle["metadata"]["version"], "repetitions": repetitions,
               "total_batches": len(table), "degradation_evaluated_batches": len(eligible),
               "stable_false_alert_rate": float(stable.any_monitor_alert.mean()),
               "degradation_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
               "alert_precision_for_degradation": tp/(tp+fp) if tp+fp else None,
               "alert_recall_for_degradation": tp/(tp+fn) if tp+fn else None,
               "groups": summaries,
               "limitations": ["Repeated batches reuse rows from a finite simulation pool; intervals are descriptive, not independent population estimates.",
                               "Feature perturbations retain historical labels: synthetic stress, not realistic causal shifts.",
                               "label_shift flips labels artificially; it is not a natural label-prevalence experiment.",
                               "Degradation requires enough positives and a batch bootstrap upper bound below the fixed validation point estimate; baseline uncertainty is not included.",
                               "Single-feature drift need not trigger the aggregate feature-drift flag; individual alerts are also evaluated."]}
    save_json(output / "summary.json", summary)
    lines = ["# Measured monitoring experiments", "", f"Model: `{summary['model_version']}`", "",
             f"Batches: {len(table)}. Repetitions per configuration: {repetitions}.",
             f"Stable false-alert rate: {summary['stable_false_alert_rate']:.3f}", "",
             "| Scenario | Strength | Batch size | Any-alert rate | Degradation rate | Mean F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        rate = f"{row['degradation_rate']:.3f}" if row["degradation_rate"] is not None else "n/a"
        lines.append(f"| {row['scenario']} | {row['strength']} | {row['size']} | {row['monitor_alert_rate']:.3f} | {rate} | {row['mean_f1']:.3f} |")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in summary["limitations"]]])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def compare_candidate(scenario="repayment_shift", strength=1.0, bootstrap=500):
    bundle = load_bundle()
    # No evaluation labels are used for training. Simulation represents labels available by cutoff.
    recent, labels, source_ids = simulate(bundle["parts"]["simulation"], scenario,
                                         len(bundle["parts"]["simulation"]), strength, 701)
    original = bundle["parts"]["train"]
    xtrain = pd.concat([original[FEATURES], recent], ignore_index=True)
    ytrain = np.concatenate([original[TARGET], labels])
    candidate = pipeline(bundle["metadata"]["selected"], seed=42)
    start = time.perf_counter()
    candidate.fit(xtrain, ytrain)
    training_seconds = time.perf_counter() - start
    evaluation, y, evaluation_ids = simulate(bundle["parts"]["evaluation"], scenario,
                                            len(bundle["parts"]["evaluation"]), strength, 702)
    before = bundle["model"].predict_proba(evaluation)[:, 1]
    candidate.predict_proba(evaluation.iloc[:20])  # warm up
    start = time.perf_counter()
    after = candidate.predict_proba(evaluation)[:, 1]
    latency = (time.perf_counter() - start) * 1000 / len(evaluation)
    current_metrics, candidate_metrics = metrics(y, before), metrics(y, after)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        differences.append(float(recall_score(y[indices], after[indices] >= .5, zero_division=0) -
                                 recall_score(y[indices], before[indices] >= .5, zero_division=0)))
    if not differences:
        raise ValueError("Insufficient bootstrap samples")
    interval = np.quantile(differences, [.025, .975]).tolist()
    cfg = bundle["metadata"]["config"]
    checks = {"recall_gain": candidate_metrics["recall"] - current_metrics["recall"] >= cfg["promotion_minimum_recall_gain"],
              "no_f1_regression": candidate_metrics["f1"] >= current_metrics["f1"],
              "auc_tolerance": candidate_metrics["roc_auc"] >= current_metrics["roc_auc"] - cfg["promotion_auc_tolerance"],
              "recall_interval_nonnegative": interval[0] >= 0,
              "latency_budget": latency <= cfg["promotion_latency_ms_per_row"]}
    version = datetime.now(timezone.utc).strftime("candidate-%Y%m%dT%H%M%S%fZ")
    result = {"candidate_version": version, "current_version": bundle["metadata"]["version"],
              "scenario": scenario, "strength": strength, "current": current_metrics, "candidate": candidate_metrics,
              "recall_difference_ci95": interval, "bootstrap_repetitions": bootstrap,
              "training_seconds": training_seconds, "latency_ms_per_row": latency,
              "checks": {k: bool(v) for k, v in checks.items()},
              "recommendation": "eligible_for_manual_promotion" if all(checks.values()) else "do_not_promote",
              "training_recent_ids": source_ids, "evaluation_ids": evaluation_ids,
              "limitations": "Synthetic shifted holdout. Repeatedly tuning against this report invalidates its untouched status. Approval remains manual."}
    new_bundle = dict(bundle)
    new_bundle["model"] = candidate
    new_bundle["metadata"] = {**bundle["metadata"], "version": version, "parent": bundle["metadata"]["version"]}
    # Candidate baseline comes from validation, not the final holdout.
    new_bundle["baseline"] = metrics(bundle["parts"]["validation"][TARGET], candidate.predict_proba(bundle["parts"]["validation"][FEATURES])[:, 1])
    joblib.dump(new_bundle, ARTIFACTS / f"{version}.joblib")
    save_json(ARTIFACTS / f"{version}-comparison.json", result)
    save_json(ARTIFACTS / "latest-comparison.json", result)
    return result


def promote(version, approved=False):
    load_bundle(version)  # validates path/version
    if not approved:
        raise ValueError("Explicit approval required: add --approve")
    comparison = json.loads((ARTIFACTS / f"{version}-comparison.json").read_text())
    if comparison["current_version"] != load_bundle()["metadata"]["version"]:
        raise ValueError("Comparison is stale: deployed model has changed")
    if comparison["recommendation"] != "eligible_for_manual_promotion":
        raise ValueError("Candidate failed predefined promotion criteria")
    save_json(ARTIFACTS / "current.json", {"version": version, "promoted_at": datetime.now(timezone.utc).isoformat()})
    return {"promoted": version}
