"""Calibrated effect sizes; statistical significance is reported, not used alone."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from sentinelml.data.schema import quality, validate
from sentinelml.settings import CATEGORICAL, FEATURES, config


def histogram(reference, current, categorical=CATEGORICAL):
    reference, current = reference.dropna(), current.dropna()
    if len(reference) == 0:
        return [0], [0], ["no reference values"]
    if reference.name in categorical:
        categories = sorted(set(reference) | set(current))
        # Frequency tables avoid O(categories x reference rows) recounting on
        # high-cardinality ACS features such as place of birth.
        ref_counts = reference.value_counts().reindex(categories, fill_value=0)
        cur_counts = current.value_counts().reindex(categories, fill_value=0)
        return (ref_counts.astype(int).tolist(), cur_counts.astype(int).tolist(),
                [str(c) for c in categories])
    interior = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(interior) == 1:
        # Isolate the constant reference value from values on either side.
        interior = np.array([interior[0], np.nextafter(interior[0], np.inf)])
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    return (np.histogram(reference, edges)[0].tolist(), np.histogram(current, edges)[0].tolist(),
            [f"{a:.3g} to {b:.3g}" for a, b in zip(edges[:-1], edges[1:])])


def feature_scores(reference, current, details=True, features=FEATURES, categorical=CATEGORICAL):
    output = {}
    for col in features:
        a, b, bins = histogram(reference[col], current[col], categorical)
        p, q = np.asarray(a, dtype=float) + 0.5, np.asarray(b, dtype=float) + 0.5
        p, q = p / p.sum(), q / q.sum()
        score = (float(jensenshannon(p, q) ** 2) if col in categorical
                 else float(np.sum((q - p) * np.log(q / p))))
        row = {"score": score, "method": "JS divergence" if col in categorical else "PSI",
               "missing_increase": float(current[col].isna().mean() - reference[col].isna().mean()),
               "unseen_rate": float((current[col].notna() & ~current[col].isin(reference[col].dropna())).mean()) if col in categorical else 0.0}
        if details:
            x, y = reference[col].dropna(), current[col].dropna()
            row["ks_pvalue"] = float(ks_2samp(x, y).pvalue) if col not in categorical and len(x) and len(y) else None
            row.update(reference_counts=a, current_counts=b, bins=bins)
        output[col] = row
    return output


def calibrate(reference, pool):
    cfg = config()
    rng = np.random.default_rng(cfg["seed"])
    thresholds, null_scores = {}, {}
    for size in cfg["calibration_batch_sizes"]:
        values = {col: [] for col in FEATURES}
        for _ in range(cfg["calibration_repetitions"]):
            batch = pool.iloc[rng.choice(len(pool), min(size, len(pool)), replace=False)]
            for col, row in feature_scores(reference, batch, details=False).items():
                values[col].append(row["score"])
        thresholds[str(size)] = {col: max(float(np.quantile(v, .99)), 1e-6) for col, v in values.items()}
        null_scores[str(size)] = {col: sorted(v) for col, v in values.items()}
    return thresholds, null_scores


def calibrate_size(reference, pool, size, repetitions, features=FEATURES, categorical=CATEGORICAL, seed=42):
    """Calibrate one size for cross-method benchmarks without altering deployed artifacts."""
    if size > len(pool) or size < 1 or repetitions < 1:
        raise ValueError("Invalid calibration size or repetition count")
    rng = np.random.default_rng(seed)
    values = {col: [] for col in features}
    for _ in range(repetitions):
        batch = pool.iloc[rng.choice(len(pool), size, replace=False)]
        for col, row in feature_scores(reference, batch, details=False, features=features, categorical=categorical).items():
            values[col].append(row["score"])
    return {col: sorted(scores) for col, scores in values.items()}


def bh_rejections(pvalues, q):
    """Benjamini-Hochberg step-up rule over one batch's feature tests."""
    ordered = sorted(pvalues, key=pvalues.get)
    cutoff = 0
    for rank, name in enumerate(ordered, start=1):
        if pvalues[name] <= q * rank / len(ordered):
            cutoff = rank
    return set(ordered[:cutoff])


def corrected_rejections(pvalues, q, method):
    """Return features rejected by BH, BY, or Bonferroni at the same q."""
    if method == "bh":
        return bh_rejections(pvalues, q)
    if method == "by":
        return bh_rejections(pvalues, q / sum(1 / rank for rank in range(1, len(pvalues) + 1)))
    if method == "bonferroni":
        return {name for name, value in pvalues.items() if value <= q / len(pvalues)}
    raise ValueError(f"Unknown correction method: {method}")


def empirical_pvalue(score, null_scores):
    """Conservative upper-tail Monte Carlo p-value; null draws exclude this batch."""
    values = np.asarray(null_scores)
    return float((1 + np.count_nonzero(values >= score)) / (len(values) + 1))


def strong_single_batch_override(pvalues, rejected, null_draws, minimum_features=3):
    """Escalate only when several corrected features reach the empirical p floor."""
    floor = 1 / (null_draws + 1)
    return sum(name in rejected and p <= floor + 1e-12 for name, p in pvalues.items()) >= minimum_features


def inspect_batch(bundle, frame, previous_streaks=None):
    frame = validate(frame)
    cfg = bundle["metadata"].get("config", config())
    reference = bundle["reference"]
    size = min(bundle["thresholds"], key=lambda n: abs(int(n) - len(frame)))
    sufficient = len(frame) >= cfg["minimum_batch_size"]
    scores = feature_scores(reference, frame)
    alerts = []
    null_scores = bundle.get("null_scores", {}).get(size)
    if null_scores is None:
        raise ValueError("Model lacks calibrated null scores. Retrain or recalibrate it before monitoring.")
    pvalues = {name: empirical_pvalue(row["score"], null_scores[name]) for name, row in scores.items()}
    rejected = corrected_rejections(pvalues, cfg["feature_fdr"], "bh") if sufficient else set()
    override = bool(cfg.get("strong_override_enabled", False) and sufficient and strong_single_batch_override(
        pvalues, rejected, len(next(iter(null_scores.values()))), cfg.get("strong_override_min_features", 3)))
    previous_streaks = previous_streaks or {}
    streaks = {name: previous_streaks.get(name, 0) + 1 for name in rejected}
    for name, row in scores.items():
        row["threshold"] = bundle["thresholds"][size][name]
        row["raw_exceedance"] = bool(sufficient and row["score"] > row["threshold"])
        row["pvalue"] = pvalues[name]
        row["drifted"] = name in rejected
        if row["drifted"]:
            persistent = streaks[name] >= cfg["persistence_batches"]
            alerts.append({"category": "feature_drift", "severity": "critical" if persistent or override else "warning",
                           "feature": name, "observed": row["score"], "threshold": row["threshold"],
                           "pvalue": row["pvalue"], "persistent": persistent, "single_batch_override": override,
                           "message": f"{name} passes batch-wise BH correction ({row['method']}). "
                                      + ("Two consecutive batches flagged; investigate. " if persistent else
                                         "Several features reached the empirical p-value floor; investigate. " if override else
                                         "First flagged batch; watch for persistence. ")
                                      + "Labels unavailable; degradation is not established."})
        if row["missing_increase"] > cfg["missing_rate_warning"]:
            alerts.append({"category": "missingness", "severity": "warning", "feature": name,
                           "observed": row["missing_increase"], "threshold": cfg["missing_rate_warning"],
                           "message": "Missing-value rate increased relative to reference."})
        if row["unseen_rate"] > 0:
            alerts.append({"category": "unseen_category", "severity": "warning", "feature": name,
                           "observed": row["unseen_rate"], "threshold": 0, "message": "Categories absent from reference; encoder will ignore unknown categories."})
    report_quality = quality(frame)
    if "_reference_quality" not in bundle:
        bundle["_reference_quality"] = quality(reference)
    reference_quality = bundle["_reference_quality"]
    for name, rate in report_quality["invalid"].items():
        # Historical undocumented codes remain visible but are not silently recoded.
        if rate > reference_quality["invalid"][name] + .02:
            alerts.append({"category": "invalid_values", "severity": "warning", "feature": name,
                           "observed": rate, "threshold": reference_quality["invalid"][name] + .02,
                           "message": "Outside documented schema expectations; investigate rather than silently drop rows."})
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    if "_reference_probability" not in bundle:
        bundle["_reference_probability"] = bundle["model"].predict_proba(reference)[:, 1]
    ref_probability = bundle["_reference_probability"]
    edges = np.linspace(0, 1, 11)
    prediction_counts = np.histogram(probabilities, edges)[0]
    reference_counts = np.histogram(ref_probability, edges)[0]
    fraction = sum(row["drifted"] for row in scores.values()) / len(scores)
    return {"rows": len(frame), "sufficient_evidence": sufficient, "calibration_batch_size": int(size),
            "flagged_features": sorted(rejected), "flag_streaks": streaks,
            "single_batch_override": override,
            "raw_feature_exceedances": sum(row["raw_exceedance"] for row in scores.values()),
            "drift_fraction": fraction,
            "drift_alert": bool(sufficient and (fraction >= cfg["feature_alert_fraction"] or
                                           any(a["category"] == "feature_drift" and a["severity"] == "critical"
                                               for a in alerts))),
            "features": scores, "quality": report_quality, "alerts": alerts, "labels_available": False,
            "prediction": {"positive_rate": float((probabilities >= .5).mean()),
                           "mean_probability": float(probabilities.mean()),
                           "near_threshold_fraction": float((np.abs(probabilities - .5) < .1).mean()),
                           "current_counts": prediction_counts.tolist(), "reference_counts": reference_counts.tolist(),
                           "js_divergence": float(jensenshannon(prediction_counts + .5, reference_counts + .5) ** 2)}}, probabilities
