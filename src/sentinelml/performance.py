"""Uncertainty-aware labelled-batch checks; no inference from unlabelled drift."""

import numpy as np


def degradation_evidence(labels, probabilities, baseline, minimum_positives=30, repetitions=500, seed=42):
    """Bootstrap the batch recall/F1; keep the validation baseline fixed.

    This is an operational screen, not a causal or fully paired hypothesis test.
    """
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(probabilities) >= .5
    positives = int(y.sum())
    evidence = {"labelled_rows": len(y), "positive_labels": positives,
                "minimum_positive_labels": minimum_positives, "repetitions": repetitions,
                "baseline_uncertainty_included": False}
    if positives < minimum_positives or positives == len(y):
        return {**evidence, "status": "insufficient_class_evidence", "degraded": None, "intervals": None}
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(y), size=(repetitions, len(y)))
    actual, guess = y[take].astype(bool), predicted[take]
    tp = np.count_nonzero(actual & guess, axis=1)
    fp = np.count_nonzero(~actual & guess, axis=1)
    fn = np.count_nonzero(actual & ~guess, axis=1)
    valid = tp + fn > 0
    recall_samples = tp[valid] / (tp[valid] + fn[valid])
    f1_samples = 2 * tp[valid] / (2 * tp[valid] + fp[valid] + fn[valid])
    intervals = {"recall": np.quantile(recall_samples, [.025, .975]).tolist(),
                 "f1": np.quantile(f1_samples, [.025, .975]).tolist()}
    degraded = any(intervals[name][1] < baseline[name] for name in ["recall", "f1"])
    return {**evidence, "status": "evaluated", "degraded": bool(degraded), "intervals": intervals}
