"""Training-fold cross-validation and sampled validation permutation importance."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate

from sentinelml.modeling import load_bundle, pipeline
from sentinelml.settings import ARTIFACTS, FEATURES, TARGET, save_json


def audit():
    bundle = load_bundle()
    train = bundle["parts"]["train"]
    cv_results = {}
    for kind in ["logistic", "forest"]:
        scores = cross_validate(pipeline(kind), train[FEATURES], train[TARGET],
                                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                scoring={"f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"},
                                n_jobs=1)
        cv_results[kind] = {metric: {"folds": scores[f"test_{metric}"].tolist(),
                                     "mean": float(np.mean(scores[f"test_{metric}"])),
                                     "std": float(np.std(scores[f"test_{metric}"]))}
                            for metric in ["f1", "roc_auc", "average_precision"]}
    validation = bundle["parts"]["validation"].sample(n=min(1000, len(bundle["parts"]["validation"])), random_state=42)
    importance = permutation_importance(bundle["model"], validation[FEATURES], validation[TARGET],
                                       scoring="average_precision", n_repeats=5, random_state=42, n_jobs=1)
    features = sorted([{"feature": col, "mean_average_precision_drop": float(importance.importances_mean[i]),
                        "std": float(importance.importances_std[i])} for i, col in enumerate(FEATURES)],
                      key=lambda item: item["mean_average_precision_drop"], reverse=True)
    result = {"model_version": bundle["metadata"]["version"], "cross_validation": cv_results,
              "permutation_importance": features, "validation_sample_rows": len(validation),
              "interpretation": "Importance is the drop in validation average precision after shuffling one feature. Correlated features can mask each other's importance. This is predictive association, not causation.",
              "evaluation_usage": "Only training folds and validation rows used. Final evaluation untouched by this audit. No model or decision thresholds changed."}
    save_json(ARTIFACTS / "model-audit.json", result)
    return result
