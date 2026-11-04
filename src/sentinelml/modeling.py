"""Leakage-safe baseline selection, immutable artifacts and reference calibration."""
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentinelml.settings import ARTIFACTS, ROOT, CATEGORICAL, NUMERIC, FEATURES, TARGET, config, save_json
from sentinelml.data.download import EXPECTED_SHA256, calculate_sha256


def metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability)
    prediction = probability >= 0.5
    two_classes = len(np.unique(y)) == 2
    return {"rows": len(y), "prevalence": float(y.mean()),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)) if two_classes else None,
            "average_precision": float(average_precision_score(y, probability)) if two_classes else None,
            "brier": float(brier_score_loss(y, probability)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def pipeline(kind, seed=42):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    model = (LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
             if kind == "logistic" else RandomForestClassifier(
                 n_estimators=150, max_depth=10, min_samples_leaf=10,
                 class_weight="balanced", n_jobs=2, random_state=seed))
    return Pipeline([("preprocess", preprocess), ("model", model)])


def partition(frame, seed=42):
    train, rest = train_test_split(frame, train_size=0.5, stratify=frame[TARGET], random_state=seed)
    validation, rest = train_test_split(rest, train_size=0.3, stratify=rest[TARGET], random_state=seed)
    reference, rest = train_test_split(rest, train_size=2/7, stratify=rest[TARGET], random_state=seed)
    evaluation, pool = train_test_split(rest, train_size=0.4, stratify=rest[TARGET], random_state=seed)
    calibration, simulation = train_test_split(pool, train_size=0.5, stratify=pool[TARGET], random_state=seed)
    return dict(train=train, validation=validation, reference=reference,
                evaluation=evaluation, calibration=calibration, simulation=simulation)


def train():
    from sentinelml.monitoring import calibrate
    archive = ROOT / "data/raw/credit_default.zip"
    if not archive.exists() or calculate_sha256(archive) != EXPECTED_SHA256:
        raise ValueError("Run the checksum-verified downloader first")
    # Extract from the verified archive rather than trust a separately edited XLS.
    import io
    import zipfile
    with zipfile.ZipFile(archive) as source:
        frame = pd.read_excel(io.BytesIO(source.read("default of credit card clients.xls")), header=1)
    if set(frame.columns) != set(["ID", TARGET, *FEATURES]) or not frame.ID.is_unique:
        raise ValueError("Dataset schema or IDs do not match the benchmark")
    parts = partition(frame, config()["seed"])
    results, models = {}, {}
    for kind in ["logistic", "forest"]:
        start = time.perf_counter()
        model = pipeline(kind, config()["seed"])
        model.fit(parts["train"][FEATURES], parts["train"][TARGET])
        result = metrics(parts["validation"][TARGET], model.predict_proba(parts["validation"][FEATURES])[:, 1])
        result["training_seconds"] = time.perf_counter() - start
        results[kind], models[kind] = result, model
    selected = max(results, key=lambda k: results[k]["f1"])
    version = datetime.now(timezone.utc).strftime("baseline-%Y%m%dT%H%M%S%fZ")
    reference = parts["reference"][FEATURES]
    thresholds, null_scores = calibrate(reference, parts["calibration"][FEATURES])
    metadata = {"version": version, "selected": selected, "selection_metric": "validation_f1",
                "decision_threshold": 0.5, "results": results, "dataset_sha256": EXPECTED_SHA256,
                "seed": config()["seed"], "python": platform.python_version(), "sklearn": sklearn.__version__,
                "partition_sizes": {k: len(v) for k, v in parts.items()}, "config": config()}
    bundle = {"model": models[selected], "reference": reference, "thresholds": thresholds, "null_scores": null_scores,
              "baseline": results[selected], "metadata": metadata, "parts": parts}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACTS / f"{version}.joblib")
    save_json(ARTIFACTS / f"{version}.json", metadata)
    save_json(ARTIFACTS / "partitions.json", {k: v.ID.astype(int).tolist() for k, v in parts.items()})
    # Training never silently replaces an already selected current model.
    if not (ARTIFACTS / "current.json").exists():
        save_json(ARTIFACTS / "current.json", {"version": version})
    return metadata


def load_bundle(version=None):
    if version is None:
        pointer = ARTIFACTS / "current.json"
        if not pointer.exists():
            raise FileNotFoundError("No trained model. Run: python -m sentinelml.cli train")
        version = json.loads(pointer.read_text())["version"]
    if not version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in version):
        raise ValueError("Invalid model version")
    return joblib.load(ARTIFACTS / f"{version}.joblib")


def recalibrate_monitor():
    """Create a new artifact with the same model and a corrected alert calibration."""
    from sentinelml.monitoring import calibrate
    bundle = load_bundle()
    original = bundle["metadata"]["version"]
    thresholds, null_scores = calibrate(bundle["reference"], bundle["parts"]["calibration"][FEATURES])
    version = datetime.now(timezone.utc).strftime("monitor-%Y%m%dT%H%M%S%fZ")
    bundle["thresholds"], bundle["null_scores"] = thresholds, null_scores
    bundle["metadata"] = {**bundle["metadata"], "version": version,
                          "monitor_recalibrated_from": original, "config": config()}
    joblib.dump(bundle, ARTIFACTS / f"{version}.joblib")
    save_json(ARTIFACTS / f"{version}.json", bundle["metadata"])
    save_json(ARTIFACTS / "current.json", {"version": version})
    return {"version": version, "model_unchanged_from": original,
            "calibration_repetitions": config()["calibration_repetitions"]}
