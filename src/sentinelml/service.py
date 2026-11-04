import uuid

from sentinelml.modeling import load_bundle, metrics
from sentinelml.monitoring import inspect_batch
from sentinelml.performance import degradation_evidence
from sentinelml.settings import config
from sentinelml.storage import Store, now


class Monitor:
    def __init__(self, bundle=None, store=None):
        self.bundle = bundle if bundle is not None else load_bundle()
        self.store = store if store is not None else Store()

    def ingest(self, frame, batch_id=None, record_ids=None, simulation_info=None):
        batch_id = batch_id or uuid.uuid4().hex
        ids = record_ids if record_ids is not None else [str(i) for i in range(len(frame))]
        if len(ids) != len(frame) or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
            raise ValueError("Record IDs must be unique nonempty strings, one per row")
        version = self.bundle["metadata"]["version"]
        previous = self.store.latest(version)
        report, probability = inspect_batch(self.bundle, frame, previous.get("flag_streaks", {}) if previous else {})
        report.update(model_version=version, batch_id=batch_id, label_coverage=0, label_status="pending",
                      baseline=self.bundle["baseline"], config=self.bundle["metadata"].get("config", config()))
        if simulation_info is not None:
            report["simulation"] = simulation_info
        for alert in report["alerts"]:
            alert.update(alert_id=uuid.uuid4().hex, batch_id=batch_id, model_version=version,
                         created_at=now(), outcome="awaiting_labels")
        self.store.insert(batch_id, version, report, dict(zip(ids, probability.tolist())))
        return report

    def label(self, batch_id, labels):
        def update(report, predictions, merged):
            cfg = report["config"]
            report["labels_available"] = True
            report["label_coverage"] = len(merged) / len(predictions)
            report["label_status"] = "complete" if len(merged) == len(predictions) else "partial"
            sufficient = len(merged) >= cfg["minimum_labelled_rows"] and len(set(merged.values())) == 2
            report["performance"] = metrics(list(merged.values()), [predictions[i] for i in merged])
            # Partial labels can be selectively observed; do not confirm batch degradation yet.
            confirmed = sufficient and report["label_status"] == "complete"
            report["degradation"] = None
            report["degradation_evidence"] = None
            report["alerts"] = [a for a in report["alerts"] if a["category"] != "performance"]
            if confirmed:
                evidence = degradation_evidence(list(merged.values()), [predictions[i] for i in merged],
                    report["baseline"], cfg.get("minimum_positive_labels", config()["minimum_positive_labels"]),
                    cfg.get("bootstrap_repetitions", config()["bootstrap_repetitions"]))
                report["degradation_evidence"] = evidence
                report["degradation"] = evidence["degraded"]
                if evidence["degraded"] is not None:
                    for alert in report["alerts"]:
                        alert["outcome"] = "degradation_observed" if evidence["degraded"] else "no_degradation_established"
                if evidence["degraded"]:
                    report["alerts"].append({"alert_id": uuid.uuid4().hex, "category": "performance",
                        "severity": "critical", "batch_id": batch_id, "model_version": report["model_version"],
                        "created_at": now(), "evidence": evidence, "outcome": "degradation_observed",
                        "message": "The batch's bootstrap upper bound for recall or F1 is below the validation point estimate. This is observational evidence, not a causal diagnosis."})
            return report
        if not labels:
            raise ValueError("Provide at least one label")
        return self.store.add_labels(batch_id, labels, update)
