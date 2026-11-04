import pytest


def test_delayed_labels_and_replay(frame, monitor):
    result = monitor.ingest(frame, "batch")
    assert result["label_status"] == "pending"
    assert "performance" not in result
    labels = {str(i): int(v) for i, v in enumerate(frame.PAY_0 > 0)}
    partial = monitor.label("batch", dict(list(labels.items())[:150]))
    assert partial["label_status"] == "partial"
    assert partial["degradation"] is None
    complete = monitor.label("batch", labels)
    assert complete["label_status"] == "complete"
    assert complete["label_coverage"] == 1
    assert monitor.label("batch", labels) == complete


def test_conflicting_labels_do_not_mutate(frame, monitor):
    monitor.ingest(frame, "batch")
    monitor.label("batch", {"0": 0})
    with pytest.raises(ValueError, match="Conflicting"):
        monitor.label("batch", {"0": 1, "1": 0})
    assert monitor.store.get("batch")["labels"] == {"0": 0}


def test_invalid_labels(frame, monitor):
    monitor.ingest(frame, "batch")
    for labels in [{"unknown": 0}, {"0": 2}, {"0": True}, {}]:
        with pytest.raises(ValueError):
            monitor.label("batch", labels)


def test_duplicate_batches_and_ids(frame, monitor):
    monitor.ingest(frame, "batch")
    with pytest.raises(ValueError, match="already exists"):
        monitor.ingest(frame, "batch")
    with pytest.raises(ValueError, match="unique"):
        monitor.ingest(frame, record_ids=["same"] * len(frame))


def test_confirmed_degradation(frame, monitor):
    result = monitor.ingest(frame, "batch")
    probabilities = monitor.store.get("batch")["predictions"]
    opposite = {key: int(value < .5) for key, value in probabilities.items()}
    result = monitor.label("batch", opposite)
    assert result["degradation"] is True
    assert any(a["category"] == "performance" for a in result["alerts"])
    assert monitor.label("batch", opposite) == result


def test_too_few_positives_does_not_confirm_degradation(frame, monitor):
    monitor.ingest(frame, "few-positives")
    labels = {str(i): int(i < 10) for i in range(len(frame))}
    result = monitor.label("few-positives", labels)
    assert result["degradation"] is None
    assert result["degradation_evidence"]["positive_labels"] == 10


def test_persistence_uses_consecutive_stored_batches(frame, monitor):
    shifted = frame.copy()
    shifted["AGE"] += 50
    first = monitor.ingest(shifted, "first")
    second = monitor.ingest(shifted, "second")
    assert any(a["feature"] == "AGE" and a["severity"] == "warning" for a in first["alerts"])
    assert any(a["feature"] == "AGE" and a["severity"] == "critical" for a in second["alerts"])
