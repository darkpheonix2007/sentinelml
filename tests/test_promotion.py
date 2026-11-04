import json

import pytest

from sentinelml import experiments


def test_promotion_requires_approval_passing_gate_and_fresh_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(experiments, "load_bundle", lambda version=None: {"metadata": {"version": version or "baseline"}})
    evidence = {"current_version": "baseline", "recommendation": "do_not_promote"}
    path = tmp_path / "candidate-comparison.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="approval"):
        experiments.promote("candidate")
    with pytest.raises(ValueError, match="failed"):
        experiments.promote("candidate", approved=True)
    evidence["recommendation"] = "eligible_for_manual_promotion"
    evidence["current_version"] = "old-baseline"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="stale"):
        experiments.promote("candidate", approved=True)
    evidence["current_version"] = "baseline"
    path.write_text(json.dumps(evidence))
    assert experiments.promote("candidate", approved=True) == {"promoted": "candidate"}
    assert json.loads((tmp_path / "current.json").read_text())["version"] == "candidate"
