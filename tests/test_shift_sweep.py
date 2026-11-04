import numpy as np

from sentinelml import experiments
from sentinelml.settings import TARGET


def test_sweep_outputs_paired_episode_groups(monkeypatch, tmp_path, frame, bundle):
    pool = frame.copy()
    pool["ID"] = np.arange(len(pool))
    pool[TARGET] = (pool.PAY_0 > 0).astype(int)
    bundle["parts"] = {"simulation": pool}
    monkeypatch.setattr(experiments, "load_bundle", lambda: bundle)
    monkeypatch.setattr(experiments, "ARTIFACTS", tmp_path)
    result = experiments.benchmark_shift_sweep(repetitions=2, size=100, stable_batches=2)
    assert len(result["groups"]) == 15
    assert {row["duration"] for row in result["groups"]} == {1, 2, 3}
    assert all(row["episodes"] == 2 for row in result["groups"])
    assert (tmp_path / "experiments/shift_sweep_summary.json").exists()
