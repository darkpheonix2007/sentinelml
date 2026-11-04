"""Paths may be overridden for isolated tests and deployments."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.getenv("SENTINELML_ROOT", Path(__file__).resolve().parents[2]))
ARTIFACTS = Path(os.getenv("SENTINELML_ARTIFACTS", ROOT / "artifacts"))
TARGET = "default payment next month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
FEATURES = ["LIMIT_BAL", *CATEGORICAL, "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def config():
    with (ROOT / "configs/monitoring.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
