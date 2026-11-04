import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt

from sentinelml.data.schema import validate
from sentinelml.modeling import load_bundle
from sentinelml.service import Monitor
from sentinelml.storage import Store

app = FastAPI(title="SentinelML", version="0.1.0", description="Local research prototype. Labels are submitted separately from monitoring inputs.")


class Batch(BaseModel):
    records: list[dict[str, float | None]] = Field(min_length=1, max_length=10000)
    batch_id: str | None = Field(default=None, min_length=1, max_length=128)
    record_ids: list[str] | None = None


class Labels(BaseModel):
    labels: dict[str, StrictInt]


def monitor():
    try:
        return Monitor()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Train a model before using this endpoint") from exc


@app.get("/health")
def health():
    try:
        bundle = load_bundle()
        Store().list()
        return {"status": "ok", "model_version": bundle["metadata"]["version"]}
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained")


@app.get("/models/current")
def current():
    return monitor().bundle["metadata"]


@app.post("/predict")
def predict(payload: Batch):
    service = monitor()
    try:
        frame = validate(pd.DataFrame(payload.records))
        p = service.bundle["model"].predict_proba(frame)[:, 1]
        return {"model_version": service.bundle["metadata"]["version"],
                "probabilities": p.tolist(), "predictions": (p >= .5).astype(int).tolist()}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/batches", status_code=201)
def ingest(payload: Batch):
    try:
        return monitor().ingest(pd.DataFrame(payload.records), payload.batch_id, payload.record_ids)
    except ValueError as exc:
        raise HTTPException(409 if "already exists" in str(exc) else 422, str(exc)) from exc


@app.get("/batches")
def batches():
    return Store().list()


@app.get("/batches/{batch_id}")
@app.get("/batches/{batch_id}/metrics")
def batch(batch_id: str):
    try:
        row = Store().get(batch_id)
        return {"id": row["id"], "created": row["created"], "version": row["version"], "report": row["report"]}
    except KeyError as exc:
        raise HTTPException(404, "Unknown batch") from exc


@app.post("/batches/{batch_id}/labels")
def label(batch_id: str, payload: Labels):
    try:
        return monitor().label(batch_id, payload.labels)
    except KeyError as exc:
        raise HTTPException(404, "Unknown batch") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/alerts")
def alerts(severity: str | None = None, model_version: str | None = None):
    return [alert for row in Store().list() for alert in row["report"]["alerts"]
            if (severity is None or alert["severity"] == severity)
            and (model_version is None or row["version"] == model_version)]


@app.get("/alerts/{alert_id}")
def alert(alert_id: str):
    for item in alerts():
        if item["alert_id"] == alert_id:
            return item
    raise HTTPException(404, "Unknown alert")
