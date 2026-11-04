import json
import sqlite3
from datetime import datetime, timezone

from sentinelml.settings import ARTIFACTS


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path=None):
        self.path = path or ARTIFACTS / "sentinelml.sqlite"
        from pathlib import Path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, created TEXT NOT NULL, version TEXT NOT NULL, report TEXT NOT NULL, predictions TEXT NOT NULL, labels TEXT NOT NULL)")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def insert(self, batch_id, version, report, predictions):
        try:
            with self.connect() as db:
                db.execute("INSERT INTO batches VALUES (?,?,?,?,?,?)", (batch_id, now(), version,
                           json.dumps(report, allow_nan=False), json.dumps(predictions, allow_nan=False), "{}"))
        except sqlite3.IntegrityError as exc:
            raise ValueError("Batch ID already exists") from exc

    def get(self, batch_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if row is None:
            raise KeyError(batch_id)
        result = dict(row)
        for name in ["report", "predictions", "labels"]:
            result[name] = json.loads(result[name])
        return result

    def list(self):
        with self.connect() as db:
            rows = db.execute("SELECT id,created,version,report FROM batches ORDER BY created DESC").fetchall()
        return [{**dict(row), "report": json.loads(row["report"])} for row in rows]

    def latest(self, version):
        with self.connect() as db:
            row = db.execute("SELECT report FROM batches WHERE version=? ORDER BY rowid DESC LIMIT 1", (version,)).fetchone()
        return json.loads(row["report"]) if row else None

    def add_labels(self, batch_id, incoming, update_report):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if row is None:
                raise KeyError(batch_id)
            predictions, labels = json.loads(row["predictions"]), json.loads(row["labels"])
            for record, value in incoming.items():
                if record not in predictions:
                    raise ValueError(f"Unknown record ID: {record}")
                if type(value) is not int or value not in [0, 1]:
                    raise ValueError("Labels must be integers 0 or 1")
                if record in labels and labels[record] != value:
                    raise ValueError(f"Conflicting label for {record}")
            if all(record in labels and labels[record] == value for record, value in incoming.items()):
                return json.loads(row["report"])
            labels.update(incoming)
            report = update_report(json.loads(row["report"]), predictions, labels)
            db.execute("UPDATE batches SET labels=?,report=? WHERE id=?", (json.dumps(labels), json.dumps(report, allow_nan=False), batch_id))
        return report
