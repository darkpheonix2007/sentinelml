from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sentinelml import modeling, storage


def test_dashboard_navigation_and_label_workflow(monkeypatch, bundle, frame, monitor):
    bundle["metadata"].update(results={"logistic": bundle["baseline"]}, selected="logistic")
    pool = frame.copy()
    pool["ID"] = range(len(pool))
    pool["default payment next month"] = (pool.PAY_0 > 0).astype(int)
    bundle["parts"] = {"simulation": pool}
    monkeypatch.setattr(modeling, "load_bundle", lambda: bundle)
    monkeypatch.setattr(storage, "Store", lambda: monitor.store)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/dashboard.py"), default_timeout=30)
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Ingest & simulate").run()
    app.selectbox[1].set_value(100)
    app.button[0].click().run()
    assert not app.exception
    assert len(monitor.store.list()) == 1
    release = next(button for button in app.button if button.label == "Release delayed labels")
    release.click().run()
    assert not app.exception
    assert monitor.store.list()[0]["report"]["label_status"] == "complete"
    for page in ["Overview", "Batch analysis", "Alert evidence", "Model comparison", "Experiments"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Dashboard error on {page}"
