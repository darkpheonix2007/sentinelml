"""Start with: python -m streamlit run app/dashboard.py"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from sentinelml.modeling import load_bundle
from sentinelml.service import Monitor
from sentinelml.settings import ARTIFACTS, FEATURES
from sentinelml.simulation import SCENARIOS, simulate
from sentinelml.storage import Store

st.set_page_config(page_title="SentinelML • Model observability", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background: #0d1421;} [data-testid="stSidebar"] {background:#111d2e;}
[data-testid="stMetric"] {background:#172335;border:1px solid #293c52;border-radius:12px;padding:18px;}
h1,h2,h3 {letter-spacing:-.03em;} .block-container {padding-top:2rem;}
</style>""", unsafe_allow_html=True)
st.sidebar.title("◈ SentinelML")
st.sidebar.caption("MODEL OBSERVABILITY / RESEARCH EDITION")
page = st.sidebar.radio("Workspace", ["Overview", "Ingest & simulate", "Batch analysis", "Alert evidence", "Model comparison", "Experiments"])
try:
    bundle = load_bundle()
except FileNotFoundError:
    st.title("Welcome to SentinelML")
    st.info("Train the baseline before opening the monitoring workspace.")
    st.code("python -m sentinelml.cli train", language="bash")
    st.stop()

store = Store()
monitor = Monitor(bundle, store)
rows = store.list()
st.sidebar.success("Model loaded · CPU inference")
st.sidebar.caption(bundle["metadata"]["version"])
st.sidebar.caption("UCI credit-default benchmark. Synthetic monitoring experiments; not a lending service.")
st.title(page)
st.caption("Evidence first. Distribution changes are warnings; labelled outcomes determine measured performance.")


def table_rows():
    return [{"batch": row["id"], "created": row["created"], "version": row["version"],
             "scenario": row["report"].get("simulation", {}).get("scenario", "not recorded / uploaded"),
             "rows": row["report"]["rows"], "drift_fraction": row["report"]["drift_fraction"],
             "labels": row["report"]["label_status"], "coverage": row["report"]["label_coverage"],
             "alerts": len(row["report"]["alerts"]),
             "f1": row["report"].get("performance", {}).get("f1")} for row in rows]


if page == "Overview":
    left, middle, right, last = st.columns(4)
    left.metric("Monitored batches", len(rows))
    middle.metric("Stored alerts", sum(len(r["report"]["alerts"]) for r in rows))
    right.metric("Awaiting complete labels", sum(r["report"]["label_status"] != "complete" for r in rows))
    last.metric("Validation F1", f"{bundle['baseline']['f1']:.3f}")
    st.subheader("Monitoring history")
    if rows:
        history = pd.DataFrame(table_rows()).sort_values("created")
        st.plotly_chart(px.line(history, x="created", y="drift_fraction", markers=True,
                               labels={"drift_fraction": "Fraction of features flagged", "created": "Batch received"},
                               template="plotly_dark"), use_container_width=True)
        st.dataframe(history, hide_index=True, use_container_width=True)
        labelled = history.dropna(subset=["f1"])
        if len(labelled):
            st.plotly_chart(px.line(labelled, x="created", y="f1", markers=True, template="plotly_dark"), use_container_width=True)
    else:
        st.info("No batches yet. Open Ingest & simulate to run the first scenario.")
    st.subheader("Baseline model selection")
    st.dataframe(pd.DataFrame(bundle["metadata"]["results"]).T.drop(columns=["confusion_matrix"]), use_container_width=True)
    audit_path = ARTIFACTS / "model-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["model_version"] == bundle["metadata"]["version"]:
            st.subheader("Which features influence predictions?")
            importance = pd.DataFrame(audit["permutation_importance"][:10]).sort_values("mean_average_precision_drop")
            st.plotly_chart(px.bar(importance, x="mean_average_precision_drop", y="feature", orientation="h", error_x="std", template="plotly_dark"), use_container_width=True)
            st.caption(audit["interpretation"])
            with st.expander("Three-fold training cross-validation"):
                st.json(audit["cross_validation"])

elif page == "Ingest & simulate":
    st.subheader("Controlled batch simulator")
    st.write("Feature shifts preserve historical labels. These are stress tests, not claims about real customer behavior. label_shift flips labels without changing inputs.")
    with st.form("simulate"):
        c1, c2, c3 = st.columns(3)
        scenario = c1.selectbox("Scenario", SCENARIOS)
        size = c2.selectbox("Rows", [100, 250, 500, 1000])
        strength = c3.slider("Shift strength", 0.0, 3.0, 1.0, .5)
        seed = st.number_input("Seed", 0, 1000000, 42)
        submitted = st.form_submit_button("Monitor simulated batch", type="primary")
    if submitted:
        x, y, source_ids = simulate(bundle["parts"]["simulation"], scenario, size, strength, int(seed))
        report = monitor.ingest(x, simulation_info={"scenario": scenario, "strength": strength,
                                                   "seed": int(seed), "source_ids": source_ids})
        st.session_state.setdefault("pending_truth", {})[report["batch_id"]] = {str(i): int(v) for i, v in enumerate(y)}
        st.success(f"Stored batch {report['batch_id']}. Labels withheld from monitoring.")
    pending = st.session_state.get("pending_truth", {})
    if pending:
        selected = st.selectbox("Release labels for a simulated batch", list(pending))
        if st.button("Release delayed labels"):
            report = monitor.label(selected, pending[selected])
            del pending[selected]
            st.success(f"Labels linked. Measured degradation: {report['degradation']}")
    st.caption("Simulation truth is kept only in this browser session until release. Restarting the session discards unreleased truth; create a new batch if needed.")
    st.divider()
    st.subheader("Upload a feature CSV")
    st.caption("Exactly the 23 model features are required; ID is optional. Do not include the target. Limit: 10,000 rows.")
    with st.expander("Required columns"):
        st.code(",".join(FEATURES))
    upload = st.file_uploader("Feature file", type="csv")
    if upload and st.button("Monitor uploaded batch"):
        try:
            frame = pd.read_csv(upload)
            if len(frame) > 10000:
                raise ValueError("Limit uploads to 10,000 rows")
            ids = frame.ID.astype(str).tolist() if "ID" in frame else None
            result = monitor.ingest(frame, record_ids=ids)
            st.success(f"Stored batch {result['batch_id']}")
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))
    st.subheader("Upload delayed labels")
    label_file = st.file_uploader("CSV containing record_id,label", type="csv", key="labels")
    target_batch = st.text_input("Batch ID for labels")
    if label_file and st.button("Submit labels"):
        try:
            labels = pd.read_csv(label_file, dtype={"record_id": str})
            if set(labels.columns) != {"record_id", "label"} or labels.record_id.duplicated().any():
                raise ValueError("Use unique record_id values and exactly record_id,label columns")
            if labels.record_id.isna().any() or not labels.label.isin([0, 1]).all():
                raise ValueError("Record IDs are required; labels must be 0 or 1")
            monitor.label(target_batch, dict(zip(labels.record_id, labels.label.astype(int))))
            st.success("Labels stored and metrics updated")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))

elif page == "Batch analysis":
    if not rows:
        st.info("Ingest a batch first.")
        st.stop()
    versions = sorted({r["version"] for r in rows})
    version = st.selectbox("Model version", versions)
    selection = st.selectbox("Batch", [r["id"] for r in rows if r["version"] == version])
    row = store.get(selection)
    report = row["report"]
    if "simulation" in report:
        st.caption(f"Simulated scenario: {report['simulation']['scenario']} · strength {report['simulation']['strength']} · seed {report['simulation']['seed']}")
    st.write(f"Labels: **{report['label_status']}** · Coverage: **{report['label_coverage']:.0%}** · Rows: **{report['rows']}**")
    if not report["sufficient_evidence"]:
        st.warning("Batch too small for feature-drift decisions. Quality checks still apply.")
    if report["label_status"] != "complete":
        st.info("Performance degradation is not established for this batch. Partial-label metrics may be biased.")
    feature_table = pd.DataFrame(report["features"]).T
    feature_table["threshold_ratio"] = feature_table["score"] / feature_table["threshold"]
    st.dataframe(feature_table.sort_values("threshold_ratio", ascending=False)[["method", "score", "threshold", "threshold_ratio", "drifted", "missing_increase", "unseen_rate"]], use_container_width=True)
    feature = st.selectbox("Compare feature distributions", FEATURES)
    values = report["features"][feature]
    a, b = values["reference_counts"], values["current_counts"]
    chart = pd.DataFrame({"bin": values["bins"], "Reference": [n / max(sum(a), 1) for n in a],
                          "Current": [n / max(sum(b), 1) for n in b]})
    st.plotly_chart(px.bar(chart, x="bin", y=["Reference", "Current"], barmode="group", template="plotly_dark"), use_container_width=True)
    if "performance" in report:
        st.subheader("Observed labelled performance")
        score = report["performance"]
        st.dataframe(pd.DataFrame([{k: v for k, v in score.items() if k != "confusion_matrix"}]), hide_index=True)
        st.write("Confusion matrix: rows = actual [0, 1], columns = predicted [0, 1]")
        st.dataframe(pd.DataFrame(score["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    with st.expander("Quality and prediction evidence"):
        st.json({"quality": report["quality"], "prediction": report["prediction"]})
    st.download_button("Download batch evidence", json.dumps(report, indent=2), f"{selection}.json", "application/json")

elif page == "Alert evidence":
    severity = st.selectbox("Severity", ["all", "warning", "critical"])
    items = [a for row in rows for a in row["report"]["alerts"] if severity == "all" or a["severity"] == severity]
    st.caption("Historical alerts remain stored after labels arrive. Outcome records whether later evidence showed degradation.")
    if not items:
        st.info("No alerts match this filter.")
    for alert in items[:100]:
        with st.expander(f"{alert['severity'].upper()} · {alert['category']} · {alert.get('feature', 'batch')} · {alert['batch_id'][:8]}"):
            st.write(alert["message"])
            st.json(alert)
    if len(items) > 100:
        st.caption(f"Showing newest 100 of {len(items)} alerts. Use the API to retrieve the complete history.")

elif page == "Model comparison":
    path = ARTIFACTS / "latest-comparison.json"
    if not path.exists():
        st.info("Run candidate comparison from the terminal first.")
        st.code("python -m sentinelml.cli compare")
    else:
        comparison = json.loads(path.read_text())
        st.subheader(comparison["recommendation"].replace("_", " ").title())
        st.dataframe(pd.DataFrame({k: comparison[k] for k in ["current", "candidate"]}).T.drop(columns=["confusion_matrix"]), use_container_width=True)
        st.write("95% paired bootstrap interval for recall gain:", comparison["recall_difference_ci95"])
        st.dataframe(pd.DataFrame([{"criterion": k, "passed": v} for k, v in comparison["checks"].items()]), hide_index=True)
        st.write(f"Training: {comparison['training_seconds']:.2f}s · Inference: {comparison['latency_ms_per_row']:.4f} ms/row (batch average)")
        st.warning("No automatic promotion. Review the evidence; terminal approval is required. Reusing this holdout for tuning compromises evaluation.")
        st.code(f"python -m sentinelml.cli promote {comparison['candidate_version']} --approve")

elif page == "Experiments":
    path = ARTIFACTS / "experiments/summary.json"
    if not path.exists():
        st.info("Generate the repeated experiment study first.")
        st.code("python -m sentinelml.cli experiment --repetitions 30")
    else:
        summary = json.loads(path.read_text())
        st.metric("Stable-batch false-alert rate", f"{summary['stable_false_alert_rate']:.1%}")
        groups = pd.DataFrame(summary["groups"])
        st.plotly_chart(px.scatter(groups, x="monitor_alert_rate", y="degradation_rate", color="scenario", size="size", hover_data=["strength"], template="plotly_dark"), use_container_width=True)
        st.dataframe(groups, hide_index=True, use_container_width=True)
        for limitation in summary["limitations"]:
            st.caption(limitation)
        st.download_button("Download experiment report", (ARTIFACTS / "experiments/report.md").read_text(), "experiment_report.md")
