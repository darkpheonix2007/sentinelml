# SentinelML

A local **drift monitoring and risk signaling** prototype for the UCI credit-default benchmark. Train a CPU model, ingest batches, inspect distribution changes, add delayed labels, and compare a candidate against explicit promotion rules.

**Research prototype, not a lending service or a failure-prediction model.** Drift does not prove model degradation. Input-only alerts are risk signals; suitable delayed labels are needed to assess observed performance.

## Main finding: a codebook change looked like drift

In the ACSIncome-derived validation, the 2014-trained monitor alerted on 77 of 78 California 2018 batches. After removing occupation (`OCCP`) from **every** year and retraining/recalibrating, it alerted on 0 of 78. The [Census changed its occupation codes in 2018](https://www.census.gov/topics/employment/industry-occupation/about/occupation.html). This is evidence that the original alert flood was largely a schema/codebook problem—not proof that model performance collapsed.

| Validation stream | Nine-feature actionable alerts | Eight-feature, no-occupation alerts | Mean F1, nine / eight features |
|---|---:|---:|---:|
| California 2018 | 77/78 | 0/78 | 0.719 / 0.714 |
| California 2021 | 76/77 | 76/77 | 0.703 / 0.702 |
| Texas 2018 | 53/54 | 53/54 | 0.714 / 0.708 |

The eight-feature comparison was **post-hoc**, prompted by the 2018 result; it is not an untouched confirmation test. Only 14 disjoint 2014 California batches were available for the ACS stable-stream check at the sampled density, so this project cannot claim a reliable live false-alert rate. The credit-card shifts are synthetic and reuse rows. See [the ACS validation](reports/acs_validation.md) and [the corrected credit sweep](reports/shift_sweep_5000.md) for methods, measured results, and limitations. No GPU is required.

## What works

- Checksum-verified UCI download and portable dataset paths
- Reproducible, stratified, disjoint data partitions
- Logistic Regression and Random Forest pipelines, selected by validation F1
- Three-fold training cross-validation audit and sampled permutation importance
- Calibrated numerical PSI and categorical Jensen–Shannon divergence; empirical null p-values with batch-wise Benjamini–Hochberg correction
- Two-consecutive-batch feature persistence; a three-feature empirical-floor override is implemented as an opt-in experiment
- Missingness, unknown categories and range checks
- SQLite history, predictions, evidence-backed alerts, partial and complete delayed labels
- Precision, recall, F1, ROC-AUC, average precision, Brier score and confusion matrices; bootstrap intervals and minimum-positive-count checks for complete labelled batches
- Five simulated scenarios, repeated monitoring experiments and downloadable reports
- Candidate retraining, paired bootstrap comparison and explicit manual promotion
- FastAPI endpoints and Streamlit dashboard with six pages
- Automated tests, GitHub Actions workflow and Docker configuration

The optional ACS validation uses XGBoost on CPU; it does not replace the credit benchmark's selected model. SHAP, MLflow, Evidently, authentication, cloud hosting and a demo video remain optional extensions. The original roadmap is preserved in [docs/ORIGINAL_PLAN.md](docs/ORIGINAL_PLAN.md); it is not a list of implemented commands.

## Quick start

From this repository in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m sentinelml.data.download
python -m sentinelml.cli train
python -m sentinelml.cli demo
python -m streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. After initial setup, only the last command is needed. The source ZIP does **not** include the UCI dataset, trained models or local database: the download, train and demo commands create them on your computer. Running `demo` again adds five batches; it does not erase history.

If upgrading a workspace with a model trained by the earlier per-feature alert rule, run `python -m sentinelml.cli recalibrate-monitor` before starting the API/dashboard. This retains the fitted model and creates a new monitoring artifact with empirical null scores. A fresh `train` already includes them.

Without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Linux/macOS: use `python3 -m venv .venv` and `source .venv/bin/activate`, followed by the same Python commands. No GPU is required. Optional Colab training can use the same package; save artifacts with compatible Python/scikit-learn versions before transferring them.

## API

In a second terminal:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive schemas. Dashboard and API use the same service and database.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Model/database readiness; 503 before training |
| GET | `/models/current` | Current model metadata |
| POST | `/predict` | Predictions without batch persistence |
| POST | `/batches` | Validate, monitor and store inputs |
| GET | `/batches` | Stored history |
| GET | `/batches/{id}` | Metadata and report |
| GET | `/batches/{id}/metrics` | Stored report |
| POST | `/batches/{id}/labels` | Partial or complete delayed labels |
| GET | `/alerts` | Filter by severity or model version |
| GET | `/alerts/{id}` | Alert evidence |

Batch JSON contains `records` (a list of feature dictionaries), optional `batch_id`, and optional `record_ids`. All 23 features listed in `src/sentinelml/settings.py` are required; ID is optional and never used for training. Target columns are rejected. When record IDs are omitted, use string row indices (`"0"`, `"1"`, etc.) for label submission. Maximum: 10,000 records per API batch.

Label JSON example: `{"labels": {"0": 1, "1": 0}}`. Integer 0/1 labels only. Identical resubmissions are safe; conflicting labels and unknown IDs are rejected atomically. Duplicate batches return 409, invalid data 422, unknown batches 404.

The app has no authentication. Bind locally as shown. Model comparison and promotion use the CLI rather than public API endpoints.

Try the supplied synthetic two-row request in PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/batches -ContentType application/json -Body (Get-Content examples/batch.json -Raw)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/batches/example-batch/labels -ContentType application/json -Body (Get-Content examples/labels.json -Raw)
```

This validates the API wiring, but two rows are deliberately insufficient for drift/degradation decisions. Choose a new batch ID for another ingestion. Example labels are invented for the synthetic request, not benchmark outcomes.

## Dashboard workflow

1. **Overview:** selected model metrics, batch history and time trends.
2. **Ingest & simulate:** generate controlled batches or upload feature CSVs. Release simulated labels separately, or upload `record_id,label` CSVs for real batches.
3. **Batch analysis:** filter by model and batch; compare reference/current feature distributions; inspect quality and labelled performance.
4. **Alert evidence:** view severity, feature, threshold, explanation and later outcome.
5. **Model comparison:** current/candidate metrics, uncertainty, gate checks and promotion recommendation.
6. **Experiments:** measured false alerts, degradation rates, scenario comparisons and report download.

Unreleased simulator truth is held only in the current browser session. Restarting the session discards it; create a new batch to repeat the simulation. It is never passed into the monitoring calculation.

## Repeated experiments

```powershell
python -m sentinelml.cli experiment --repetitions 30
python -m sentinelml.cli benchmark-alerts --batches 500 --size 500
python -m sentinelml.cli benchmark-shift-sweep --episodes 100 --stable-batches 500 --size 500
python -m sentinelml.cli benchmark-shift-sweep --episodes 100 --stable-batches 500 --size 500 --null-draws 5000
python -m sentinelml.cli compare
python -m sentinelml.cli audit
```

The study evaluates 1,560 batches: four batch sizes × (one stable condition + four scenarios × three strengths) × 30 repetitions. It runs on CPU and can take several minutes. `--repetitions 2` is useful for smoke checks, not dependable rate estimates.

Outputs in the ignored `artifacts/` directory:

- `experiments/batches.csv`: metrics, scenario parameters, seeds and source row IDs
- `experiments/summary.json`: rates, descriptive intervals and confusion counts
- `experiments/report.md`: readable measured report
- `latest-comparison.json`: latest candidate result
- `<version>-comparison.json`: evidence for each candidate
- `<version>.joblib`: pipeline, reference data, calibration and partition data
- `current.json`: explicit current-model pointer
- `partitions.json`: source-row partition manifest
- `sentinelml.sqlite`: batch history and label reconciliation

The historical per-feature-rule run is documented in [reports/experiment_summary.md](reports/experiment_summary.md). The corrected-rule before/after benchmark is in [reports/alert_rule_comparison.md](reports/alert_rule_comparison.md), the original weak-shift/duration comparison is in [reports/shift_sweep.md](reports/shift_sweep.md), and the clean-state 5,000-draw follow-up is in [reports/shift_sweep_5000.md](reports/shift_sweep_5000.md). Do not compare these studies as if they used identical alert or degradation definitions.

## Data design

### Optional real-shift validation: ACSIncome-derived task

Install the separate CPU-only benchmark dependencies with `python -m pip install -e ".[acs]"`. Then run, from the repository root:

```powershell
python -m sentinelml.acs_validation --download --years 2015 2016 2017 2018 2021 --compare-state TX --compare-year 2018 --density 0.5 --max-batches 200 --calibration-draws 5000
```

This downloads official Census ACS person records through [Folktables](https://github.com/socialfoundations/folktables) into ignored `data/raw/acs/`, which can require substantial disk space, RAM, network time, and CPU time. Start with `--years 2015 --density 0.2 --max-batches 8 --calibration-draws 500` as a smoke check. The project does **not** require a GPU. Do not put the raw Census files in Git or the project ZIP.

The benchmark trains XGBoost on 2014 California only. Disjoint subsets of that year serve training, baseline evaluation, reference distribution, null calibration, and a stable held-out stream. Later years and an optional different state are never used for model fitting or calibration. Each stream is split into non-overlapping 500-row batches; it reports BH, Bonferroni, BY and BH-with-strong-single-batch-override under the same feature scores, q and two-batch persistence. It also reports labelled F1/recall/ROC-AUC and counts of alerts without confirmed degradation and confirmed degradation without alerts. `artifacts/acs_validation/summary.json` and per-stream CSVs contain measured outputs; a missing year must never be silently presented as tested.

This is an **ACSIncome-derived nine-feature common-schema task**, not the stock ten-feature Folktables task: `RELP` is omitted in *every* year because [the Census removed it after 2018](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/ACS2019_PUMS_README.pdf). The stock adult filter and income-over-$50,000 target are retained. Occupation and other codebooks can also change by year, producing apparent drift unrelated to a changed population. The 2020 one-year ACS data are excluded because [the Census labels them experimental and not comparable](https://www.census.gov/newsroom/press-releases/2021/experimental-2020-acs-1-year-data.html). This benchmark does not use survey weights, so its rates describe sampled records rather than population estimates. The nominal $50,000 label can also change meaning with inflation. Natural temporal and geographic shifts still do not establish causality or prove drift causes degradation. Protected/sensitive fields are present for benchmark comparison; this model must not be used for employment, benefits, lending or other decisions about people.

The [Census switched occupation coding in 2018](https://www.census.gov/topics/employment/industry-occupation/about/occupation.html). The default nine-feature run deliberately exposes this break as a monitoring failure mode. To measure sensitivity, rerun with `--exclude-occupation`; this uses an eight-feature task and retrains/recalibrates from 2014, writing separate outputs under `artifacts/acs_validation_no_occupation/`. Compare it with the nine-feature run before interpreting 2018–2021 alerts as population change.

UCI archive SHA-256:

```text
56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602
```

Training reads the XLS directly from the verified ZIP. An edited extracted XLS cannot silently change training input. The downloader caches complete verified files, uses a network timeout, and extracts only the known XLS member.

| Partition | Rows | Purpose |
|---|---:|---|
| Train | 15,000 | Fit preprocessing/models |
| Validation | 4,500 | Select by F1; baseline metrics |
| Reference | 3,000 | Expected distributions |
| Calibration | 2,250 | Stable draws for thresholds |
| Simulation | 2,250 | Experiments and eligible recent data |
| Final evaluation | 3,000 | Current/candidate comparison only |

Partitions are random and stratified, not temporal. The benchmark lacks an authentic production event timeline. IDs are excluded from features. Imputers, scalers and encoders fit on training rows only.

Models use documented fixed hyperparameters and threshold 0.5; no exhaustive tuning is claimed. The separate `audit` command evaluates three training-only folds and five-repeat permutation importance on 1,000 validation rows, without changing the selected model. Importance is predictive association, not causation, and correlated features can mask each other's contribution. Historical undocumented categories are retained and monitored rather than silently recoded. Average precision is named explicitly, rather than called trapezoidal PR-AUC. Undefined single-class ROC-AUC/average precision is JSON null.

## Monitoring methodology

Numerical bins derive from reference quantiles with infinite outer bins and a special constant-feature case. PSI uses 0.5-count smoothing. Categorical JS divergence is the square of SciPy's Jensen–Shannon distance. For each of four batch sizes, 500 separately seeded stable calibration draws provide feature-wise empirical upper-tail p-values; draws reuse calibration rows and are not statistically independent. Benjamini–Hochberg correction at configured FDR 0.10 is applied across the 23 features within each batch. Arbitrary batch sizes use the nearest calibrated size, an approximation. Empirical p-values have finite resolution, and BH false-discovery control depends on assumptions about feature-test dependence; neither alone guarantees a production false-alert rate. The separate 5,000-draw benchmark gives BY a finer p-value grid without silently changing the deployed credit artifact.

Each BH-corrected feature flag creates a warning. The same feature must flag in two consecutive batches before its warning becomes a critical/actionable alert; an intervening clean batch resets the streak. A separately labelled strong-signal override can escalate immediately when at least three BH-rejected features reach the smallest empirical p-value (exactly 1/(draws+1), never below it). This is an experimental heuristic **disabled by default in the live credit monitor** (`strong_override_enabled: false`); the 5,000-draw sweep does not establish safety for the deployed 500-draw calibration. Enabling it requires an explicitly revised model/configuration and a matched stable-stream test. Data-quality anomalies remain immediate warnings, not paging alerts. Raw uncorrected exceedances are retained for diagnosis and the before/after benchmark, not promoted to actionable alerts. The aggregate feature-drift flag requires at least 20% of features to pass BH. KS p-values are informational and are not part of the correction. Persistence is tracked per model version in the local SQLite batch order; concurrent production ingestion would need stronger sequencing guarantees.

Partial labels produce metrics and coverage but cannot confirm whole-batch degradation. Complete labels need at least 100 rows, both classes and 30 positives. A 500-resample bootstrap interval is computed for batch recall and F1; degradation is signaled only if an upper 95% bound falls below the validation point estimate. If evidence is insufficient, the outcome stays unknown rather than becoming false. This is an operational screen, not a causal proof: the validation-baseline uncertainty is not included, and class mix and sampling bias can still affect it.

Alert records preserve their original message and add an updated `outcome` after labels arrive. Old batches retain their model, baseline and configuration even after promotion.

## Scenarios and limits

| Scenario | Change | Purpose |
|---|---|---|
| stable | Unchanged held-out inputs | False-alert measurement |
| age_shift | Increase AGE, capped at 120 | Drift that may or may not affect performance |
| repayment_shift | Increase three repayment-status features | Stress a predictive part of the input |
| fractional_repayment_shift | Increase three integer repayment statuses by one for a chosen fraction of rows | Weak-shift and duration sweep without invalid decimal statuses |
| missingness | Null selected PAY_0/LIMIT_BAL entries | Quality alerts and imputation |
| label_shift | Flip some labels, leave X unchanged | Synthetic blind spot for input-only monitoring |

Feature changes retain historical labels: these are stress tests, not causal simulations of real borrowers. `label_shift` is synthetic corruption, not a natural prevalence-shift model. Degradation is measured rather than assumed. Repeated draws share rows from a finite pool; Wilson intervals are descriptive, not independent population estimates.

## Candidate evaluation and promotion

The candidate trains on original training rows plus simulated recent rows considered labelled at a synthetic cutoff. Both models are evaluated on identical transformed final-evaluation rows. The report uses 500 paired bootstrap resamples for recall gain.

Passing requires recall gain ≥0.03, no F1 regression, ROC-AUC drop ≤0.01, nonnegative lower bootstrap bound, and batch-average inference below 10 ms/row. Only after reviewing a passing result:

```powershell
python -m sentinelml.cli promote CANDIDATE_VERSION --approve
```

Failed or stale comparisons cannot be promoted. Training another baseline never silently replaces the current model. Model artifacts remain versioned. Refresh the dashboard after promotion. Repeated tuning against the final holdout invalidates its untouched status.

## Development and verification

```powershell
python -m pytest -q
```

Tests use synthetic data and require no UCI download. They cover validation, drift, multiple-testing correction, persistence, minimum-positive checks, missing/unknown values, constant features, disjoint splits, delayed labels, conflicts, API behavior, candidate gates and dashboard interactions.

Dependencies are declared in `pyproject.toml`. The original `requirements.txt` is a historical Day 1 Windows snapshot and does not contain the completed app dependencies. Use `pip install -e ".[dev]"`. Runtime versions and actual test results are recorded in `reports/verification.md`.

GitHub Actions runs tests on Python 3.11 after the workflow is pushed; local success does not imply the remote workflow has already run.

## Docker

```powershell
docker build -t sentinelml .
docker run --rm -p 127.0.0.1:8501:8501 --mount "type=bind,source=$((Get-Location).Path)\artifacts,target=/app/artifacts" sentinelml
```

Train first, then mount the resulting artifacts. The container uses an unprivileged account; Linux bind-mount permissions may need adjustment. Joblib artifacts require compatible Python/scikit-learn versions and must come from trusted sources. Docker verification status is stated in the verification report.

## Files

```text
app/                         API and dashboard entry points
src/sentinelml/data/         Acquisition and schema
src/sentinelml/modeling.py   Training, metrics, partitions and artifacts
src/sentinelml/monitoring.py Drift and calibration
src/sentinelml/simulation.py Controlled scenarios
src/sentinelml/storage.py    SQLite transactions
src/sentinelml/service.py    Shared batch/label logic
src/sentinelml/experiments.py Studies and candidate gates
src/sentinelml/cli.py        Terminal commands
configs/monitoring.yaml      Seeds, thresholds and gates
tests/                      Automated tests
notebooks/                  Original learning notebook
reports/                    Measured results
docs/ORIGINAL_PLAN.md        Original roadmap
```

Set `SENTINELML_ROOT` to override the root containing configs/data, or `SENTINELML_ARTIFACTS` to isolate models/database/results. Set variables before launching Python.

This release has no production load-testing guarantee, subgroup fairness audit, real-world deployment validation, authentication or automatic retraining scheduler. Sensitive demographic variables exist in the dataset; aggregate performance does not establish fairness. No production or unmeasured accuracy claim is made.

Dataset: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Check dataset terms before redistribution. A source-code license has not been selected.
