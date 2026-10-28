# SentinelML

**An intelligent ML failure-prediction and self-monitoring platform for tabular classification models**

SentinelML is a reproducible machine-learning monitoring project that detects data-quality problems, identifies changes in incoming feature distributions, tracks model performance when delayed labels become available, and records evidence for every alert.

The central research question is:

> When does data drift correspond to meaningful model degradation, and when is it only a harmless change in the input distribution?

This project uses the UCI Default of Credit Card Clients dataset as a research benchmark. It is not a real credit-decision system and must not be used to approve, reject, or rank people for financial products.

## Project status

SentinelML is intended to be implemented incrementally over six to eight weeks. The first milestone is a reproducible monitoring prototype. A persistent API, dashboard, candidate-model comparison, Docker image, and continuous integration are later milestones.

## Why this project exists

A model can continue returning predictions after its operating environment has changed. The code may be healthy while its assumptions are no longer reliable. Monitoring only uptime therefore misses several important failure modes:

- Incoming features can contain missing, invalid, or unseen values.
- Feature distributions can move away from the training reference.
- Prediction confidence and positive-prediction rates can change.
- Actual precision, recall, and F1 can degrade after labels arrive.
- Drift can occur without harming predictive performance.
- Retraining after every drift alert can waste resources or make the model worse.

SentinelML separates these signals instead of collapsing them into one red warning.

## Goals

The system should:

1. Train a reproducible baseline classification pipeline.
2. Preserve a reference profile describing expected input distributions.
3. Accept production-like batches and validate their schema and quality.
4. Detect numerical, categorical, missingness, and prediction drift.
5. Work when ground-truth labels are initially unavailable.
6. Recalculate batch performance when delayed labels arrive.
7. Link earlier drift alerts to later performance evidence.
8. Compare a deployed model with a retrained candidate on identical untouched data.
9. Recommend promotion only when predefined criteria pass.
10. Store enough evidence to reproduce every alert and decision.

## Non-goals

The first version will not attempt to provide:

- A production lending or credit-scoring service
- Guaranteed concept-drift detection without labels
- Fully automatic model replacement
- A distributed streaming architecture
- Kubernetes deployment
- Real-time feature stores
- Large-scale deep-learning training
- A claim that drift proves model failure

## Main technical principle

SentinelML treats the following as separate states:

| State | Input drift | Performance degradation | Interpretation |
|---|---:|---:|---|
| Stable | No | No | Normal operation |
| Harmless drift | Yes | No | Investigate, but do not assume failure |
| Hidden degradation | Weak or undetected | Yes | Monitoring blind spot or possible concept change |
| Confirmed risk | Yes | Yes | Strong evidence for investigation and candidate evaluation |

When labels are unavailable, SentinelML can identify risk signals but cannot generally establish true predictive performance.

## Dataset

The initial benchmark is the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

It contains approximately 30,000 historical records with demographic, credit-limit, repayment-status, billing, and payment features. The target represents default in the following month.

### Dataset limitations

- The data is static and does not represent a live production stream.
- It does not provide a suitable event timeline for authentic temporal monitoring.
- Several categorical features are integer-encoded.
- Synthetic shifts may not reflect realistic socioeconomic or behavioral changes.
- Results demonstrate the monitoring methodology, not safety for real lending.

All incoming batches in the project are therefore described as **simulated production-like batches**.

## Technology stack

### Core requirements

- Python 3.11+
- pandas and NumPy for data processing
- scikit-learn for preprocessing, baseline models, and evaluation
- SciPy for statistical tests and distribution distances
- XGBoost as an optional gradient-boosting model
- SHAP as an optional explanation layer
- FastAPI for inference and monitoring endpoints
- Streamlit and Plotly for the interactive dashboard
- SQLite for local persistence
- PyYAML for configuration
- pytest for automated tests
- joblib for model artifacts

### Optional tools

- Evidently for comparison reports, not as the only monitoring implementation
- MLflow after the basic experiment registry is reliable
- Docker for reproducible application execution
- GitHub Actions for automated tests

The project does not require a local GPU. The dataset and monitoring calculations are small enough for CPU execution. Free Google Colab can be used for larger hyperparameter searches, repeated candidate training, or SHAP calculations.

## Proposed repository structure

```text
sentinelml/
├── app/
│   ├── api.py
│   └── dashboard.py
├── artifacts/
│   ├── models/
│   └── reference_profiles/
├── configs/
│   ├── monitoring.yaml
│   └── training.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── experiments/
│   ├── run_degradation_study.py
│   ├── run_drift_study.py
│   └── run_retraining_study.py
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── experiment_summary.md
├── src/
│   └── sentinelml/
│       ├── data/
│       │   ├── download.py
│       │   ├── preparation.py
│       │   └── schema.py
│       ├── modeling/
│       │   ├── evaluate.py
│       │   ├── registry.py
│       │   └── train.py
│       ├── monitoring/
│       │   ├── alerts.py
│       │   ├── data_quality.py
│       │   ├── drift.py
│       │   ├── performance.py
│       │   └── thresholds.py
│       ├── services/
│       │   ├── monitoring_service.py
│       │   └── prediction_service.py
│       ├── simulation/
│       │   ├── batch_generator.py
│       │   └── scenarios.py
│       └── storage/
│           ├── database.py
│           └── repositories.py
├── tests/
│   ├── fixtures/
│   ├── integration/
│   └── unit/
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

## System architecture

```text
                         ┌──────────────────────┐
                         │ Reference dataset    │
                         └──────────┬───────────┘
                                    │
                         train and profile
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
        ┌───────▼────────┐                     ┌────────▼────────┐
        │ Model artifact │                     │ Reference profile│
        └───────┬────────┘                     └────────┬────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                         ┌──────────▼───────────┐
 Incoming batch ────────►│ Monitoring service  │
                         ├──────────────────────┤
                         │ Schema validation    │
                         │ Data-quality checks  │
                         │ Drift calculations   │
                         │ Model predictions    │
                         │ Alert evaluation     │
                         └──────────┬───────────┘
                                    │
                              store evidence
                                    │
                         ┌──────────▼───────────┐
                         │ SQLite               │
                         │ batches, metrics,    │
                         │ labels and alerts    │
                         └──────┬────────┬──────┘
                                │        │
                       ┌────────▼──┐  ┌──▼──────────┐
                       │ FastAPI   │  │ Streamlit   │
                       └───────────┘  └─────────────┘
```

The API and dashboard must call the same service layer. Monitoring logic must not be independently duplicated in the user interface.

## End-to-end workflow

1. Download and validate the benchmark dataset.
2. Create fixed, stratified, non-overlapping data partitions.
3. Fit preprocessing only on training data.
4. Train baseline models and select one using validation results.
5. Save preprocessing and model logic as one inference pipeline.
6. Build a reference profile from the monitoring-reference partition.
7. Generate stable and shifted production-like batches.
8. Validate each incoming batch before prediction.
9. Calculate data-quality, feature-drift, and prediction-drift metrics.
10. Save predictions and monitoring evidence without using hidden labels.
11. Release labels after a simulated delay.
12. Calculate performance and associate it with earlier warnings.
13. Evaluate alert true positives, false positives, and missed degradation.
14. Train a candidate using only data eligible at the retraining cutoff.
15. Compare the current and candidate models on identical untouched data.
16. Produce an explicit promote, reject, or insufficient-evidence recommendation.

## Data partitioning

The default experimental split is:

| Partition | Suggested size | Purpose |
|---|---:|---|
| Training | 50% | Fit preprocessing and models |
| Validation | 15% | Model and hyperparameter selection |
| Monitoring reference | 10% | Expected distributions and reference statistics |
| Simulation pool | 15% | Generate stable and shifted incoming batches |
| Final evaluation | 10% | Untouched current-versus-candidate comparison |

The split must be created once using a recorded random seed. Row identifiers for every partition should be saved so that experiments are reproducible and overlap can be tested.

Because the source dataset is not genuinely temporal, this design should be described as leakage-safe experimental partitioning rather than true time-aware validation.

## Data preparation

### Schema

The schema module should define:

- Required columns
- Target column
- Unique identifier column
- Numerical features
- Categorical features
- Allowed categories
- Valid numerical ranges
- Nullable and non-nullable fields
- Expected data types

### Cleaning rules

Cleaning decisions must be explicit and tested. Do not silently discard invalid rows. For each transformation, record the number of affected values.

The preprocessing pipeline should:

- Impute missing numerical values using training-only statistics.
- Impute or represent missing categorical values explicitly.
- One-hot encode categorical features with safe unknown-category handling.
- Scale numerical features for Logistic Regression where appropriate.
- Leave compatible tree-model inputs unscaled where possible.
- Preserve feature names for monitoring and explanation.

### Leakage prevention

- Never fit imputers, encoders, scalers, or feature selectors on validation or evaluation data.
- Do not use the final evaluation partition during threshold tuning.
- Do not use delayed labels before their simulated arrival time.
- Do not select retraining examples using knowledge from future evaluation rows.

## Model training

### Baselines

Train at least:

1. Logistic Regression with class weights
2. Random Forest or HistGradientBoosting
3. XGBoost only if it provides a meaningful improvement

Start with simple models. Additional complexity is justified only when validation results improve or the model offers a useful experimental comparison.

### Metrics

Record:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix
- Brier score or another calibration measure
- Training time
- Batch inference time

Accuracy may be reported but should not be the primary metric for an imbalanced target.

### Model selection

Define the primary metric before tuning. For example, use recall or F1 if missing positive cases is costly, while constraining precision above an acceptable minimum.

Use randomized search instead of an enormous grid. A practical initial budget is 20–40 configurations with three-fold cross-validation. Enable early stopping for boosting models.

### Saved artifact

Save a single pipeline containing both preprocessing and prediction logic when possible. Save accompanying metadata:

```json
{
  "model_version": "1.0.0",
  "algorithm": "logistic_regression",
  "created_at": "ISO-8601 timestamp",
  "dataset_version": "uci-credit-default-v1",
  "training_seed": 42,
  "decision_threshold": 0.5,
  "feature_schema_version": "1.0.0",
  "metrics": {
    "precision": null,
    "recall": null,
    "f1": null,
    "roc_auc": null,
    "pr_auc": null
  }
}
```

Do not insert claimed metric values until they have been measured.

## Reference profile

The reference profile describes normal input and prediction behavior. It should contain:

- Number of reference rows
- Feature types
- Numerical summary statistics
- Numerical histograms with fixed bin edges
- Categorical frequency tables
- Missing-value rates
- Allowed categories and ranges
- Baseline prediction distribution
- Reference model version
- Dataset and schema versions
- Profile creation timestamp

Persist the histogram boundaries and category sets. Recomputing incompatible bins for every batch makes historical scores difficult to compare.

## Data-quality monitoring

For every incoming batch calculate:

- Row count
- Duplicate-row rate
- Duplicate-identifier rate
- Missing rate by feature
- Invalid type count
- Out-of-range count
- Invalid-category count
- Unseen-category rate
- Constant and near-constant features
- Prediction failures

Small batches should be marked as insufficient evidence when statistical estimates are unreliable.

## Drift detection

### Numerical features

Implement complementary measurements:

- Kolmogorov-Smirnov test for distribution differences
- Population Stability Index for magnitude of change
- Optional Wasserstein distance for an interpretable distance measure
- Missing-rate difference as a separate signal

P-values must not be used alone. Large batches can make negligible differences statistically significant, while small batches can hide meaningful changes.

### Categorical features

Implement:

- Jensen-Shannon divergence between reference and current frequencies
- Absolute category-frequency changes
- Unseen-category rate
- Optional chi-square test when expected counts are adequate

### Aggregate drift

Do not average every metric without thought. A practical aggregate report can include:

- Number and percentage of features over their calibrated thresholds
- Maximum feature drift score
- Weighted drift score using feature importance
- Top changed features
- Batch-level severity

Feature-importance weighting should be reported separately from unweighted drift so that weak-feature changes remain visible.

## Prediction monitoring without labels

When labels are delayed, calculate:

- Predicted-positive rate
- Mean predicted probability
- Probability histogram
- Fraction of predictions near the decision threshold
- Mean confidence
- Low-confidence prediction rate
- Change in prediction distribution from the reference

These are proxy signals. They can justify investigation but cannot prove that accuracy, recall, or F1 has degraded.

## Performance monitoring with delayed labels

When labels arrive:

1. Match labels to stored predictions using stable record identifiers.
2. Reject or safely handle duplicate label submissions.
3. Verify that all labels belong to the expected batch.
4. Store label coverage for partial batches.
5. Calculate metrics only when the minimum labelled count is met.
6. Compare batch metrics with the baseline and configured tolerances.
7. Update prior alerts with confirmed, unconfirmed, or insufficient-evidence outcomes.

Record precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, prevalence, and calibration where sample size allows.

## Alert model

An alert must be an evidence record, not only a color in the dashboard.

Each alert should store:

- Alert ID
- Batch ID
- Timestamp
- Severity
- Alert category
- Triggered rule
- Metric name
- Observed value
- Threshold
- Reference value
- Affected features
- Model version
- Schema version
- Label availability
- Human-readable explanation
- Recommended next action
- Later performance outcome, when available

Example explanation:

> Warning: `PAY_0` exceeded its calibrated PSI threshold. The observed PSI was 0.28 compared with a warning threshold of 0.20. Labels are not yet available, so performance degradation has not been established.

### Example configuration

```yaml
monitoring:
  minimum_batch_size: 200
  minimum_labelled_rows: 100

data_quality:
  missing_rate_increase_warning: 0.02
  missing_rate_increase_critical: 0.05
  unseen_category_rate_warning: 0.01

drift:
  psi_warning: 0.20
  psi_critical: 0.30
  feature_warning_fraction: 0.20
  feature_critical_fraction: 0.40

performance:
  recall_drop_warning: 0.05
  recall_drop_critical: 0.10
  f1_drop_warning: 0.05
  roc_auc_drop_warning: 0.03
```

These values are placeholders. Final thresholds must be calibrated through stable-batch experiments rather than copied from generic rules of thumb.

## Simulation scenarios

Every generated batch should contain scenario metadata, transformation parameters, source row identifiers, and a random seed.

### Stable

Sample from the simulation pool without applying an intentional distribution shift. Use repeated stable batches to estimate normal metric variation and false-positive rates.

### Weak-feature drift

Shift a feature that has low predictive importance. This scenario is designed to demonstrate that detectable drift may not materially change model quality.

### Important-feature drift

Shift a strongly predictive feature, particularly in regions near or across the decision boundary. Degradation must be measured rather than assumed.

### Missingness shift

Increase missing values for selected features at controlled rates. Evaluate both the quality alert and the effect of imputation on predictions.

### Category shift

Change category proportions or introduce a valid but previously unseen category. Verify that preprocessing does not crash and that monitoring records the change.

### Prevalence shift

Change the target-class proportion in an offline experiment. Clearly distinguish this from ordinary covariate drift.

### Concept-like synthetic shift

Modify the relationship between selected inputs and the target in a clearly labelled synthetic experiment. Do not claim that arbitrary feature perturbation is concept drift.

## Experiment A: Can drift be detected?

Run repeated trials across:

- Batch sizes: 100, 250, 500, and 1,000
- Shift strengths: small, medium, and large
- Multiple features and scenario types
- At least 30 random repetitions per combination when runtime permits

Report:

- True-positive rate
- False-positive rate
- Alert precision and recall
- Detection results by batch size
- Detection results by feature
- Threshold sensitivity
- Confidence intervals across repeated runs

Thresholds should first be estimated from stable-batch score distributions. The final experiment should then evaluate those thresholds on separate simulation runs.

## Experiment B: Does drift predict degradation?

For every batch store:

- Scenario and strength
- Batch size and seed
- Per-feature drift metrics
- Aggregate drift values
- Prediction-distribution changes
- Baseline metrics
- Batch metrics after labels arrive
- Absolute and relative performance changes
- Whether predefined degradation criteria were met

Analyze:

- Correlation between drift magnitude and performance change
- Confusion matrix for drift alerts versus observed degradation
- Drift without degradation
- Degradation without a strong drift alert
- Features most frequently associated with degradation
- Effect of batch size on both drift and performance estimates

The objective is not to prove that one drift score always predicts model failure. The objective is to identify when it provides useful evidence and where it fails.

## Experiment C: Can retraining help?

1. Freeze the currently deployed model.
2. Choose a retraining cutoff.
3. Include only labels that would be available by that cutoff.
4. Train the candidate on eligible original and recent data.
5. Evaluate current and candidate models on identical untouched rows.
6. Calculate bootstrap confidence intervals for metric differences.
7. Measure training time and inference latency.
8. Check every predefined promotion criterion.
9. Store the final recommendation and evidence.

Example promotion policy:

- Candidate recall improves by at least 0.03.
- Candidate F1 does not decrease.
- Candidate ROC-AUC does not decrease by more than 0.01.
- Bootstrap analysis does not indicate a material recall regression.
- Inference latency remains within the configured budget.
- Required data-quality and schema checks pass.
- A human explicitly approves promotion.

Promotion may correctly result in **do not promote** or **insufficient evidence**.

## Storage design

SQLite is sufficient for the first implementation.

### Suggested tables

#### `models`

- `model_version`
- `algorithm`
- `artifact_path`
- `metadata_path`
- `training_timestamp`
- `status`
- `schema_version`

#### `batches`

- `batch_id`
- `received_at`
- `row_count`
- `model_version`
- `scenario`
- `scenario_strength`
- `seed`
- `label_status`

#### `predictions`

- `record_id`
- `batch_id`
- `model_version`
- `predicted_class`
- `predicted_probability`
- `created_at`

#### `labels`

- `record_id`
- `batch_id`
- `label`
- `received_at`

#### `metrics`

- `metric_id`
- `batch_id`
- `metric_group`
- `feature_name`
- `metric_name`
- `metric_value`
- `reference_value`
- `calculated_at`

#### `alerts`

- `alert_id`
- `batch_id`
- `severity`
- `category`
- `rule_name`
- `observed_value`
- `threshold`
- `evidence_json`
- `label_availability`
- `outcome`
- `created_at`

Use database migrations or an explicit schema-initialization script. Do not create inconsistent tables from dashboard code.

## API design

Suggested endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Application, database, and model health |
| `GET` | `/models/current` | Current model metadata |
| `POST` | `/predict` | Validate and predict individual or small inputs |
| `POST` | `/batches` | Ingest and monitor a batch |
| `GET` | `/batches/{batch_id}` | Batch metadata and status |
| `GET` | `/batches/{batch_id}/metrics` | Quality, drift, prediction, and performance metrics |
| `POST` | `/batches/{batch_id}/labels` | Add delayed labels and update performance |
| `GET` | `/alerts` | Filter and list alerts |
| `GET` | `/alerts/{alert_id}` | Retrieve full alert evidence |
| `POST` | `/models/compare` | Compare current and candidate artifacts |

API requests should use validated schemas and return useful error messages. Batch ingestion should be idempotent or clearly reject duplicate batch identifiers.

## Dashboard design

### 1. Overview

- Current model version
- System health
- Number of monitored batches
- Latest labelled performance
- Latest unlabelled monitoring status
- Label coverage and delay
- Open warnings by severity
- Recent performance and drift trends

### 2. Batch analysis

- Batch metadata
- Data-quality checks
- Top changed features
- Reference versus current distributions
- Prediction distribution
- Label availability
- Performance after labels arrive

### 3. Alert evidence

- Triggered rule
- Observed value and threshold
- Stable-batch comparison
- Affected features
- Model and schema versions
- Label status
- Later confirmed or unconfirmed degradation
- Recommended action

### 4. Model comparison

- Current and candidate metrics
- Absolute metric differences
- Confidence intervals
- Training and inference time
- Pass or fail for each promotion rule
- Final recommendation

The dashboard should read stored results. It should not rerun expensive experiments every time the page refreshes.

## Local installation

The following commands describe the intended setup after `pyproject.toml` and source files have been implemented.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Configuration

Application configuration should come from version-controlled YAML files and environment variables for machine-specific paths.

Example environment variables:

```text
SENTINELML_DATABASE_URL=sqlite:///artifacts/sentinelml.db
SENTINELML_MODEL_PATH=artifacts/models/current.joblib
SENTINELML_PROFILE_PATH=artifacts/reference_profiles/current.json
SENTINELML_CONFIG_PATH=configs/monitoring.yaml
```

Do not store passwords, private keys, or access tokens in configuration files committed to Git.

## Intended commands

Once implemented, the project should expose commands similar to:

```bash
python -m sentinelml.data.download
python -m sentinelml.data.preparation
python -m sentinelml.modeling.train --config configs/training.yaml
python -m sentinelml.simulation.batch_generator --scenario stable
python experiments/run_drift_study.py
python experiments/run_degradation_study.py
python experiments/run_retraining_study.py
pytest
uvicorn app.api:app --reload
streamlit run app/dashboard.py
```

These commands document the desired interface; they become functional as the corresponding modules are implemented.

## CPU computer and Google Colab workflow

SentinelML does not require a local GPU. Use the local CPU computer for application development and Colab only for occasional expensive training.

### Run locally

- Data validation and preprocessing development
- Logistic Regression and initial tree-model baselines
- Drift and data-quality calculations
- Batch simulation development
- FastAPI and Streamlit
- SQLite
- Unit and integration tests
- Docker
- Model inference

### Run in free Google Colab when useful

- Larger XGBoost tuning jobs
- Repeated candidate-model training
- Large SHAP calculations
- Expensive experiment combinations

### Artifact transfer

After Colab training, save or download:

- `model.joblib`
- `model_metadata.json`
- `reference_profile.json`
- `best_parameters.json`
- `training_metrics.json`
- Experiment result CSV files

Record package versions, seeds, and configuration with every exported artifact. A model trained with GPU acceleration should still be tested locally for CPU inference before integration.

Avoid placing the entire application inside Colab notebooks. Reusable logic should live in Python modules that both notebooks and local applications import.

## Testing strategy

### Unit tests

- Required-column validation
- Invalid type and range detection
- Unknown-category handling
- Numerical drift calculations
- Categorical divergence calculations
- Missing-rate comparisons
- Alert threshold evaluation
- Performance calculations
- Promotion-rule decisions

### Integration tests

- Train a small model and load its artifact.
- Submit a valid batch and store predictions.
- Submit a shifted batch and verify evidence creation.
- Add delayed labels and update performance.
- Retrieve batch metrics through the API.
- Compare current and candidate models on identical rows.

### Scientific-validity tests

- Verify that data partitions do not overlap.
- Verify that preprocessing is fitted only on training data.
- Verify deterministic results with fixed seeds.
- Verify that stable batches remain within the calibrated false-alert budget.
- Verify that strong synthetic shifts affect the intended features.
- Verify that tiny batches are marked insufficient rather than trusted.
- Verify that current and candidate models use identical evaluation rows.

### Edge cases

- Empty batches
- One-row batches
- All-missing features
- Constant features
- Infinite values
- Duplicate identifiers
- Partial label arrival
- Duplicate label submission
- A batch containing only one target class
- Unseen categories
- Missing model artifacts
- Database unavailability

## Continuous integration

GitHub Actions should run:

1. Dependency installation
2. Static checks and formatting verification
3. Unit tests
4. API integration tests
5. A small smoke-training test on reduced data

Do not run the full hyperparameter search or all simulation repetitions in every pull request.

## Docker

The first container should run the API or dashboard with mounted artifact and database directories. Avoid a complex multi-container design until the single-container application is reliable.

The image should:

- Use a pinned Python base image
- Install dependencies from the project definition
- Run as a non-root user where practical
- Include a health check
- Exclude raw data and local virtual environments
- Accept configuration through environment variables

## Implementation roadmap

### Week 1: Reproducible baseline

- Create the package structure and environment.
- Add dataset download and checksum verification.
- Define the feature schema.
- Perform EDA and document invalid values.
- Create and persist fixed data partitions.
- Train Logistic Regression and a tree baseline.
- Save metrics and the selected pipeline.
- Add initial schema and leakage tests.

**Exit criteria:** A fresh checkout can recreate the data partitions, train the baseline, and reproduce the recorded validation metrics.

### Week 2: Model selection and reference profile

- Add controlled randomized search.
- Evaluate class weights and decision thresholds.
- Add XGBoost only if justified.
- Measure calibration and inference latency.
- Create the reference profile.
- Add sampled SHAP or permutation importance.

**Exit criteria:** The selected model is supported by measured results, and the untouched final evaluation partition has not been used.

### Week 3: Monitoring engine

- Implement quality checks.
- Implement numerical and categorical drift metrics.
- Implement prediction-distribution monitoring.
- Add configuration-driven thresholds.
- Create structured alert evidence.
- Add unit tests for mathematical and edge cases.

**Exit criteria:** Stable data produces expected low-risk results, a strong controlled shift produces feature-specific evidence, and tiny batches are handled safely.

### Week 4: Simulation and calibration

- Implement scenario generators.
- Run repeated stable batches.
- Calibrate preliminary thresholds.
- Run shifted batches across sizes and strengths.
- Measure false positives and detection rates.
- Analyze drift versus degradation.

**Exit criteria:** The report includes repeated-run results, batch-size effects, and at least one documented drift-without-degradation case.

### Week 5: Persistence and API

- Add SQLite schema and repositories.
- Add batch ingestion and prediction endpoints.
- Add delayed-label ingestion.
- Store metrics and alert evidence.
- Add API validation and integration tests.

**Exit criteria:** A batch can be ingested, monitored, labelled later, and retrieved with complete evidence.

### Week 6: Dashboard

- Build overview, batch, alert, and model-comparison pages.
- Add filters for batch and model version.
- Add reference/current distribution charts.
- Clearly distinguish labelled and unlabelled evidence.
- Cache expensive reads and calculations.

**Exit criteria:** A recruiter can follow one batch from ingestion through drift alert to delayed performance outcome.

### Week 7: Candidate evaluation and reliability

- Add candidate training with a valid cutoff.
- Add bootstrap metric comparison.
- Implement promotion policy.
- Add Docker.
- Add GitHub Actions.

**Exit criteria:** The system can recommend promotion, rejection, or insufficient evidence without automatically replacing the current model.

### Week 8: Publication

- Complete the experiment report.
- Add architecture and data-flow diagrams.
- Document limitations and ethical boundaries.
- Capture screenshots and a short demo video.
- Add measured results to the README.
- Prepare interview explanations.

**Exit criteria:** Another developer can reproduce the major results from the repository instructions.

## Definition of done

The project is complete when it demonstrates all of the following:

- Reproducible baseline training
- Fixed, non-overlapping data partitions
- Saved inference pipeline and reference profile
- At least three controlled monitoring scenarios
- Repeated drift-detection experiments
- False-positive rates on stable batches
- Batch-size sensitivity
- Drift without degradation
- Degradation supported by delayed labels
- Evidence-backed alerts
- Candidate comparison on untouched data
- Explicit promotion criteria
- Automated tests
- Functional API and dashboard
- Honest documentation of limitations

## Expected report outputs

The final report should contain measured values for:

- Baseline model metrics
- Drift detection rate by scenario
- Stable-batch false-positive rate
- Alert precision and recall for degradation
- Effect of batch size
- Feature-level drift rankings
- Model-performance changes by scenario
- Current-versus-candidate metric differences
- Bootstrap confidence intervals
- Training and inference time
- API latency under a documented local test

Do not fill these sections with illustrative or invented numbers.

## Reproducibility checklist

- [ ] Dataset source and checksum recorded
- [ ] Python and dependency versions recorded
- [ ] Random seeds recorded
- [ ] Data partitions persisted
- [ ] Training configuration versioned
- [ ] Monitoring configuration versioned
- [ ] Model metadata stored with the artifact
- [ ] Reference profile tied to a model version
- [ ] Simulation parameters stored with each batch
- [ ] Experiment results exportable as CSV or JSON
- [ ] Tests pass from a clean environment
- [ ] Final evaluation set remains untouched until candidate comparison

## Ethical and technical limitations

- This is an educational monitoring prototype, not a validated credit-risk system.
- The dataset contains sensitive demographic variables and reflects a historical context.
- Synthetic drift may not represent real changes in customer behavior.
- Statistical drift does not establish causal model failure.
- True performance generally cannot be calculated without suitable labels.
- Delayed or biased labels can make performance monitoring misleading.
- SHAP values describe model behavior, not causal effects or moral justification.
- Aggregate metrics can hide subgroup regressions.
- Retraining can amplify bias or instability if recent data is unrepresentative.
- Monitoring thresholds are environment-specific and must be recalibrated.

## Honest project claims

Appropriate description:

> SentinelML is a reproducible research prototype that detects selected forms of simulated data drift, monitors prediction behavior without labels, evaluates degradation after delayed labels arrive, and compares candidate models using predefined evidence-based criteria.

Avoid claims such as:

- “SentinelML prevents model failures.”
- “The system proves concept drift in real time.”
- “A drift score proves that accuracy decreased.”
- “The platform is production-ready.”
- “Automatic retraining always improves the model.”

## Resume entry template

**SentinelML — ML Monitoring and Drift Evaluation Platform**  
Python, scikit-learn, XGBoost, FastAPI, Streamlit, SQLite, Docker

- Developed a reproducible monitoring system for data-quality issues, feature-distribution drift, prediction changes, and delayed-label performance evaluation.
- Designed repeated controlled experiments to measure drift-detection rates, stable-batch false alarms, batch-size sensitivity, and the relationship between drift and degradation.
- Implemented evidence-backed alerts that record observed values, calibrated thresholds, affected features, model versions, and label availability.
- Compared deployed and candidate models on identical untouched data using predefined promotion criteria and uncertainty estimates.

Replace general statements with measured results only after completing the experiments.

## License

Choose and add a license before public release. An MIT or Apache-2.0 license may be appropriate for the source code, but the dataset remains subject to its own terms and attribution requirements.

## Acknowledgements

- UCI Machine Learning Repository for the benchmark dataset
- scikit-learn, SciPy, pandas, NumPy, XGBoost, SHAP, FastAPI, Streamlit, Plotly, and pytest contributors

---

The value of SentinelML is not the number of tools included. Its value comes from demonstrating, with reproducible evidence, when drift matters, when it does not, and how model-update decisions can be made without confusing an alert with proof of failure.
