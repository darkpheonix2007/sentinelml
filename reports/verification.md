# Local verification

Development run: 2026-10-30; alert-rule update: 2026-11-03. Environment: Windows, Python 3.11.9, CPU only.

## Baseline training

Verified archive SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.

Model version: `baseline-20261030T130806128508Z`.

| Model | Validation precision | Recall | F1 | ROC-AUC | Average precision | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3755 | 0.6000 | 0.4619 | 0.6968 | 0.4737 | 0.753 |
| Random Forest (selected) | 0.5082 | 0.5618 | 0.5337 | 0.7723 | 0.5284 | 5.936 |

These are validation/model-selection results, not final holdout estimates. Training time excludes drift calibration and depends on hardware and load.

The separate training-only three-fold audit returned mean F1 0.4772 for Logistic Regression and 0.5432 for Random Forest; mean ROC-AUC was 0.7262 and 0.7789 respectively. It did not change model selection or use the final evaluation partition.

Permutation importance on 1,000 validation rows (five shuffles per feature, average-precision scoring) ranked PAY_0 first: mean AP decrease 0.1205. AGE had near-zero mean contribution (0.00012) on this sample, supporting its use as a weak-feature stress scenario without guaranteeing harmlessness. Correlation between features limits interpretation.

## Candidate experiment

Candidate: `candidate-20261030T131321669020Z`. Scenario: repayment_shift, strength 1.0, 3,000 identical untouched evaluation rows for both models.

| Metric | Deployed model | Candidate |
|---|---:|---:|
| Precision | 0.2470 | 0.3695 |
| Recall | 0.9232 | 0.7334 |
| F1 | 0.3897 | 0.4914 |
| ROC-AUC | 0.6265 | 0.7640 |
| Average precision | 0.3430 | 0.5040 |
| Brier score | 0.4166 | 0.2225 |

95% paired bootstrap interval for recall difference: [-0.2230, -0.1617], 500 resamples.

Decision: **do not promote**. The candidate improves F1, precision and AUC but fails the required recall gain and nonnegative uncertainty bound. No promotion was performed. Training took 6.895 seconds; batch-average inference was 0.0614 ms/row, not an API request latency measurement.

## Application checks

- Synthetic pytest suite after alert-rule, bootstrap and sweep changes: **28 passed, 1 dependency deprecation warning** on 2026-11-04.
- Streamlit interaction test covers simulated ingestion, delayed-label release and all six pages.
- Dashboard rendered successfully in the browser with five stored demo batches.
- Live API `/health` returned `status: ok` and the selected baseline version.
- Live Streamlit `/_stcore/health` returned `ok`.
- One dependency warning remains: current Starlette deprecates the httpx-based test client. Tests still execute; this does not affect production API routes.
- Docker is not installed in this environment, so the supplied Dockerfile has not been built or run here.
- GitHub Actions has been configured but not run remotely; changes have not been pushed.

## Repeated monitoring experiment

The original 30-repeat, 1,560-batch study used the **old** per-feature rule and fixed-drop degradation definition. Stable batches generated an alert 43.3% of the time; any-alert precision for that old threshold-defined degradation outcome was 48.1%, recall 75.2%. This high false-alert rate motivated the update. See [experiment_summary.md](experiment_summary.md) for historical scenario examples. The old generated CSV/JSON remain local artifacts; rerunning the experiment with the updated code uses the new definitions.

The paired updated-rule benchmark used 500 stable and 500 repayment-shift batches of size 500. Stable actionable alerts fell from 150/500 to 1/500; repayment detection changed from 500/500 to 499/500 with first-alert delay rising from one to two batches. See [alert_rule_comparison.md](alert_rule_comparison.md). These are synthetic, finite-pool results, not production guarantees.

The original expanded sweep used 100 episodes per strength/duration (five strengths, three durations) and a separate 500-batch stable stream. Its finite-pool caveats are in [shift_sweep.md](shift_sweep.md). The 1/500 stable actionable rate has a naive independent-binomial Wilson upper bound near 1.1%, but row reuse prevents treating that as a deployment confidence bound. A follow-up with 5,000 null draws, clean episode starts, BY and a strong-single-batch override is documented in [shift_sweep_5000.md](shift_sweep_5000.md). It found 79/500 BH stable pre-persistence flags and 3/500 actionable alerts; the override recovered 90/100 one-batch +0.50 events with 0/500 override triggers on the reused-row stable stream.

The full local test suite after the ACS and override changes passed on 2026-11-04: **34 passed, 1 dependency deprecation warning**. An ACS 2014→2015 California smoke test completed with 8 disjoint 500-row batches per stream at 20% sampling; results are preliminary and must not be treated as an operating-rate estimate. The categorical-frequency optimization has a count-preservation regression test. A separate test confirms that the experimental strong-single-batch override is disabled in the live credit monitor by default.

The completed optional ACS run trained XGBoost on 2014 California, then evaluated 2015–2018 and 2021 California plus 2018 Texas with 5,000 null draws. It produced 75–78 disjoint batches per California year and 54 for Texas, plus 14 disjoint 2014 stable batches. The nine-feature 2018 California stream had 77/78 actionable alerts; a **post-hoc** occupation-code sensitivity check with `OCCP` removed from every year had 0/78. This is documented in [acs_validation.md](acs_validation.md), with raw machine-readable outputs in ignored `artifacts/acs_validation*/`.

## Tested direct runtime versions

Python 3.11.9; NumPy 2.4.6; pandas 3.1.0; scikit-learn 1.9.1; SciPy 1.17.1; joblib 1.6.0; PyYAML 6.0.3; xlrd 2.0.2; FastAPI 0.162.0; Uvicorn 0.54.0; Streamlit 1.66.1; Plotly 6.9.0; pytest 9.2.0; httpx 0.28.1.

This is a tested version list, not a complete transitive lockfile. Model artifacts are tied to the training library versions. Raw data, model binaries and SQLite databases remain ignored by Git.
