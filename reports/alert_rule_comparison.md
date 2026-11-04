# Paired alert-rule comparison

Measured on 2026-11-03 with the same fitted Random Forest and the same source-row seeds in both rule evaluations. Each scenario used 500 sequential, separately seeded 500-row batches from the held-out simulation pool. Separately seeded does **not** mean statistically independent: rows are reused heavily. The repayment stress increases PAY_0, PAY_2 and PAY_3 by two units (clipped to the documented range) while retaining historical labels. Each scenario starts with an empty persistence streak.

| Measure | Old per-feature rule | BH + two-batch persistence |
|---|---:|---:|
| Stable actionable/any-alert batches | 150 / 500 (30.0%) | 1 / 500 (0.2%) |
| Repayment-shift detection batches | 500 / 500 (100.0%) | 499 / 500 (99.8%) |
| First repayment alert after shift starts | Batch 1 | Batch 2 |

The old-rule column means any raw per-feature 99th-percentile exceedance or data-quality warning. The new actionable column means a BH-corrected feature flag (empirical p-values, FDR target 0.10) on at least two consecutive batches. The same 500 calibration draws per batch size supply both rules' cutoffs; the comparison is paired at the source-row level. The model itself did not change.

Warnings were **not** silently removed: with the new rule, 62/500 stable batches (12.4%) and 166/500 repayment batches (33.2%) still had at least one warning. First corrected feature flags and immediate quality issues stay visible as warnings; only persistent corrected feature flags are actionable.

The observed 1/500 actionable stable rate is 0.2%. Under an *independent-binomial assumption*, its 95% Wilson interval is about 0.04%–1.12%; a sensible rough description is an upper end near 1%, **not** a guaranteed 0.2% operating rate. Because rows are reused, this interval is illustrative rather than a defensible population confidence interval. The 12.4% stable warning rate is operational noise even if warnings are not paged; users may learn to ignore them.

The historical 43.3% figure in [experiment_summary.md](experiment_summary.md) came from a different 30-repeat study with 100 calibration draws and multiple scenarios. It is not the before column of this paired benchmark.

## What these numbers do and do not show

- Strong evidence that this revised rule reduces *actionable* false alerts in the chosen stable simulation, with a one-batch detection delay for this strong repayment perturbation.
- No proof that it retains detection of weaker, shorter-lived, or different shifts. Persistence can miss a one-batch event entirely.
- The 500 batches repeatedly sample a finite pool of 2,250 rows, so they are not 500 independent real-world periods. Calibration and simulation pools are already disjoint; splitting the simulation pool again would leave even fewer unique rows. At most four non-overlapping 500-row batches fit in it. A larger independent, preferably time-ordered test set is needed for stronger population claims.
- Feature tests are dependent, and empirical p-values have 1/501 resolution. BH assumptions and the chosen 0.10 target should be stress-tested with other seeds, sizes, and datasets.
- The repayment perturbation retains historical labels. This benchmark measures shift detection, not causal model failure or detection of real credit-risk deterioration.

Reproduce with `python -m sentinelml.cli benchmark-alerts --batches 500 --size 500` after training or `recalibrate-monitor`. The ignored local outputs are `artifacts/experiments/alert_rule_comparison.json` and `alert_rule_batches.csv`.
