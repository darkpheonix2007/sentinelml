# Weak-shift and duration sweep (initial 500-draw analysis)

The main result is the noise before persistence: BH flagged 75/500 stable batches (15%), while only 1/500 became actionable after two-batch persistence. Bonferroni was similar at 73/500 flagged and 1/500 actionable. Persistence, not the choice between BH and Bonferroni, explains most of the reduction. These batches reuse rows, so neither count is a precise population false-alarm probability. See the 5,000-draw follow-up for the fairer BY comparison and single-batch override.

Measured on 2026-11-04 using the same fitted model, reference rows, and 500-draw calibration as the paired alert-rule report. Five repayment-shift strengths (expected pre-clipping +0.05, +0.10, +0.25, +0.50, +1.00 per affected feature) were tested for 1-, 2-, and 3-batch events. PAY_0, PAY_2 and PAY_3 are integer statuses, so each selected row receives a shared +1 with probability equal to the strength; values are clipped at 9. These are **synthetic feature shifts with historical labels retained**.

The +0.05 and +0.10 settings were added after the first +0.25/+0.50/+1.00 sweep saturated BH detection. This is exploratory follow-up on the same simulation pool, not a preregistered or untouched final test.

Each setting has 100 episodes of 500-row batches. A separate stable batch precedes each episode so an existing warning can carry into the first shifted batch. The same source-row seeds are paired across strengths, durations, and correction methods. The rule is actionable when the same feature passes correction in two consecutive batches. Counts below are episodes with at least one actionable alert *during* the event; they do not prove that the event caused the alert.

| Expected increment | Event duration | BH + persistence | Bonferroni + persistence | BY + persistence |
|---:|---:|---:|---:|---:|
| +0.05 | 2 batches | 2/100 | 1/100 | 0/100 |
| +0.05 | 3 batches | 4/100 | 2/100 | 0/100 |
| +0.10 | 2 batches | 11/100 | 8/100 | 0/100 |
| +0.10 | 3 batches | 20/100 | 15/100 | 0/100 |
| +0.25 | 2 batches | 99/100 | 99/100 | 5/100 |
| +0.25 | 3 batches | 100/100 | 100/100 | 10/100 |
| +0.50 | 2 batches | 100/100 | 100/100 | 98/100 |
| +0.50 | 3 batches | 100/100 | 100/100 | 99/100 |
| +1.00 | 2 batches | 100/100 | 100/100 | 100/100 |
| +1.00 | 3 batches | 100/100 | 100/100 | 100/100 |

On a separate 500-batch stable stream (same model and batch size), BH and Bonferroni each generated 1 actionable alert; BY generated 0. Before persistence, BH flagged features in 75/500 stable batches, Bonferroni in 73/500, and BY in 6/500. The earlier combined **warning** rate was 62/500 (12.4%) on different stable seeds and included data-quality warnings. Warnings remain noisy even when actionable alerts are rare.

## Interpretation

- Persistence costs short-event detection. For a one-batch event from a *clean* state, actionable detection is impossible by design. One-batch rows are omitted from the main table: nonzero counts in the raw CSV come from a matching feature flag in the stable prelude and should **not** be claimed as reliable one-batch detection.
- +0.25 and stronger shifts mostly saturate BH and Bonferroni once the event lasts two batches. At +0.10 over three batches, BH catches 20/100 and Bonferroni 15/100, with the same one stable actionable alert. This is a small, uncertain difference, not evidence that BH is clearly better. +0.05 results are near background-alert level.
- BY is much more conservative in this finite calibration. With 23 feature tests and 500 null draws, the smallest possible empirical p-value is 1/501 ≈ 0.0020, but BY's rank-one cutoff at q=0.10 is about 0.0012. A single feature therefore cannot pass BY at this resolution; multiple very small p-values can pass together. Its low weak-shift detection is partly a **resolution limit**, not proof that BY is intrinsically unsuitable.
- Calibration and simulation partitions are already disjoint. The 500 stable batches repeatedly reuse rows from a 2,250-row simulation pool; only four non-overlapping 500-row batches can be formed. The episode repetitions also reuse rows. These counts characterize this finite stress test, **not** independent-trial population detection probabilities or trustworthy binomial confidence intervals.
- Every method uses the same q=0.10 and two-batch persistence. BH controls expected false discovery proportion under suitable dependence assumptions, not the repeated-stream actionable-alert rate; empirical stable monitoring remains necessary.

This is an archived exploratory run whose episode prelude was later removed; the current runner starts episodes clean and will not reproduce the one-batch carryover counts above. Run `python -m sentinelml.cli benchmark-shift-sweep --episodes 100 --stable-batches 500 --size 500 --null-draws 5000` for the current comparison. Its ignored JSON/CSV outputs retain counts, delays and seeds. No GPU is required.
