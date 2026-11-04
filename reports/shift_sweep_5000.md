# Clean-state shift sweep and 5,000-draw correction check

Measured 2026-11-04 on the UCI credit benchmark. Each method uses the same 500-row batches, 5,000 null calibration draws, per-batch q=0.10, and two-consecutive-batch persistence. The optional BH override additionally escalates a single batch if at least three BH-rejected features reach the empirical p-value floor (1/5,001). Each simulated episode starts with an **empty** alert state, so persistence-only methods correctly have **N/A by design** for one-batch actionable detection. The override is a heuristic to recover only very strong short events.

The finding to lead with is pre-persistence noise: on the 500-batch stable stream, BH flagged 79/500 batches (15.8%); Bonferroni flagged 76/500 (15.2%); BY flagged 31/500 (6.2%). Two-batch persistence reduced actionable counts to 3/500, 3/500, and 0/500 respectively. The override fired on **0/500** stable batches, so BH plus override also had 3/500 actionable. In the original 500-draw run, BH flagged **75/500** (15%) stable batches and generated 1/500 actionable; the change to 79 and 3 is one reason not to advertise a fixed operating rate. These batches reuse rows from a 2,250-row simulation pool; none of these counts is a defensible independent-trial estimate or deployment guarantee.

## Actionable episodes by shift strength

Each setting has 100 episodes. The shift adds one integer status step to PAY_0, PAY_2 and PAY_3 on a random fraction of rows equal to the stated strength, with clipping at 9; historical labels are retained. The same source-row seeds are paired across methods and strengths. Counts are episodes with at least one actionable alert during the event.

| Expected increment | Duration | BH | Bonferroni | BY | BH + strong override |
|---:|---:|---:|---:|---:|---:|
| +0.05 | 2 | 0/100 | 0/100 | 0/100 | 0/100 |
| +0.05 | 3 | 2/100 | 1/100 | 0/100 | 2/100 |
| +0.10 | 2 | 7/100 | 4/100 | 2/100 | 7/100 |
| +0.10 | 3 | 15/100 | 10/100 | 5/100 | 15/100 |
| +0.25 | 2 | 98/100 | 98/100 | 97/100 | 98/100 |
| +0.25 | 3 | 100/100 | 100/100 | 99/100 | 100/100 |
| +0.50 | 2 | 100/100 | 100/100 | 100/100 | 100/100 |
| +0.50 | 3 | 100/100 | 100/100 | 100/100 | 100/100 |
| +1.00 | 2 | 100/100 | 100/100 | 100/100 | 100/100 |
| +1.00 | 3 | 100/100 | 100/100 | 100/100 | 100/100 |

One-batch events from a clean state are **N/A by design for BH, Bonferroni and BY plus persistence**. The override caught 0/100 at +0.05, 0/100 at +0.10, 2/100 at +0.25, 90/100 at +0.50, and 100/100 at +1.00. These are the relevant short-event results, and the 0/500 stable override count is encouraging but far too small and row-reused to establish its real false-alert rate.

## Interpretation and limits

- BH versus Bonferroni remains nearly a wash. Both had three stable actionable batches, and the weak-shift gaps (for example 15 vs 10 of 100 at +0.10 for three batches) are small relative to finite-pool reuse. Do not claim BH is clearly superior.
- The 5,000-draw p-value floor is about 0.00020, below BY's rank-one cutoff near 0.0012 for 23 tests at q=0.10. BY now detects 97/100 two-batch +0.25 events; its near-zero result at that strength in the 500-draw run was substantially a calibration-resolution artifact.
- Feature-wise calibration draws themselves reuse rows, and the episode/stable batches repeatedly reuse simulation rows. The same source pool supports exploratory threshold comparisons; it is not an independent external test. Pairing makes comparisons more interpretable but does not remove uncertainty.
- An input shift with historical labels retained may produce unrealistic feature/label combinations. The second, natural-shift ACS validation uses genuinely later-year inputs and labels, but Census code changes and sampling still complicate interpretation.
- With 15%–16% first-pass BH/Bonferroni flags, warning noise is an operational concern even if only critical alerts page someone. Persistence does most of the alert reduction. A low observed critical count does not make the warning stream usable.

Reproduce with `python -m sentinelml.cli benchmark-shift-sweep --episodes 100 --stable-batches 500 --size 500 --null-draws 5000`. Ignored outputs: `artifacts/experiments/shift_sweep_summary_5000.json` and `shift_sweep_episodes_5000.csv`. No GPU is required.
