# ACSIncome-derived natural-shift validation

Measured 2026-11-04 using Folktables 0.0.12, XGBoost 3.2.0 CPU histogram trees, and 20% deterministic subsampling of official ACS person files. The model was fitted only on 2014 California; disjoint 2014 subsets provided a baseline, reference distributions, 5,000-draw null calibration, and a stable holdout. Later years and Texas were not used for fitting or calibration. Each stream used non-overlapping 500-row batches, though the benchmark is not a survey-weighted population estimate. The nine-feature task drops `RELP` in every year because the field is unavailable after 2018; all other standard ACSIncome features and the adult filter/nominal $50,000 target were retained.

The 2014 validation baseline had F1 **0.722**, recall **0.694**, and ROC-AUC **0.885**. The table reports batch means (not pooled metrics) and BH at q=0.10 with two-batch persistence. A “degradation” is an operational bootstrap screen: the batch recall or F1 upper 95% bound is below the fixed 2014 validation point estimate, with at least 30 positive labels. The baseline's own uncertainty is omitted. No causal interpretation is justified.

| Stream | Disjoint batches | Mean accuracy | Mean F1 | BH flags | Actionable BH alerts | Alerts without confirmed degradation | Degradation without alert |
|---|---:|---:|---:|---:|---:|---:|---:|
| CA 2014 held-out | 14 | 0.814 | 0.729 | 1 | 0 | 0 | 0 |
| CA 2015 | 75 | 0.809 | 0.726 | 9 | 0 | 0 | 4 |
| CA 2016 | 76 | 0.803 | 0.727 | 8 | 0 | 0 | 2 |
| CA 2017 | 77 | 0.802 | 0.730 | 8 | 0 | 0 | 6 |
| CA 2018 | 78 | 0.789 | 0.719 | 78 | 77 | 59 | 0 |
| CA 2021 | 77 | 0.765 | 0.703 | 77 | 76 | 21 | 1 |
| TX 2018 | 54 | 0.789 | 0.714 | 54 | 53 | 50 | 0 |

Bonferroni and BY also produced 0 actionable alerts on the 2014 holdout and on 2015–2017 California. They produced the same 77/78, 76/77 and 53/54 actionable counts for CA 2018, CA 2021 and TX 2018 respectively; the methods are nearly indistinguishable for these strong shifts. BH plus the single-batch override matched BH on California streams and raised TX 2018 to 54/54 actionable batches. Do **not** infer that this override has a low stable false-alert rate from only 14 stable ACS batches.

## The 2018 codebook confound

On a randomly sampled 500-row diagnostic batch, occupation (`OCCP`) Jensen–Shannon divergence against 2014 reference was about 0.05 in 2015/2017 and about 0.15–0.16 in 2018/2021. These are within-feature comparisons, not calibrated p-values. The [Census changed ACS occupation coding in 2018](https://www.census.gov/topics/employment/industry-occupation/about/occupation.html), so the 2018/2021 alert flood is at least partly a schema/codebook incompatibility, **not** evidence that workers or model reliability suddenly changed. The old 2014 XGBoost encoder may also treat new occupation codes as unseen, affecting prediction quality. An occupation-excluded sensitivity run is required before drawing a temporal conclusion from those years. Other variables, survey design, inflation and label definition may still differ afterward.

The 2015–2017 mismatch is also instructive: 12 batches collectively screened as degraded without an actionable feature-drift alert. Input monitoring cannot establish performance without labels, and a two-batch rule can miss short or feature-subtle changes. Conversely, CA 2018 had 59 actionable batches without confirmed degradation, and TX 2018 had 50. “No confirmed degradation” is not proof of equal performance; it includes finite-sample uncertainty and a fixed-baseline screening rule.

## Occupation-excluded sensitivity run

I reran 2014 fitting/calibration and 2018/2021/Texas evaluation with `OCCP` removed in **every** year. This is a different eight-feature model, not a post-hoc suppression of alerts from the nine-feature model. Data rows, sampling density, 5,000 null draws, q, batch size and persistence settings stayed the same.

| Stream | Nine-feature BH actionable | Eight-feature BH actionable | Eight-feature mean F1 | Eight-feature degradation without alert |
|---|---:|---:|---:|---:|
| CA 2014 held-out | 0/14 | 0/14 | 0.711 | 1 |
| CA 2018 | 77/78 | **0/78** | 0.714 | 12 |
| CA 2021 | 76/77 | **76/77** | 0.702 | 0 |
| TX 2018 | 53/54 | **53/54** | 0.708 | 0 |

The 2018 California collapse from 77 alerts to zero is strong evidence that the occupation coding transition dominated the original monitor result. It also exposes 12 batches with degradation evidence that input drift no longer escalates. The 2021 and Texas alerts survive this ablation, but their causes are **not isolated**. In particular, the [Census cautions that its race-question and coding changes starting in 2020 affect comparisons with prior race estimates](https://www.census.gov/programs-surveys/acs/technical-documentation/user-notes/2021-03.html); the 2021 `RAC1P` distribution is a plausible contributor, not a proven sole cause. Texas mixes geography with possible composition/coding differences. Repeated post-hoc feature removal would overfit the benchmark; a defensible next study would predefine harmonized feature crosswalks and then hold out a new year/state.

Run `python -m sentinelml.acs_validation --download --years 2015 2016 2017 2018 2021 --compare-state TX --compare-year 2018 --density 0.2 --max-batches 100 --calibration-draws 5000`. Exact machine-readable counts and per-batch results are in ignored `artifacts/acs_validation/`. The 2014 stable stream has only 14 disjoint batches, so no robust false-alert-rate interval is claimed. Raw files are ignored by Git and excluded from the project ZIP.

Run the sensitivity check with the same command plus `--exclude-occupation` (the cached raw data will be reused). Its separate outputs are under ignored `artifacts/acs_validation_no_occupation/`.
