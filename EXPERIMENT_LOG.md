# Experiment log (Group 16)

One entry per change: what, why, result, kept or dropped. Numbers are validation (Sep 2026) unless marked test (Oct 2026).

| Date | Change | Why | Result | Decision |
|---|---|---|---|---|
| 10-08 | Baselines: persistence, hist mean (hospital × weekday × hour) at 3 h / 24 h | Proposal: every model must beat both | Hist mean wins at 3 h and 24 h (MAE 51.6 / 51.1); persistence only at 1 h | Kept as the bar |
| 10-30 | Added 2023-2024 general holidays (gov.hk lists) | 1823 iCal feed starts in 2025; all earlier holidays were 0 | Holiday hours 12,942 → 28,494 | Kept |
| 10-30 | Downloaded Oct 2026 as an untouched test month | Proposal asks for new data | Archive has 0 versions on 7 days | Kept; wrote `pull_live.py` backup for Nov |
| 10-30 | Hourly grid re-index before shifting | Gaps made `shift(k)` ≠ k hours | Exact-hour lags | Kept |
| 10-30 | Weather enters as previous-day value | Same-day daily value is not known at forecast time | Leakage removed | Kept |
| 10-30 | Hist-mean for "> 2 h" computed on new format only | All-years training set is 70% positive vs 53% in September | Baseline acc 0.62 → 0.77 (val) | Kept |
| 10-30 | v1 models: OLS, ridge, lasso, RF, grad boosting, logistic | Proposal progression | 3 h MAE 40.9 (GB) vs 51.5; 24 h 50.1 vs 51.5 | Kept GB as main model |
| 10-30 | Ridge/lasso alpha grid | Regularisation | MAE flat across alpha (43.68 → 43.69) | Not a lever; report only |
| 10-30 | Drop-one-group ablation | Secondary question | History +13.0, calendar +1.9, hospital +1.5, weather −0.3, AQHI −0.1 (3 h) | → v2 |
| 10-30 | v2: remove weather + AQHI | Ablation + no Oct weather published | 3 h test 41.6 → 41.1; 24 h test 51.1 → 52.4 (val 50.1 → 48.7) | Kept for 3 h; 24 h still open |
| 10-30 | Era flag (A) vs new-format-only training (B) | Regime shift | Test AUC 0.892 vs 0.893 at 3 h | Equal; A kept (more data) |
| 11-04 | Downloaded 30 Oct – 3 Nov; saved separately as `hourly_features_latest.csv` | Live test needs November; keep the 30 Oct table frozen so 10-30 numbers stay reproducible | 595,530 rows to 4 Nov 17:00; no empty days in this window; HKO weather still ends 30 Sep | Kept |
| 11-04 | 24 h: residual target, shrinkage towards the hist mean, slow history (7-day same-hour mean, 7-day rolling mean) | 24 h model did not beat the hist mean in October | Val 48.73 (original) → 48.61 residual, 48.40 residual + slow, 48.02 slow; best shrinkage weight ≈ 1. Oct test 50.73 vs hist 51.97 | Slow history kept; residual and shrinkage rejected |
| 11-04 | 3 h: 54-configuration HGB search (learning rate, leaves, min leaf, L2) | 3-point grid might be too coarse | Val 40.62 → 40.51; Oct test 41.12 → 41.20 | No test gain; original setting kept |
| 11-04 | Isotonic calibration of the 3 h classifier | Check probability quality | Sept Brier 0.1207 → 0.1197 (fitted on Sept); Oct 0.1326 → 0.1333 | Not adopted; already well calibrated |
| 11-11 | Downloaded 1–10 Nov (archive empty on 6–7 Nov); `live_test.py` scores the frozen models, no tuning | Proposal: test on new data | 3 h: 39.5 vs hist 53.2 (diff −13.7, day-bootstrap 95% CI −18.4 to −10.4). 24 h slow history: 47.4 vs 53.2 (−5.8, CI −8.8 to −2.7). Full Oct 24 h: −1.5 (CI −2.6 to −0.4) | 3 h confirmed; 24 h gain later shown to be level drift, not skill (see the post-hoc row below) |
| 11-11 | Pairwise check on November: which of two hospitals is quicker in 3 h (`live_predictions.py`) | Turn MAE into a decision a patient makes | Correct calls: board now 69.3%, hist mean 74.0%, model 81.5% (25,529 pair-hours with a real gap of 30 min or more) | Reported; nothing tuned on it |
| 11-11 | Independent review by a second AI agent, then `posthoc_checks.py`: three one-parameter rules, a decision tree, boosting with level features, quantile boosting | Reviewer argued the historical mean (a 10-month average) was too weak a baseline | Best rule (28-day mean + 0.62 × current anomaly): 3 h 43.4 Oct / 41.2 Nov vs boosting 41.1 / 39.5 (gain 2.4 and 1.7 min, CIs exclude 0, 22/24 and 8/8 days). 24 h: rule 50.4 / 47.9 vs slow-history boosting 50.2 / 47.4, within noise. Pairwise: rule 81% vs model 82%. 10–90% range covers 76% | Conclusions rewritten: most of the 3 h gain is information a simple rule can use; 24 h result is negative. Post-hoc, not blind |
| 11-11 | Zero-shot foundation model: `amazon/chronos-bolt-base`, context 512 h per hospital, no training (`foundation_baseline.py`) | Course is about foundation models; a general pretrained forecaster is a natural baseline | 3 h: 42.8 Oct / 40.5 Nov (rule 43.4 / 41.2, boosting 41.1 / 39.5); 24 h: 52.0 / 50.3 (worse than the rule); 10–90% range covers 79% / 81% at 3 h | Reported as a baseline. Post-hoc, not blind |
