\begin{{abstract}}
We forecast how long semi-urgent and non-urgent patients will wait at Hong Kong's {n_hosp} public Accident and Emergency (A&E) departments—specifically the Hospital Authority's published category 4/5 median wait—3 hours and 24 hours ahead. Because the live feed publishes only current waits, we rebuilt {n_rows_blind} hospital-hours of history (1 Jan 2023 to 25 Nov 2026) from the government archive, reconciled an October 2025 format change, and compared five model families against simple baselines. On October 2026, held out from model fitting, gradient boosting cuts 3-hour MAE from {hm_mae3} to {gb_mae3} minutes ({gain_mae3}% below the historical average); on 1–10 November it scores {nov_gb3} against {nov_hm3}. However, a one-parameter rule (28-day same-hour mean plus scaled current deviation) already reaches {r3_oct} minutes on complete October and {r3_nov} in early November, so boosting adds ~2 minutes and zero-shot Chronos-Bolt lands within {fm_gap} minutes. In a post-freeze blind test on 11–25 November (14 days, preserving the pre-registered 11–15 November subset), boosting scores {gb3_ext} against {r3_ext} for the rule and {ext_hm} for the historical mean; Chronos-Bolt matches 3-hour boosting ({fm3_ext}) with better-calibrated 10–90% coverage ({fmc3_ext} vs {qc3_ext}), and at 24 hours no supervised model beats the rule (only Chronos-Bolt edges ahead on 14 blind days, {fm24_ext} vs {r24_ext}). Recent waits drive the signal; daily weather and AQHI added none.
\end{{abstract}}


# Introduction

Almost everyone in Hong Kong has waited at an A&E, where congestion varies widely across hospitals and hours. Prior ED wait-time studies predict individual triage percentiles via quantile regression \citep{{sun2012edwait}} or combine queueing fluid estimators with regularised regression (Q-Lasso) to avoid rolling-average underestimation \citep{{ang2016edwait}}; both rely on internal queue telemetry. Here, the Hospital Authority (HA) publishes only a public 15-minute feed \citep{{ha_feed}}, giving current waits without a forecast. A patient deciding whether to go now, later, or elsewhere needs the expected wait at arrival.

**Objective.** Predict HA's published category 4/5 median wait (rounded in 30-minute steps) 3 hours and 24 hours ahead, in minutes (regression) and as "over 2 hours or not" (classification), and test which feature groups matter most.

**Contributions.** (i) A cleaned dataset of {n_rows_blind} hospital-hours rebuilt from {n_files} archived snapshots, with two feed formats reconciled and archive gaps documented. (ii) A chronological protocol using only information available at forecast time, with one validation month and three test windows, culminating in a 14-day post-freeze blind test (11–25 November, preserving the pre-registered 11–15 November subset). (iii) Evidence that the 3-hour wait is predictable, with an honest decomposition: most of the gain is available to a one-parameter rule, gradient boosting adds a consistent margin, and zero-shot Chronos-Bolt matches the trained model under distribution shift; at 24 hours we report a negative result. (iv) An ablation showing congestion follows its own momentum and the clock, not weather or air quality.

# Data and Preprocessing

## Sources and access

**A&E waiting times.** HA's open-data feed \citep{{ha_feed}} is one JSON file covering all {n_hosp} public A&Es. The live file holds only the latest snapshot, so we rebuilt the history through the data.gov.hk historical-archive API \citep{{datagov_archive}}, which stores past versions of the file. Three problems shaped the downloader (`src/download_ae_archive.py`):

- the version-listing endpoint silently truncates at about 35,000 results, so we query one calendar month at a time;
- it rejects end dates of today or later, so each query is capped at yesterday;
- 96 snapshots a day over nearly four years is ~130,000 files, so we kept one snapshot per hour and ran 8 resumable workers.

The result is {n_files} snapshot files parsed into {n_hours_blind} non-empty hourly slots ({n_rows_blind} hospital-hours) from 1 January 2023 to 25 November 2026 (15 archived responses returned server errors, so every retained row holds one snapshot; 17,889 of the 33,540 retained snapshots carry timestamps after `:00` and are floored to the hour bin). The archive itself is incomplete: it has **no versions at all** on {n_gap_days} days of October 2026 ({gap_days}), 6–7 November, 11 November, and 12 days from 29 August to 9 September 2025.

**Auxiliary data.** Daily weather (rainfall, humidity, mean, maximum and minimum temperature) from the Hong Kong Observatory \citep{{hko_daily}}; the hourly Air Quality Health Index (AQHI) from the Environmental Protection Department \citep{{epd_aqhi}}, matched to the station nearest each hospital; and public holidays \citep{{hk_holidays}}. The government's holiday feed starts in 2025, so every 2023–2024 holiday was silently labelled a normal day until we added the official lists by hand ({n_holiday_hours} holiday hospital-hours after the fix, 12,942 before).

## Cleaning and feature preparation

- **Text to minutes.** Waits arrive as strings such as "less than 15 minutes" or "2.5 hours". The parser (`src/parse_ae.py`) is unit-tested and left no unparsed value. The values are coarse: every new-format wait in our hourly table is a multiple of 30 minutes, and {q120} of hours sit at exactly 120 minutes.
- **Two formats.** Before 13 October 2025 each hospital reported one category such as "Over 3 hours"; since then it reports the median and 95th-percentile wait per triage group. The old category is only a lower bound, so the minute-level target exists only from October 2025 ({n_rows_new_blind} hospital-hours). The "over 2 hours" label covers all four years, defined as strictly `> 120` minutes in the new format (so exact `120`-minute rows are labelled `0`) versus `"Over 2 hours"` and above in the old format.
- **Exact time steps.** Missing snapshots mean "the previous row" is not always "one hour earlier". We re-index every hospital to a complete hourly grid before computing any lag, so a lag of *k* is exactly *k* hours and missing hours stay missing ({n_missing_pct_blind}% of the full grid).
- **No look-ahead.** Forecasts are aligned to the hourly grid, issued once that hour's snapshot arrives, and target the 3rd or 24th hourly bin ahead; features use waits up to the origin snapshot, AQHI at that hour, and the *previous* day's weather. In archive gap hours lacking an origin snapshot ({miss3_blind} blind rows at 3 h; {miss24_blind} at 24 h), forecasts still enter evaluation using each method's missing-input fallback (excluding them preserves all conclusions, Section 5.4).

## What the data shows

![Left: mean category 4/5 wait by hour of day. Right: monthly share of hours with a wait over 2 hours; the dashed line is the format change on 13 October 2025.]({fig_eda}){{width=74%}}

- **A strong daily cycle (through September 2026).** On weekdays the mean wait bottoms out at about {trough_min} minutes at {trough_hr}:00 and peaks at about {peak_min} minutes at {peak_hr}:00. Weekend and holiday evenings are {wkend_gap} minutes shorter at 21:00.
- **Large differences between hospitals**, from {least_busy_min} minutes ({least_busy}) to {busiest_min} minutes ({busiest}) on average.
- **A break at the format change.** In October 2025 itself the old feed put {oct25_old}% of hours above 2 hours and the new feed {oct25_new}%. Part of the drop is the definition (lower bound against strict `> 120` median), not the patients. A model trained on the old years without accounting for this over-predicts long waits.

# Methodology

**Task.** Each row is one hospital at one forecast hour *t*; the target is the value at *t* + *h* for *h* = 3 and 24 hours.

**Baselines.** *Persistence*: the wait at *t* + *h* equals the wait now. *Historical mean*: the training-set average for the same hospital, weekday and hour. A model is only useful if it beats both. After the live test we added stronger rules (below and Section 5.4).

**Features.**

| Group | Features |
|------------------------|------------------------------------------------------------------------|
| Calendar (target hour) | hour, weekday, month, weekend, public holiday |
| Hospital | hospital identity |
| Recent history | wait now, 1 h and 3 h ago; same clock hour yesterday and last week; 6 h and 24 h rolling means; over-2 h flag now |
| Historical mean | the baseline value itself, as a feature |
| Air quality, weather | AQHI now; previous day's rainfall, humidity and temperatures (dropped in the final model, Section 5.2) |
| Era | old or new feed format (classification only) |
| Slow history (24 h only) | mean of the same clock hour over the last 7 days; 7-day rolling mean (added in Section 5.2) |

**Models.** We followed the progression taught in the course: ordinary least squares (OLS) with calendar and hospital only, OLS with all features, ridge and lasso, random forest \citep{{breiman2001}}, and histogram gradient boosting \citep{{friedman2001}} with an absolute-error loss, which matches the MAE metric and is robust to the long right tail of waits. For "over 2 hours" we used logistic regression, random forest and gradient boosting. All models come from scikit-learn \citep{{scikit-learn}}.

**Stronger rules and foundation model (Section 5.4).** We tested three rules using only waits up to *t*: the 28-day mean of the target's clock hour; the historical mean plus a weight times the current deviation from it; and the *level + anomaly* rule,

```{{=latex}}
\[\hat{{y}}_{{i,t+h}} = \bar{{y}}^{{(28)}}_{{i,\,t+h}} + \beta_h \bigl(y_{{i,t}} - \bar{{y}}^{{(28)}}_{{i,\,t}}\bigr),\]
```

where $\bar{{y}}^{{(28)}}$ is the trailing 28-day same-hour mean and $\beta_h$ is a single least-squares coefficient fitted on training rows. We also fitted one decision tree, quantile gradient boosting (10%, 50%, 90%), and ran Chronos-Bolt \citep{{ansari2024chronos}} zero-shot on the last 512 hours of each hospital's own series.

**Main decisions and the alternatives we considered.**

| Decision | Alternative considered | Evidence | Outcome |
|--------------------|--------------------|--------------------------|--------------|
| Hist. mean on new format only (cls.) | Average over all years | Old {pos_old}% vs {pos_new}% now; val acc {hm_acc_old} → {hm_acc_val} (all-yrs → new-fmt) | Kept: fairer bar |
| Train classifier on all years + era | New format only | Test AUC {auc_A3} vs {auc_B3} at 3 h | Equal for boosting; kept |
| Gradient boosting as main model | Linear models, forest | 3 h test MAE {gb_mae3} vs {ols_mae3} (OLS) and {rf_mae3} (RF) | Kept |
| Drop weather and AQHI | Keep all features | Dropping either lowers 3 h val MAE (Fig 2a); Oct weather missing | Dropped |
| Slow history for 24 h | Residual target; shrinkage | Val MAE {v_slow} vs {v_orig} (Fig 2b) | Kept on val; no gain vs rule (5.4) |
| Default 3-point grid for 3 h | 54-config search | Val {tuned_val} vs {gb_val3}; test {tuned_test} vs {gb_mae3} | No gain; original kept |
| Compare vs one-parameter rules | Historical mean only | Best rule scores {r3_oct} vs {post_hm_oct} at 3 h (Table 3) | Conclusions rewritten |
| Uncalibrated classifier | Isotonic calibration \citep{{zadrozny2002}} | Better on Sep in-sample; Oct Brier {brier_iso} vs {brier_raw} raw | Kept raw (decided on Oct) |

\Needspace{{15\baselineskip}}

# Experiments and Reproducibility

## Experimental Protocol

**Splits.** Rows are split by the time of the *target*, so no training target lies in a later period (H = 24 classification uses 570,672 training rows; complete October 1–31 has 10,242 rows).

| Set | Target hours | Use | Regression rows | Classification rows |
|-----------|-------------------|--------------------------------|----------:|----------:|
| Train | to 31 Aug 2026 | fitting | {n_train_reg} | {n_train_cls} |
| Validation | September 2026 | choices for the reported models | {n_val} | {n_val} |
| Test | 1–29 Oct 2026 | scored after the 30 Oct choices | {n_test} | {n_test} |
| Live test | 1–10 Nov 2026 | downloaded on 11 Nov, scored once | {nov_n} | {nov_n} |
| Blind test | 11–25 Nov 2026 | post-freeze blind (4 d {prereg_n} + 10 d; reg & pairwise) | {ext_n} | – |

**Leakage controls.** (i) Chronological split by target time. (ii) Lags on an exact hourly grid. (iii) Previous-day weather only. (iv) `hist_mean` uses training rows only (< 1 Sep); on training rows itself it acts as mild target encoding that cannot leak val/test labels (dropping it changes val MAE by {abl_v2_hm3} / {abl_v2_hm24} min in the final model, and {abl_hm3} / {abl_hm24} min with all features). (v) Hyper-parameters, feature groups, and the 24 h variant were selected on September and scored on October and 1–10 Nov (October tested the initial all-feature 3 h model at {v1_gb_mae3} vs {gb_mae3} min without weather/AQHI and informed the 4 Nov tuning/slow-history checks; both settings appear side by side). Section 5.4 was added after 1–10 Nov, so we pre-registered `PREREGISTRATION.md` on 11 Nov and re-fitted deterministically on pre-September rows with frozen settings for 11–25 Nov (regression, intervals, and 3 h pairwise; classification stopped at 1–10 Nov).

**Metrics.** Regression: mean absolute error (MAE, minutes) and R²; RMSE is in the result files. Classification: accuracy, precision and recall at a 0.5 threshold, ROC-AUC and the Brier score. For differences between methods we report a 95% bootstrap interval that resamples whole days (2,000 resamples), because hours within a day are correlated. With 8 days in early November and 4 in 11–15 Nov, day-level percentile bootstrap intervals are coarse and can under-cover (while remaining wide), so we also count days won and extend the blind window to 14 days (11–25 Nov).

**Hyper-parameters and seeds.** Gradient boosting: 400 iterations, no early stopping, learning rate 0.05, 63 leaves for regression (chosen from three settings on validation) and 31 for classification. Random forest: 150 trees, minimum leaf 20, half of the features per split, fitted on at most 150,000 sampled rows. Ridge and lasso strengths chosen on validation from 0.1–100 and 0.01–1. Logistic regression: C = 1. Decision tree: minimum leaf 50, depth chosen on validation from 6, 10 and 14 ({dt3} at 3 hours, {dt24} at 24). Rule weights: *level + anomaly* {b3} at 3 hours and {b24} at 24; *historical mean + damped anomaly* {b1_3} and {b1_24}. Quantile boosting uses the regression settings. Every random seed is 0.

**Differences from the proposal.** October became a held-out test month instead of a second validation month. The 1–15 November test window from the proposal became the 1–10 Nov live test plus the 11–15 Nov pre-registered blind test, then extended to 25 Nov (22 November days total). The decision tree is in Section 5.4. We report precision and recall instead of a confusion matrix, and a drop-one-group ablation instead of tree feature importance. The planned live-feed collector was written but never scheduled, so the tests rely on the archive.

## Data and Code for Reproduction

**Data and code versions.** Raw data: archive versions of the two HA feed URLs \citep{{ha_feed,datagov_archive}}, one per hour, 1 January 2023 to 25 November 2026. The 30 October runs read `hourly_features_full.csv` ({n_rows} rows to 29 Oct); the 11 November live test reads `hourly_features_latest.csv` ({n_rows_latest} rows to 10 Nov); and the blind test reads `hourly_features_blind.csv` ({n_rows_blind} rows to 25 Nov). Code, result files and the experiment log are public at <https://github.com/JimmyChan233/elec3544-ae-waiting-times>: commit `2eb14ae` (tag `freeze-2026-11-11`, original ≥8 pp pairwise bar), same-day amendment `ae49a32` (≥2 pp after scoring board ties consistently as half, before 11–15 Nov data existed), `ba7ad35` (4-day pre-reg run), `7374b68` (14-day extension), and tag `final-2026-11-27` (this verified report). All three processed CSVs are in release `data-2026-11-26` (`Group16-Project-data.zip`, SHA-256 `{zip_sha}…`).

**Reproducing the results.** Clone the repository at `final-2026-11-27`, install `requirements.txt`, and unzip the data release into `data/processed/` (or rebuild from the archive). Running the scripts in Supplementary Materials regenerates every number, table and figure, and `python3 report/build_final.py --official --zip data/processed/Group16-Project-data.zip` compiles this PDF. On complete October (including 30–31 Oct), the frozen 3-hour model scores {oct_full_gb3} against {oct_full_hm} minutes, the same conclusion. The full experiment log is `EXPERIMENT_LOG.md`.

## Computing Platform and Environment

Apple MacBook Pro, M4 Pro (14 CPU cores, no GPU used), 48 GB memory, macOS 27.0; Python 3.12.5, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.1.3, matplotlib 3.10.8 (pinned in `requirements.txt`). The main training run takes ~10 minutes, the November test ~8, post-hoc checks ~5, and the 14-day blind test ~3. The foundation model ran on the same CPU in a separate environment (`chronos-forecasting 2.3.2`, PyTorch 2.15.0; 18,090 Oct/early-Nov forecasts in 71 s; 6,858 blind forecasts in 26 s).

# Results and Discussion

\Needspace{{16\baselineskip}}

## October 2026 test

{table_reg}

**Three hours ahead, every model that uses recent waits beats both original baselines.** The error falls from {hm_mae3} minutes for the historical mean to {ols_mae3} for OLS, {rf_mae3} for the random forest and {gb_mae3} for gradient boosting, {gain_mae3}% below the baseline. OLS with only the calendar and the hospital is *worse* than the historical mean; adding recent waits is what makes a linear model useful, and the tree models gain a few more minutes from interactions. Ridge and lasso match OLS: with {n_train_reg} training rows there is little to regularise.

**Twenty-four hours ahead, no first-round model has a clear or stable advantage over the historical mean.** Gradient boosting looked better on validation ({gb_val24} against {hm_val24}) but not on the test month ({gb_mae24} against {hm_mae24}); the full-feature linear models and the forest end within 0.3 minutes of the baseline. The classifier tells the same story (Table 4 and Figure 4, Supplementary Materials). Logistic regression trained on all years falls below the baseline; trained on the new format only it reaches AUC {logit_b} at 3 hours, so the era flag is enough for trees but not for a linear model.

## What each further experiment taught us

![(a) Change in validation MAE when one feature group is removed and gradient boosting is refitted. (b) The historical mean and four 24-hour boosting variants on September validation.]({fig_dev}){{width=80%}}

- **Ablation (Figure 2a).** Removing recent waits changes the 3-hour validation error by {abl_hist3} minutes, the calendar by {abl_cal3} and the hospital by {abl_hosp3}. Removing weather ({abl_wx3}) or AQHI ({abl_air3}) slightly *improves* the model, so the final model drops both. This answers the secondary question: A&E congestion follows its own momentum and the clock. Dropping weather was not a clear win at 24 hours, though: validation improved by {abl_wx24_abs} minutes but the October error moved from 51.1 to {gb_mae24}, both within noise of the baseline. Two cautions: October weather was not yet published ({wx_missing} of test-month values missing), and the 3-hour validation differences are 0.1–0.3 minutes in one month. "Weather adds nothing" is a finding about September, with daily city-wide data.
- **The 24-hour horizon (Figure 2b).** We tried three ideas: predict the residual from the historical mean, shrink the prediction towards the historical mean, and add slower history. Slow history helped most on validation ({v_slow} vs {v_orig}; residual alone reached 48.6); the best shrinkage weight was close to 1, meaning no shrinkage. On October the slow-history model scored {slow_oct} minutes against {hm_mae24} for the baseline, a gain of about one minute.
- **Tuning the 3-hour model.** A 54-configuration search improved validation by only 0.1 minute ({tuned_val} against {gb_val3}) and gave {tuned_test} on October against {gb_mae3} for the original setting. The model is limited by its inputs, not its hyper-parameters. We saw October before deciding, so we report both settings side by side. The same applies to isotonic calibration (Section 3).

## Live test on 1–10 November 2026

All choices above were fixed by 4 November. On 11 November we downloaded 1–10 November (8 days; the archive has nothing for 6–7 November) and scored the frozen models, trained on data before September, without any change.

{table_nov}

- **The 3-hour result holds.** The error is {nov_gb3} minutes against {nov_hm3} for the baseline, a gain of {nov_d3} minutes (95% interval {nov_ci3}), and the model wins on {w3_live} days (Figure 3, Supplementary Materials). AUC rises from {nov_auc_hm} to {nov_auc_gb3}.
- **At 24 hours both boosting models beat the historical mean, but slow history is not the reason.** The first model already scores {nov_gb24} ({nov_dgb24} below the baseline, 95% interval {nov_cigb24}); slow history gives {nov_slow24} ({nov_d24} below, interval {nov_ci24}) and wins on {w24_live} days. Over the complete October slow history gained only {oct_d24} minutes (interval {oct_ci24}). The historical mean is a ten-month average and November ran below it, which flatters any method that tracks the recent level (Section 5.4).
- **A practical check.** For every pair of hospitals and every November hour where the real waits ended at least 30 minutes apart ({pw_n} cases; a tie counts as half), we asked which hospital each method expected to be quicker three hours later. The shorter wait on the board now is right {pw_now} of the time, the historical mean {pw_hist}, and the model {pw_model} (`src/live_predictions.py`). The one-parameter rule of Section 5.4 is right {pw_rule} of the time, level with the model.

## Stronger baselines, foundation model, and post-freeze blind test

An adversarial AI review of our draft asked whether the ten-month historical mean is too weak a bar. On 11 November, after 1–10 Nov had been scored, we added three simple rules, the proposal's decision tree, boosting with level features, quantile boosting, and zero-shot Chronos-Bolt. Because those Oct (1–31 Oct, 10,242 rows) and 1–10 Nov checks were post-hoc, we committed `PREREGISTRATION.md` on 11 Nov before later data existed, scored the pre-registered 12–15 Nov window first (`ba7ad35`, {prereg_n} rows), and extended the same frozen run through 25 Nov (`7374b68`, 14 days 12–25 Nov, {ext_n} rows; 11 Nov absent; Table 3).

\Needspace{{13\baselineskip}}

{table_post}

- **Three hours.** *Level + anomaly* scores {r3_oct} (Oct), {r3_nov} (1–10 Nov), and {r3_ext} (14-day blind; {r3_prereg} on 12–15 Nov). Gradient boosting beats it by {g3_oct} min in October (CI {g3_oct_ci}, {w3_oct} d), {g3_nov} min in 1–10 Nov (CI {g3_nov_ci}, {w3_nov} d), {g3_prereg} min on 12–15 Nov (CI {g3_prereg_ci}, {w3_prereg} d), {g3_ext10} min on 16–25 Nov (CI {g3_ext10_ci}, {w3_ext10} d), and {g3_ext} min over all 14 blind days (CI {g3_ext_ci}, {w3_ext} d; pairwise {pw_ext_model} vs {pw_ext_rule}, vs {pw_ext_hist} historical mean and {pw_ext_now} board over {pw_ext_n} pairs; clean origins only: {clean_gb3_blind} vs {clean_r3_blind}). Consistent with the 40-min downward drift in mean wait from training ({mean_tr} min, {pos_tr} > 2 h) through Oct ({mean_oct}) and 1–10 Nov ({mean_nov1}) to {mean_ext} min on 12–25 Nov ({pos_ext} > 2 h), historical-mean MAE rises to {ext_hm} min; the rule removes most drift, and boosting adds ~2 min.
- **Twenty-four hours.** The rule scores {r24_oct}, {r24_nov}, and {r24_ext}. Slow-history boosting fails to beat it cleanly in any window ({s24_oct} Oct, CI {s24_oct_ci}; {s24_nov} 1–10 Nov, CI {s24_nov_ci}; {s24_prereg_diff} 12–15 Nov, CI {s24_prereg_ci}; {s24_ext_diff} across 14 blind days, CI {s24_ext_ci}, {w24_ext} d), sealing our negative supervised 24-hour finding.
- **Foundation model vs quantile shift.** Chronos-Bolt \citep{{ansari2024chronos}} sees only each hospital's last 512 hourly waits (no calendar, no training on our data). At 3 hours it matches boosting on 14 blind days ({fm3_ext} vs {gb3_ext}; diff {fm3_vs_gb}, CI {fm3_vs_gb_ci}, {fm3_vs_gb_wins} d) and beats the rule everywhere, while its 10–90% range covers {fmc_oct}, {fmc_nov}, and {fmc3_ext} (widths {fmw3_oct}/{fmw3_nov}/{fmw3_ext} min). Consistent with the downward shift, quantile boosting misses low ({q_below3_ext} below $q_{{10}}$ vs {q_above3_ext} above $q_{{90}}$), dropping coverage from {qc_oct}/{qc_nov} to {qc3_ext} despite almost identical width ({qw3_oct}/{qw3_nov}/{qw3_ext} min). On 12 Nov—whose 432 day-ahead origins fell inside the 11 Nov archive outage—Chronos handles missing recent inputs cleanly ({fm24_nov12} vs GB {gb24_nov12}, slow GB {sgb24_nov12}, rule {r24_nov12}), and even excluding 12 Nov leads the rule by {ex12_fm24_diff} min while slow GB trails by {ex12_sgb24_diff}. Over 14 blind days Chronos edges past the rule ({fm24_ext} vs {r24_ext}; diff {fm24_vs_rule}, CI {fm24_vs_rule_ci}, {fm24_vs_rule_wins} d; coverage {fmc24_ext} over {fmw24_ext} min vs GB {qc24_ext} over {qw24_ext}; Table 5).

## Limitations

- **Coverage, gaps & granularity.** October misses {n_gap_days} days; 1–10 Nov has 8 scored days and 11–25 Nov has 14 (11 Nov absent; some scored rows lack origin snapshots, Section 2.2), and 4-day bootstrap intervals are coarse. Moreover, the target exists since October 2025 (one winter season) and measures HA's published 30-min hospital median rather than true individual waits; a ~40-minute MAE makes calibrated intervals preferable to a point guess.

# Conclusion

**Supported by the evidence.** Hong Kong A&E waits can be forecast 3 hours ahead: across October, early November, and a 14-day post-freeze blind test (11–25 Nov), gradient boosting cuts error 21%–35% below the historical average. Most of that gain comes from recent level and current deviation, which a one-parameter rule captures; boosting adds ~2 min, and zero-shot Chronos-Bolt matches boosting at 3 hours with better-calibrated 10–90% intervals than quantile boosting. Weather and air quality, as measured here, added nothing.

**Not supported.** A day ahead, no supervised model we built beat the one-parameter rule across all windows (losing by 1.0 min over the 14 blind days); only zero-shot Chronos-Bolt finished slightly ahead on 14 blind days ({fm24_ext} vs {r24_ext} min). Our initial 24-hour gains against the historical mean reflected level drift, not forecasting skill.

**Next steps.** Schedule the live-feed collector to eliminate archive gaps; add slower epidemiological signals such as influenza surveillance for the 24-hour horizon; calibrate quantile forecast ranges; and test hourly district rainfall before ruling weather out completely.

\Needspace{{12\baselineskip}}

# Author Contributions {{.unnumbered}}

| Member name | Specific contributions | Evidence or output |
|-------------|------------------------------------------------|--------------------------------|
| CHEN Hongxi | Data sourcing and archive download; parsing and cleaning; feature pipeline; baselines and first models; November live and blind tests; post-hoc checks and Chronos-Bolt baseline (Section 5.4); integration of the report and slides | `download_ae_archive.py`, `parse_ae.py`, `build_features.py`, `train_models.py`, `live_test.py`, `posthoc_checks.py`, `foundation_baseline.py`, `blind_test_nov.py` |
| NI Haoran | Designed and verified 24 h experiments (residual target, slower history, shrinkage); reviewed code and validation/test results | `ni_24h_experiments.py`, `hn_24h_*.csv` |
| ZHU Yongkai | Designed and verified 3 h analyses (54-config tuning, probability calibration, hospital/hour errors); reviewed val vs test | `zhu_3h_analysis.py`, `zy_3h_*.csv` |

# AI Tool Use {{.unnumbered}}

Claude Code (Anthropic) drafted Python scripts, report prose, and slides from our design; Kimi (Moonshot) assisted execution of the 4 Nov scripts designed and checked by NI Haoran and ZHU Yongkai. Antigravity (Gemini), Codex (`gpt-6.2-astra`), and Claude (`claude-opus-5-5`) conducted read-only audits across the 11–25 Nov extension, metrics, and clean-clone build. All numbers were verified via `pytest`, chronological splits, independent re-runs (24 h slow history val {v_slow}, Oct {slow_oct}), `build_final.py`, unmodified Chronos-Bolt, and `PREREGISTRATION.md`.


```{{=latex}}
\Needspace{{8\baselineskip}}
{{\small
\bibliographystyle{{plainnat}}
\bibliography{{references}}
}}
```

\clearpage

# Supplementary Materials {{.unnumbered}}

**Commands to reproduce the results**, run from a clone of the repository (`tag final-2026-11-27`), in this order. The PDF build requires pandoc, a TeX distribution, and `elec3544_report.sty` from the Moodle template placed in `report/template/` (or `report/`). Save `Group16-Project-data.zip` to `data/processed/` (or pass its path to `--zip`) and unzip it there to obtain all three frozen CSVs (`hourly_features_{{full,latest,blind}}.csv`); Chronos-Bolt runs in a separate Python 3.12.5 environment (`venv-chronos`, `requirements-foundation.txt`, pinned revision `5d9f166d`).

```
pip install -r requirements.txt && python3 -m pytest -q tests/
python3 -m venv venv-chronos && venv-chronos/bin/pip install -r requirements-foundation.txt
# Canonical: save & unzip Group16-Project-data.zip in data/processed/ (*_full, *_latest, *_blind.csv)
# Optional archive rebuild from scratch (*_blind.csv only; Oct/Nov frozen cuts use the release ZIP):
#   python3 src/download_ae_archive.py --start 20230101 --end 20261125 --every 4 \
#     --out ~/elec3544_data/raw/ae && python3 src/fetch_aux.py
#   python3 src/parse_ae.py --raw ~/elec3544_data/raw/ae \
#     --out data/processed/ae_snapshots_blind.csv
#   python3 src/build_features.py --ae data/processed/ae_snapshots_blind.csv \
#     --out data/processed/hourly_features_blind.csv
python3 src/train_models.py --out results/models_2026-10-30                       # all features (ablation)
python3 src/train_models.py --drop weather air --out results/models_2026-10-30_v2 # Tables 1 & 4 (*_full.csv)
python3 src/ni_24h_experiments.py && python3 src/zhu_3h_analysis.py               # Fig 2b, 54-config, calib
python3 src/live_test.py && python3 src/live_predictions.py                       # 1-10 Nov (*_latest.csv)
python3 src/posthoc_checks.py && python3 src/posthoc_pairwise.py                  # Sec 5.4 post-hoc (Table 3)
venv-chronos/bin/python src/foundation_baseline.py                                # Chronos-Bolt Oct/early Nov
python3 src/blind_test_nov.py && venv-chronos/bin/python src/blind_foundation_nov.py # 11-25 Nov (*_blind.csv)
python3 src/plot_checkpoint.py && python3 src/plot_report.py
python3 report/build_final.py --official --zip data/processed/Group16-Project-data.zip
```

**November day-by-day error (1–10 Nov live, 12–15 Nov pre-registered, 16–25 Nov extended blind).** Figure 3 shows daily MAE across all 22 available November days, marking archive gaps on 6–7 and 11 Nov; Chronos-Bolt stays competitive even on gap-adjacent days.

![Daily MAE across 22 available November days (gaps marked on 6–7 and 11 Nov); 3-hour boosting beats the rule on most blind days and tracks Chronos closely while 24-hour supervised models track the rule.]({fig_nov}){{width=86%}}

**Where the model fails.** The 3-hour error is largest around midnight (about {err_night} minutes at 00:00–02:00) and smallest in the morning ({err_morning} minutes at 07:00–08:00), following the level of the wait itself. By hospital, the model helps most where waits are long and volatile: {top_gain}. It adds almost nothing at the most stable departments: {low_gain}.

**Classification detail and the full model comparison.** Table 4 gives the October classification results summarised in Section 5.1; Figure 4 shows the October regression MAE and classifier ROC-AUC.

{table_cls}

![October 2026 test results for every model. The dotted line marks the historical-mean baseline at 3 hours.]({fig_models}){{width=86%}}

\Needspace{{10\baselineskip}}

**Pre-registration scorecard (11–15 Nov and extended 11–25 Nov).** Table 5 compares the five expectations fixed in `PREREGISTRATION.md` on 11 November (computed programmatically against both windows). Full 12–15 Nov pre-registered MAEs (3 h / 24 h, n = {prereg_n}): {prereg_all_mae}.

{table_prereg}
