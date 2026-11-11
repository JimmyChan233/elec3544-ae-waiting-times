---
title: "Predicting A&E Waiting Times in Hong Kong Public Hospitals"
subtitle: "\\normalfont\\Large Final project report · Group 16 · 17 November 2026\\vspace{{0.35em}}"
author: "CHEN Hongxi · NI Haoran · ZHU Yongkai"
---

**Abstract.** We forecast how long semi-urgent and non-urgent patients (triage categories 4 and 5) will wait at Hong Kong's {n_hosp} public Accident and Emergency (A&E) departments, 3 hours and 24 hours ahead. The Hospital Authority publishes only the current wait, so we rebuilt {n_rows_latest} hospital-hours of history (1 January 2023 to 10 November 2026) from the government's archive, reconciled a change of feed format in October 2025, and compared five model families with simple baselines. On October 2026, held out from the fitting and selection of the reported 3-hour model, gradient boosting cut the 3-hour error from {hm_mae3} to {gb_mae3} minutes, {gain_mae3}% below the same-slot historical average; on 1–10 November, downloaded after the models were frozen, it scored {nov_gb3} against {nov_hm3}. A check we added after that test changes how the result should be read: a one-parameter rule, the recent level plus a share of the current deviation, already reaches {r3_oct} minutes on the complete October and {r3_nov} in November, so boosting adds about 2 minutes on top of it, and a pretrained time-series foundation model used zero-shot, with no training on our data, comes within {fm_gap} minutes of our trained model. At 24 hours no model reliably beat such a rule. Recent waits carry almost all the signal; weather and air quality added none in our tests.

# 1 Introduction

Almost everyone in Hong Kong has waited at an A&E. Waits for non-urgent patients often exceed several hours and differ widely between hospitals and across the day. The Hospital Authority (HA) publishes the current wait at each A&E every 15 minutes [1], but no forecast. A patient deciding whether to go now, later, or to another hospital needs the wait at the time they will arrive.

**Objective.** Given the hospital, the time, the calendar and recent waits, predict the category 4/5 median wait 3 hours and 24 hours ahead, in minutes (regression) and as "over 2 hours or not" (classification). A secondary question is which group of factors matters most: the clock, the hospital, recent congestion, weather or air quality.

**Contributions.**

- A cleaned dataset of {n_rows_latest} hospital-hours rebuilt from {n_files} archived snapshots, with the two feed formats reconciled and the gaps in the archive documented.
- A forecasting protocol that uses only information available at forecast time, with one validation month for choosing the reported models and two later periods (October and 1–10 November 2026) scored afterwards. Section 4 lists the side experiments that did see October.
- Evidence that the wait 3 hours ahead is predictable (about a fifth less error than the historical average), together with an honest decomposition: most of that gain is available to a one-parameter rule, and gradient boosting adds about 2 minutes more. At 24 hours we report a negative result.
- An ablation showing that congestion follows its own momentum and the clock, not the weather or air quality.

# 2 Data and Preprocessing

## 2.1 Sources and access

**A&E waiting times.** HA's open-data feed [1] is one JSON file covering all {n_hosp} public A&Es. The live file holds only the latest snapshot, so we rebuilt the history through the data.gov.hk historical-archive API [2], which stores past versions of the file. Three problems shaped the downloader (`src/download_ae_archive.py`):

- the version-listing endpoint silently truncates at about 35,000 results, so we query one calendar month at a time;
- it rejects end dates of today or later, so each query is capped at yesterday;
- 96 snapshots a day over almost four years is about 130,000 files, so we kept one snapshot per hour and ran 8 resumable workers.

The result is {n_files} snapshot files ({n_rows_latest} hospital-hours) from 1 January 2023 to 10 November 2026. The archive itself is incomplete: it has **no versions at all** on {n_gap_days} days of October 2026 ({gap_days}) and on 6–7 November.

**Auxiliary data.** Daily weather (rainfall, humidity, mean, maximum and minimum temperature) from the Hong Kong Observatory [3]; the hourly Air Quality Health Index (AQHI) from the Environmental Protection Department [4], matched to the station nearest each hospital; and public holidays [5]. The government's holiday feed starts in 2025, so every 2023–2024 holiday was silently labelled a normal day until we added the official lists by hand ({n_holiday_hours} holiday hospital-hours after the fix, 12,942 before).

## 2.2 Cleaning and feature preparation

- **Text to minutes.** Waits arrive as strings such as "less than 15 minutes" or "2.5 hours". The parser (`src/parse_ae.py`) is unit-tested and left no unparsed value. The values are coarse: every new-format wait in our hourly table is a multiple of 30 minutes, and {q120} of hours sit exactly at 120 minutes, the boundary of the "over 2 hours" label.
- **Two formats.** Before 13 October 2025 each hospital reported one category such as "Over 3 hours"; since then it reports the median and 95th-percentile wait per triage group. The old category is only a lower bound, so the minute-level target exists only from October 2025 ({n_rows_new_latest} hospital-hours). The "over 2 hours" label can be derived from both formats and covers all four years.
- **Exact time steps.** Missing snapshots mean "the previous row" is not always "one hour earlier". We re-index every hospital to a complete hourly grid before computing any lag, so a lag of *k* is exactly *k* hours and missing hours stay missing ({n_missing_pct}% of the grid).
- **No look-ahead.** Features use waits up to the forecast hour, AQHI at that hour, and the *previous* day's weather, because a day's weather summary is not known until the day ends. The calendar of the target hour is known in advance.

## 2.3 What the data shows

![Left: mean category 4/5 wait by hour of day. Right: monthly share of hours with a wait over 2 hours; the dashed line is the format change on 13 October 2025.]({fig_eda}){{width=84%}}

- **A strong daily cycle.** On weekdays the mean wait bottoms out at about {trough_min} minutes at {trough_hr}:00 and peaks at about {peak_min} minutes at {peak_hr}:00. Weekend and holiday evenings are {wkend_gap} minutes shorter at 21:00.
- **Large differences between hospitals**, from {least_busy_min} minutes ({least_busy}) to {busiest_min} minutes ({busiest}) on average.
- **A break at the format change.** In October 2025 itself the old feed put {oct25_old}% of hours above 2 hours and the new feed {oct25_new}%. Part of the drop is the definition (lower bound against median), not the patients. A model trained on the old years without accounting for this over-predicts long waits.

# 3 Methodology

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

**Models.** We followed the progression taught in the course: ordinary least squares (OLS) with calendar and hospital only, OLS with all features, ridge and lasso, random forest [7], and histogram gradient boosting [8] with an absolute-error loss, which matches the MAE metric and is robust to the long right tail of waits. For "over 2 hours" we used logistic regression, random forest and gradient boosting. All models come from scikit-learn [6].

\needspace{{8\baselineskip}}

**Methods added after the live test (results in Section 5.4).** Three rules that use only waits observed up to the forecast hour: the mean of the target's clock hour over the previous 28 days; the historical mean plus a weight times the current deviation from it; and the 28-day mean plus a weight times the current deviation from the 28-day mean. Each weight is a single least-squares coefficient fitted on the training rows. We also fitted one decision tree, quantile gradient boosting for the 10%, 50% and 90% levels, and ran Chronos-Bolt [10], a pretrained time-series foundation model, zero-shot on the last 512 hours of each hospital's own series.

**Main decisions and the alternatives we considered.**

| Decision | Alternative considered | Evidence | Outcome |
|--------------------|--------------------|--------------------------|--------------|
| Historical mean computed on the new format only (classification) | Average over all years | Old feed has {pos_old}% positives against {pos_new}% now; baseline accuracy on validation rose from {hm_acc_old} to {hm_acc_val} | Kept: a fairer bar |
| Train the classifier on all years with an era flag | Train on the new format only | Test AUC {auc_A3} against {auc_B3} at 3 h | Equal for boosting; era flag kept |
| Gradient boosting as the main model | Linear models, random forest | 3 h test MAE {gb_mae3} against {ols_mae3} (OLS) and {rf_mae3} (forest) | Kept |
| Regularised linear models | Plain OLS | Validation MAE flat across every ridge and lasso strength | Not a lever |
| Drop weather and AQHI | Keep all features | Removing either lowers the 3 h validation error (Figure 3a); October weather was not yet published | Dropped |
| Slow history for 24 h | Residual target; shrinking towards the baseline | Validation MAE {v_slow} against {v_orig} (Figure 3b) | Kept on validation; no better than a simple rule (5.4) |
| Default 3-point grid for the 3 h model | 54-configuration search | Validation {tuned_val} against {gb_val3}, test {tuned_test} against {gb_mae3} | No gain; original reported |
| Judge models against one-parameter rules | Historical mean as the only bar | Best rule scores {r3_oct} against {post_hm_oct} at 3 h (Table 4) | Conclusions rewritten |
| Uncalibrated classifier | Isotonic calibration [9] | Slightly better on September in sample; October Brier {brier_iso} against {brier_raw} without it | Not adopted (decided on October) |

# 4 Experiments and Reproducibility

**Splits.** Rows are split by the time of the *target*, so no training target lies in a later period.

| Set | Target hours | Use | Regression rows | Classification rows |
|-----------|-------------------|--------------------------------|-----------|-----------|
| Train | to 31 Aug 2026 | fitting | {n_train_reg} | {n_train_cls} |
| Validation | September 2026 | every modelling choice | {n_val} | {n_val} |
| Test | 1–29 Oct 2026 | scored after the 30 Oct choices | {n_test} | {n_test} |
| Live test | 1–10 Nov 2026 | downloaded on 11 Nov, scored once | {nov_n} | {nov_n} |

**Leakage controls.** (i) Chronological split by target time. (ii) Lags on an exact hourly grid. (iii) Previous-day weather only. (iv) The historical mean is computed from training rows only; as a feature it is also used on those same training rows, a mild form of target encoding that cannot affect the validation or test scores. (v) Hyper-parameters, feature groups and the 24 h variant were chosen on September; October and November were scored afterwards. One caveat: the 24 h slow-history variant and the 54-configuration search were run on 4 November, after we had seen the first October results, so for those two experiments October is not strictly blind, and November is the clean test.

**Metrics.** Regression: mean absolute error (MAE, minutes) and R²; RMSE is in the result files. Classification: accuracy, precision and recall at a 0.5 threshold, ROC-AUC and the Brier score. For differences between methods we report a 95% bootstrap interval that resamples whole days (2,000 resamples), because hours within a day are correlated. With only 8 days in November these intervals are likely too narrow, so we also count the days each method wins.

**Differences from the proposal.** October became a held-out test month instead of a second validation month. The final test covers 1–10 November, not 1–15, because the report was completed on 11 November. The single decision tree is reported in Section 5.4. We report precision and recall instead of a full confusion matrix, and a drop-one-group ablation instead of tree feature importance. The live-feed collector we planned was written but never scheduled, so the tests rely on the archive.

**Hyper-parameters and seeds.** Gradient boosting: 400 iterations, no early stopping, learning rate 0.05, 63 leaves for regression (chosen from three settings on validation) and 31 for classification. Random forest: 150 trees, minimum leaf 20, half of the features per split, fitted on at most 150,000 sampled rows. Ridge and lasso strengths chosen on validation from 0.1–100 and 0.01–1. Logistic regression: C = 1. Decision tree: minimum leaf 50, depth chosen on validation from 6, 10 and 14 ({dt3} at 3 hours, {dt24} at 24). Rule weights: {b3} at 3 hours, {b24} at 24. Quantile boosting uses the regression settings. Every random seed is 0.

**Data and code versions.** Raw data: archive versions of the two HA feed URLs [1, 2], one per hour, 1 January 2023 to 10 November 2026. The October results use the table frozen on 30 October (`hourly_features_full.csv`, {n_rows} rows); the November test uses the table rebuilt on 11 November (`hourly_features_latest.csv`, {n_rows_latest} rows). Code, result files and the experiment log are public at <https://github.com/JimmyChan233/elec3544-ae-waiting-times>, commit `2eb14ae` (tag `freeze-2026-11-11`). The processed tables are the release asset `Group16-Project-data.zip` (SHA-256 `{zip_sha}…`).

**Platform.** Apple MacBook Pro, M4 Pro (14 CPU cores, no GPU used), 48 GB memory, macOS 27.0; Python 3.12.5, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.1.3, matplotlib 3.10.8 (pinned in `requirements.txt`). The main training run takes about 10 minutes, the November test about 8 and the post-hoc checks about 5. The foundation model ran on the same CPU in a separate environment (chronos-forecasting 2.3.2, PyTorch 2.15.0; 18,090 forecasts in 71 seconds).

**Reproducing the results.** From a clone of the repository, these commands rebuild the data and every number, table and figure in this report, in order (the table frozen on 30 October, `hourly_features_full.csv`, is in the release asset):

```
pip install -r requirements.txt && python3 -m pytest -q tests/
# data: unzip the release asset Group16-Project-data.zip into data/processed/, or rebuild it:
python3 src/download_ae_archive.py --start 20230101 --end 20261110 --every 4 --out ~/elec3544_data/raw/ae
python3 src/fetch_aux.py
python3 src/parse_ae.py --raw ~/elec3544_data/raw/ae --out data/processed/ae_snapshots_latest.csv
python3 src/build_features.py --ae data/processed/ae_snapshots_latest.csv \
        --out data/processed/hourly_features_latest.csv
python3 src/train_models.py --out results/models_2026-10-30               # all features (ablation)
python3 src/train_models.py --drop weather air --out results/models_2026-10-30_v2   # Tables 1-2
python3 src/ni_24h_experiments.py       # 24 h variants (Figure 3b)
python3 src/zhu_3h_analysis.py          # 54-configuration search, calibration
python3 src/live_test.py                # November live test (Table 3, Figure 4)
python3 src/live_predictions.py         # per-hour November forecasts, pairwise check
python3 src/posthoc_checks.py && python3 src/posthoc_pairwise.py   # Section 5.4 (Table 4)
python3 src/foundation_baseline.py      # Chronos-Bolt, separate env: requirements-foundation.txt
python3 src/plot_report.py && python3 report/build_final.py   # PDF: needs pandoc and XeLaTeX (TeX Live)
```

The October runs read the table frozen on 30 October (`hourly_features_full.csv`, data to 29 October). Rebuilding from the archive today also returns 30–31 October; on the complete month the frozen 3-hour model scores {oct_full_gb3} against {oct_full_hm} minutes, the same conclusion. The full experiment log is `EXPERIMENT_LOG.md` in the repository.

# 5 Results and Discussion

## 5.1 October 2026 test

{table_reg}

![October 2026 test results for every model. The dotted line marks the historical-mean baseline at 3 hours.]({fig_models}){{width=86%}}

**Three hours ahead, every model that uses recent waits beats both original baselines.** The error falls from {hm_mae3} minutes for the historical mean to {ols_mae3} for OLS, {rf_mae3} for the random forest and {gb_mae3} for gradient boosting, {gain_mae3}% below the baseline. OLS with only the calendar and the hospital is *worse* than the historical mean; adding recent waits is what makes a linear model useful, and the tree models gain a few more minutes from interactions. Ridge and lasso match OLS: with {n_train_reg} training rows there is little to regularise.

**Twenty-four hours ahead, no first-round model has a clear or stable advantage over the historical mean.** Gradient boosting looked better on validation ({gb_val24} against {hm_val24}) but not on the test month ({gb_mae24} against {hm_mae24}); the linear models and the forest end within 0.3 minutes of the baseline. The classifier tells the same story. Logistic regression trained on all years falls below the baseline; trained on the new format only it reaches AUC {logit_b} at 3 hours, so the era flag is enough for trees but not for a linear model.

{table_cls}

## 5.2 What each further experiment taught us

![(a) Change in validation MAE when one feature group is removed and gradient boosting is refitted. (b) The four 24-hour variants on September validation.]({fig_dev}){{width=86%}}

- **Ablation (Figure 3a).** Removing recent waits changes the 3-hour validation error by {abl_hist3} minutes, the calendar by {abl_cal3} and the hospital by {abl_hosp3}. Removing weather ({abl_wx3}) or AQHI ({abl_air3}) slightly *improves* the model, so the final model drops both. This answers the secondary question: A&E congestion follows its own momentum and the clock. Dropping weather was not a clear win at 24 hours, though: validation improved by {abl_wx24_abs} minutes but the October error moved from 51.1 to {gb_mae24}, both within noise of the baseline. Two cautions: October weather was not yet published ({wx_missing} of test-month values missing), and the validation differences are 0.1–0.3 minutes in one month. "Weather adds nothing" is a finding about September, with daily city-wide data.
- **The 24-hour horizon (Figure 3b).** We tried three ideas: predict the residual from the historical mean, shrink the prediction towards the historical mean, and add slower history. Only the last helped on validation ({v_slow} against {v_orig}); the best shrinkage weight was close to 1, meaning no shrinkage. On October the slow-history model scored {slow_oct} minutes against {hm_mae24} for the baseline, a gain of about one minute.
- **Tuning the 3-hour model.** A 54-configuration search improved validation by only 0.1 minute ({tuned_val} against {gb_val3}) and gave {tuned_test} on October against {gb_mae3} for the original setting. The model is limited by its inputs, not its hyper-parameters. We saw October before deciding, so we report both settings side by side. The same applies to isotonic calibration (Section 3).

## 5.3 Live test on 1–10 November 2026

All choices above were fixed by 4 November. On 11 November we downloaded 1–10 November (8 days; the archive has nothing for 6–7 November) and scored the frozen models, trained on data before September, without any change.

{table_nov}

![Error on each November day for the frozen models against the historical-mean baseline.]({fig_nov}){{width=82%}}

- **The 3-hour result holds.** The error is {nov_gb3} minutes against {nov_hm3} for the baseline, a gain of {nov_d3} minutes (95% interval {nov_ci3}), and the model wins on {w3_live} days. AUC rises from {nov_auc_hm} to {nov_auc_gb3}.
- **At 24 hours both boosting models beat the historical mean, but slow history is not the reason.** The first model already scores {nov_gb24} ({nov_dgb24} below the baseline, 95% interval {nov_cigb24}); slow history gives {nov_slow24} ({nov_d24} below, interval {nov_ci24}) and wins on {w24_live} days. Over the complete October slow history gained only {oct_d24} minutes (interval {oct_ci24}). The historical mean is a ten-month average and November ran below it, which flatters any method that tracks the recent level (Section 5.4).
- **A practical check.** For every pair of hospitals and every November hour where the real waits ended at least 30 minutes apart ({pw_n} cases), we asked which hospital each method expected to be quicker three hours later. The shorter wait on the board now is right {pw_now} of the time, the historical mean {pw_hist}, and the model {pw_model} (`src/live_predictions.py`). The one-parameter rule of Section 5.4 is right {pw_rule} of the time, level with the model.

## 5.4 Were the baselines too easy? (added after the live test)

An independent review of our draft asked whether the historical mean, a ten-month average, is too weak a bar. On 11 November, after November had been scored, we added three simple rules, each using only waits observed up to the forecast hour and at most one weight fitted on the training rows, the decision tree promised in the proposal, a boosting model given the same level information, and a pretrained foundation model. Nothing in this section is a blind test.

{table_post}

- **Three hours.** The best rule is *level + anomaly*: the 28-day mean for the target's clock hour, plus {b3} of the gap between the wait now and the 28-day mean for the current hour. It scores {r3_oct} and {r3_nov} minutes. Gradient boosting is {g3_oct} minutes better in October (interval {g3_oct_ci}) and {g3_nov} better in November (interval {g3_nov_ci}), ahead on {w3_oct} October days and {w3_nov} November days. So of the 11–14 minutes gained over the historical mean, about four fifths is information a single weight can use; the model adds a small, consistent remainder.
- **Twenty-four hours.** The same rule scores {r24_oct} and {r24_nov}. Slow-history boosting is within noise of it ({s24_oct} in October, interval {s24_oct_ci}; {s24_nov} in November, interval {s24_nov_ci}), and the first boosting model is worse. Our conclusion for this horizon is negative: no model we built reliably beats a one-parameter rule. Giving boosting the same level features changes little at either horizon.
- **A foundation model, zero-shot.** Chronos-Bolt [10] saw only the last 512 hourly waits of one hospital: no training on our data, no calendar, no hospital identity. At 3 hours it scores {fm3_oct} and {fm3_nov} minutes, better than every simple rule and {fm_gap} minutes behind our trained model; it is ahead of our model on {fm_days} November days. Its 10–90% range contained {fmc_oct} and {fmc_nov} of outcomes, closer to the nominal 80% than ours. At 24 hours ({fm24_oct} and {fm24_nov}) it is worse than the best rule. It recovers most of what our pipeline achieved at 3 hours with no feature engineering.
- **A range instead of a point.** Quantile gradient boosting (10%, 50%, 90%) gives a 3-hour range about {qw} minutes wide that contained {qc_oct} of October and {qc_nov} of November outcomes, against a nominal 80%: slightly too narrow, but more honest than a single number with a 40-minute error.

## 5.5 Where the model fails

The 3-hour error is largest around midnight (about {err_night} minutes at 00:00–02:00) and smallest in the morning ({err_morning} minutes at 07:00–08:00), following the level of the wait itself. By hospital, the model helps most where waits are long and volatile: {top_gain}. It adds almost nothing at the most stable departments: {low_gain}.

## 5.6 Limitations

- **Short test periods.** October is one month with {n_gap_days} days missing, and November is 8 days. The 3-hour gain over the historical mean is clear; the 2-minute gain over the best simple rule is consistent across days but small, and Section 5.4 was added after both test periods had been seen.
- **One year of minute-level data.** The target exists only since October 2025, so seasonal effects such as the winter influenza peak were seen once.
- **An error of 40 minutes is still large** against a typical wait of two to three hours, and the target moves in 30-minute steps. A range (Section 5.4) is the better way to present the forecast.

# 6 Conclusion

**Supported by the evidence.** A&E waits in Hong Kong can be forecast a few hours ahead: on two periods held out from development, the 3-hour error is about a fifth to a quarter below the historical average. Most of that comes from two pieces of information, the recent level and the current deviation from it, which a one-parameter rule already uses; gradient boosting adds about 2 minutes more, consistently, and a zero-shot foundation model lands between the two. Weather and air quality, as we measured them, added nothing.

**Not supported.** A day ahead, no model reliably beat a simple rolling rule. The 24-hour gains we first reported against the historical mean reflect a drifting level, not skill.

**Next steps.** Schedule the live-feed collector to cover archive gaps and extend the test period; score the blind test on 11–15 November that we pre-registered on 11 November (`PREREGISTRATION.md` in the repository); add slower signals such as influenza surveillance for the 24-hour horizon; calibrate the forecast range; and try hourly district rainfall before ruling weather out.

\needspace{{24\baselineskip}}

# Contributions

| Member | Specific work | Evidence / outputs |
|-------------|------------------------------------------------|--------------------------------|
| CHEN Hongxi | Data sourcing and archive download; parsing and cleaning; feature pipeline; baselines and first models; November live test; post-hoc baseline checks and the foundation-model baseline (Section 5.4); integration of the report and slides | `download_ae_archive.py`, `parse_ae.py`, `build_features.py`, `train_models.py`, `live_test.py`, `posthoc_checks.py`, `foundation_baseline.py` |
| NI Haoran | 24-hour forecasting experiments: residual-target modelling, slower-history features and prediction shrinkage. Reviewed the experiment code and the validation and test outputs, compared the selected approach with the historical-mean baseline, and examined the limits of the observed improvement | `ni_24h_experiments.py`, `hn_24h_*.csv` |
| ZHU Yongkai | 3-hour model analysis: hyper-parameter tuning, probability calibration and error comparisons across hospitals and hours. Reviewed the analysis code and outputs, compared validation and test performance, and examined why tuning and calibration did not consistently improve generalisation | `zhu_3h_analysis.py`, `zy_3h_*.csv` |

# Use of AI tools

We used Claude Code (Anthropic) throughout the project. It wrote most of the Python code for downloading, parsing, feature building, model training and the November test, and produced first drafts of this report and of the slides, working from our questions, instructions and decisions. Kimi was used to run the two follow-up experiments of 4 November (the 24-hour variants and the 3-hour search and calibration). We verified the outputs as follows:

- unit tests for the wait-string parser, and a check that no value was left unparsed;
- snapshot counts per day against the expected 24, which is how we found the empty archive days;
- a chronological protocol in which test periods were scored only after the choices were fixed, and a November test on data downloaded only after the models were frozen;
- an independent re-run of the 24-hour slow-history experiment with the main code, which reproduced its numbers (validation {v_slow}, October {slow_oct});
- the tables and nearly all numbers in the text are filled in by a script from the result files; the few typed by hand were re-checked against those files;
- the pretrained Chronos-Bolt model [10] is used only as a baseline (Section 5.4), unmodified, scored on the same rows as every other method;
- a second AI agent reviewed the finished draft as a strict marker. It spot-checked the numbers, audited the code for leakage (none found) and argued that our baselines were too weak. We tested that claim ourselves in `src/posthoc_checks.py`, found it correct, and rewrote our conclusions (Section 5.4).

# References

1. Hospital Authority. *Accident and Emergency Waiting Time* (open data, JSON). <https://www.ha.org.hk/opendata/aed/aedwtdata2-en.json> (from 13 Oct 2025) and `aedwtdata-en.json` (before). Accessed 8 Oct – 11 Nov 2026.
2. Digital Policy Office, HKSAR Government. *DATA.GOV.HK Historical Archive API.* <https://api.data.gov.hk/v1/historical-archive>. Accessed 8 Oct – 11 Nov 2026.
3. Hong Kong Observatory. *Daily climatological data, HKO headquarters* (rainfall, humidity, temperature). <https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/>. Accessed 4 Nov 2026.
4. Environmental Protection Department. *Past Air Quality Health Index records.* <https://www.aqhi.gov.hk/epd/ddata/html/history/>. Accessed 4 Nov 2026.
5. HKSAR Government. *General holidays* (1823 iCal feed for 2025–2026; gov.hk lists for 2023–2024). Accessed 30 Oct 2026.
6. Pedregosa, F. et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* 12, 2825–2830 (2011). Version 1.8.0 used.
7. Breiman, L. Random forests. *Machine Learning* 45, 5–32 (2001).
8. Friedman, J. H. Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29, 1189–1232 (2001).
9. Zadrozny, B. & Elkan, C. Transforming classifier scores into accurate multiclass probability estimates. *Proc. KDD* (2002).
10. Ansari, A. F. et al. Chronos: learning the language of time series. *Transactions on Machine Learning Research* (2024). Model used: `amazon/chronos-bolt-base` from Hugging Face, unmodified.
