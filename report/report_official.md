\begin{{abstract}}
We forecast how long semi-urgent and non-urgent patients (triage categories 4 and 5) will wait at Hong Kong's {n_hosp} public Accident and Emergency (A&E) departments, 3 hours and 24 hours ahead. The Hospital Authority publishes only the current wait, so we rebuilt {n_rows_blind} hospital-hours of history (1 January 2023 to 25 November 2026) from the government's archive, reconciled a feed format change in October 2025, and compared five model families with simple baselines. On October 2026, held out from fitting and selection of the 3-hour model, gradient boosting cut the 3-hour error from {hm_mae3} to {gb_mae3} minutes ({gain_mae3}% below the same-slot historical average); on 1–10 November, downloaded after the models were frozen, it scored {nov_gb3} against {nov_hm3}. A post-hoc check changed how that gain should be read: a one-parameter rule (recent 28-day level plus a share of the current deviation) already reaches {r3_oct} minutes on complete October and {r3_nov} in early November, so boosting adds about 2 minutes on top of it, and a pretrained time-series foundation model used zero-shot lands within {fm_gap} minutes. Tested blindly after model freeze on 11–25 November (14 days, including the pre-registered 11–15 November window), boosting scores {gb3_ext} against {r3_ext} for the rule and {ext_hm} for the historical mean, zero-shot Chronos-Bolt matches boosting ({fm3_ext}) with calibrated interval coverage ({fmc3_ext} vs {qc3_ext}), and at 24 hours no model beats the one-parameter rule. Recent waits carry almost all the signal; weather and air quality added none in our tests.
\end{{abstract}}


# Introduction

Almost everyone in Hong Kong has waited at an A&E. Waits for non-urgent patients often exceed several hours and differ widely between hospitals and across the day. The Hospital Authority (HA) publishes the current wait at each A&E every 15 minutes \citep{{ha_feed}}, but no forecast. A patient deciding whether to go now, later, or to another hospital needs the wait at the time they will arrive.

**Objective.** Given the hospital, the time, the calendar and recent waits, predict the category 4/5 median wait 3 hours and 24 hours ahead, in minutes (regression) and as "over 2 hours or not" (classification). A secondary question is which group of factors matters most: the clock, the hospital, recent congestion, weather or air quality.

**Contributions.** (i) A cleaned dataset of {n_rows_blind} hospital-hours rebuilt from {n_files} archived snapshots, with two feed formats reconciled and archive gaps documented. (ii) A chronological protocol using only information available at forecast time, with one validation month and three test windows, culminating in a 14-day post-freeze blind test (11–25 November, preserving the pre-registered 11–15 November subset). (iii) Evidence that the 3-hour wait is predictable, with an honest decomposition: most of the gain is available to a one-parameter rule, gradient boosting adds a consistent margin, and zero-shot Chronos-Bolt matches the trained model under distribution shift; at 24 hours we report a negative result. (iv) An ablation showing congestion follows its own momentum and the clock, not weather or air quality.

# Data and Preprocessing

## Sources and access

**A&E waiting times.** HA's open-data feed \citep{{ha_feed}} is one JSON file covering all {n_hosp} public A&Es. The live file holds only the latest snapshot, so we rebuilt the history through the data.gov.hk historical-archive API \citep{{datagov_archive}}, which stores past versions of the file. Three problems shaped the downloader (`src/download_ae_archive.py`):

- the version-listing endpoint silently truncates at about 35,000 results, so we query one calendar month at a time;
- it rejects end dates of today or later, so each query is capped at yesterday;
- 96 snapshots a day over nearly four years is ~130,000 files, so we kept one snapshot per hour and ran 8 resumable workers.

The result is {n_files} snapshot files aggregated into {n_hours_blind} hourly slots ({n_rows_blind} hospital-hours) from 1 January 2023 to 25 November 2026 (up to 15 duplicate/quarter-hour boundary files collapse within the same hour). The archive itself is incomplete: it has **no versions at all** on {n_gap_days} days of October 2026 ({gap_days}), 6–7 November, 11 November, and 12 days from 29 August to 9 September 2025.

**Auxiliary data.** Daily weather (rainfall, humidity, mean, maximum and minimum temperature) from the Hong Kong Observatory \citep{{hko_daily}}; the hourly Air Quality Health Index (AQHI) from the Environmental Protection Department \citep{{epd_aqhi}}, matched to the station nearest each hospital; and public holidays \citep{{hk_holidays}}. The government's holiday feed starts in 2025, so every 2023–2024 holiday was silently labelled a normal day until we added the official lists by hand ({n_holiday_hours} holiday hospital-hours after the fix, 12,942 before).

## Cleaning and feature preparation

- **Text to minutes.** Waits arrive as strings such as "less than 15 minutes" or "2.5 hours". The parser (`src/parse_ae.py`) is unit-tested and left no unparsed value. The values are coarse: every new-format wait in our hourly table is a multiple of 30 minutes, and {q120} of hours sit at exactly 120 minutes.
- **Two formats.** Before 13 October 2025 each hospital reported one category such as "Over 3 hours"; since then it reports the median and 95th-percentile wait per triage group. The old category is only a lower bound, so the minute-level target exists only from October 2025 ({n_rows_new_blind} hospital-hours). The "over 2 hours" label covers all four years, defined as strictly `> 120` minutes in the new format (so exact `120`-minute rows are labelled `0`) versus `"Over 2 hours"` and above in the old format.
- **Exact time steps.** Missing snapshots mean "the previous row" is not always "one hour earlier". We re-index every hospital to a complete hourly grid before computing any lag, so a lag of *k* is exactly *k* hours and missing hours stay missing ({n_missing_pct_blind}% of the full grid).
- **No look-ahead.** Features use waits up to the forecast hour, AQHI at that hour, and the *previous* day's weather, because a day's weather summary is not known until the day ends. The calendar of the target hour is known in advance.

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
| Historical mean computed on the new format only (classification) | Average over all years | Old feed has {pos_old}% positives against {pos_new}% now; baseline accuracy on validation rose from {hm_acc_old} to {hm_acc_val} | Kept: a fairer bar |
| Train the classifier on all years with an era flag | Train on the new format only | Test AUC {auc_A3} against {auc_B3} at 3 h | Equal for boosting; era flag kept |
| Gradient boosting as the main model | Linear models, random forest | 3 h test MAE {gb_mae3} against {ols_mae3} (OLS) and {rf_mae3} (forest) | Kept |
| Drop weather and AQHI | Keep all features | Removing either lowers the 3 h validation error (Figure 2a); October weather was not yet published | Dropped |
| Slow history for 24 h | Residual target; shrinking towards the baseline | Validation MAE {v_slow} against {v_orig} (Figure 2b) | Kept on validation; no better than a simple rule (5.4) |
| Default 3-point grid for the 3 h model | 54-configuration search | Validation {tuned_val} against {gb_val3}, test {tuned_test} against {gb_mae3} | No gain; original reported |
| Judge models against one-parameter rules | Historical mean as the only bar | Best rule scores {r3_oct} against {post_hm_oct} at 3 h (Table 3) | Conclusions rewritten |
| Uncalibrated classifier | Isotonic calibration \citep{{zadrozny2002}} | Slightly better on September in sample; October Brier {brier_iso} against {brier_raw} without it | Not adopted (decided on October) |

\Needspace{{15\baselineskip}}

# Experiments and Reproducibility

## Experimental Protocol

**Splits.** Rows are split by the time of the *target*, so no training target lies in a later period (`H=24` classification uses 570,672 training rows; complete October 1–31 has 10,242 rows).

| Set | Target hours | Use | Regression rows | Classification rows |
|-----------|-------------------|--------------------------------|----------:|----------:|
| Train | to 31 Aug 2026 | fitting | {n_train_reg} | {n_train_cls} |
| Validation | September 2026 | choices for the reported models | {n_val} | {n_val} |
| Test | 1–29 Oct 2026 | scored after the 30 Oct choices | {n_test} | {n_test} |
| Live test | 1–10 Nov 2026 | downloaded on 11 Nov, scored once | {nov_n} | {nov_n} |
| Blind test | 11–25 Nov 2026 | post-freeze blind (4 d pre-reg `{prereg_n}` + 10 d ext.) | {ext_n} | – |

**Leakage controls.** (i) Chronological split by target time. (ii) Lags on an exact hourly grid. (iii) Previous-day weather only. (iv) The historical mean is computed from training rows only; as a feature it is also used on those same training rows, a mild form of target encoding that cannot affect validation or test scores. (v) Hyper-parameters, feature groups and the 24 h variant were chosen on September; October and 1–10 November were scored afterwards (except the 4 November 24 h slow-history and 54-config checks, for which October was not strictly blind). Section 5.4 was added after 1–10 November, so we pre-registered `PREREGISTRATION.md` on 11 November and ran the post-freeze blind test (`11–15 Nov` pre-registered, extended to `11–25 Nov`) strictly once with zero refitting or re-tuning.

**Metrics.** Regression: mean absolute error (MAE, minutes) and R²; RMSE is in the result files. Classification: accuracy, precision and recall at a 0.5 threshold, ROC-AUC and the Brier score. For differences between methods we report a 95% bootstrap interval that resamples whole days (2,000 resamples), because hours within a day are correlated. With 8 days in early November and 4 in `11–15 Nov`, intervals are narrow, so we also count days won and extend the blind window to 14 days (`11–25 Nov`).

**Hyper-parameters and seeds.** Gradient boosting: 400 iterations, no early stopping, learning rate 0.05, 63 leaves for regression (chosen from three settings on validation) and 31 for classification. Random forest: 150 trees, minimum leaf 20, half of the features per split, fitted on at most 150,000 sampled rows. Ridge and lasso strengths chosen on validation from 0.1–100 and 0.01–1. Logistic regression: C = 1. Decision tree: minimum leaf 50, depth chosen on validation from 6, 10 and 14 ({dt3} at 3 hours, {dt24} at 24). Rule weights: *level + anomaly* {b3} at 3 hours and {b24} at 24; *historical mean + damped anomaly* {b1_3} and {b1_24}. Quantile boosting uses the regression settings. Every random seed is 0.

**Differences from the proposal.** October became a held-out test month instead of a second validation month. The 1–15 November test window from the proposal became the 1–10 Nov live test plus the 11–15 Nov pre-registered blind test, then extended to 25 Nov (22 November days total). The decision tree is in Section 5.4. We report precision and recall instead of a confusion matrix, and a drop-one-group ablation instead of tree feature importance. The planned live-feed collector was written but never scheduled, so the tests rely on the archive.

## Data and Code for Reproduction

**Data and code versions.** Raw data: archive versions of the two HA feed URLs \citep{{ha_feed,datagov_archive}}, one per hour, 1 January 2023 to 25 November 2026. The 30 October runs read `hourly_features_full.csv` ({n_rows} rows to 29 Oct); the 11 November live test reads `hourly_features_latest.csv` ({n_rows_latest} rows to 10 Nov); and the blind test reads `hourly_features_blind.csv` ({n_rows_blind} rows to 25 Nov). Code, result files and the experiment log are public at <https://github.com/JimmyChan233/elec3544-ae-waiting-times>: `commit 2eb14ae` (`tag freeze-2026-11-11`) records the frozen models and pre-registration before 11–25 Nov data existed, and `tag final-2026-11-26` reproduces all updated figures, the 14-day blind test, and this report. All three processed CSVs are in release `data-2026-11-26` (`Group16-Project-data.zip`, SHA-256 `{zip_sha}…`).

**Reproducing the results.** Clone the repository at `final-2026-11-26`, install `requirements.txt`, and unzip the data release into `data/processed/` (or rebuild from the archive). Running the scripts in Supplementary Materials regenerates every number, table and figure, and `python3 report/build_final.py --official` compiles this PDF. On complete October (including 30–31 Oct), the frozen 3-hour model scores {oct_full_gb3} against {oct_full_hm} minutes, the same conclusion. The full experiment log is `EXPERIMENT_LOG.md`.

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
- **The 24-hour horizon (Figure 2b).** We tried three ideas: predict the residual from the historical mean, shrink the prediction towards the historical mean, and add slower history. Only the last helped on validation ({v_slow} against {v_orig}); the best shrinkage weight was close to 1, meaning no shrinkage. On October the slow-history model scored {slow_oct} minutes against {hm_mae24} for the baseline, a gain of about one minute.
- **Tuning the 3-hour model.** A 54-configuration search improved validation by only 0.1 minute ({tuned_val} against {gb_val3}) and gave {tuned_test} on October against {gb_mae3} for the original setting. The model is limited by its inputs, not its hyper-parameters. We saw October before deciding, so we report both settings side by side. The same applies to isotonic calibration (Section 3).

## Live test on 1–10 November 2026

All choices above were fixed by 4 November. On 11 November we downloaded 1–10 November (8 days; the archive has nothing for 6–7 November) and scored the frozen models, trained on data before September, without any change.

{table_nov}

- **The 3-hour result holds.** The error is {nov_gb3} minutes against {nov_hm3} for the baseline, a gain of {nov_d3} minutes (95% interval {nov_ci3}), and the model wins on {w3_live} days (Figure 3, Supplementary Materials). AUC rises from {nov_auc_hm} to {nov_auc_gb3}.
- **At 24 hours both boosting models beat the historical mean, but slow history is not the reason.** The first model already scores {nov_gb24} ({nov_dgb24} below the baseline, 95% interval {nov_cigb24}); slow history gives {nov_slow24} ({nov_d24} below, interval {nov_ci24}) and wins on {w24_live} days. Over the complete October slow history gained only {oct_d24} minutes (interval {oct_ci24}). The historical mean is a ten-month average and November ran below it, which flatters any method that tracks the recent level (Section 5.4).
- **A practical check.** For every pair of hospitals and every November hour where the real waits ended at least 30 minutes apart ({pw_n} cases; a tie counts as half), we asked which hospital each method expected to be quicker three hours later. The shorter wait on the board now is right {pw_now} of the time, the historical mean {pw_hist}, and the model {pw_model} (`src/live_predictions.py`). The one-parameter rule of Section 5.4 is right {pw_rule} of the time, level with the model.

## Stronger baselines, foundation model, and post-freeze blind test

An independent review of our draft asked whether the historical mean, a ten-month average, is too weak a bar. On 11 November, after 1–10 November had been scored, we added three simple rules, the proposal's decision tree, boosting with level features, quantile boosting, and zero-shot Chronos-Bolt. Because those Oct/1–10 Nov comparisons were post-hoc, we committed `PREREGISTRATION.md` on 11 November before later data existed, and scored the frozen methods once on 11–25 November: 14 days with data (`12–25 Nov`, {ext_n} rows; 11 Nov missing), reporting both the exact pre-registered `12–15 Nov` subset ({prereg_n} rows) and the full 14-day blind window. Table 3 compares complete October (1–31 Oct, 10,242 rows, unlike 1–29 Oct in Table 1), 1–10 November, and 12–25 November.

\Needspace{{13\baselineskip}}

{table_post}

- **Three hours.** *Level + anomaly* scores {r3_oct} (Oct), {r3_nov} (1–10 Nov), and {r3_ext} (14-day blind; {r3_prereg} on 12–15 Nov). Gradient boosting beats it by {g3_oct} min in October (CI {g3_oct_ci}, {w3_oct} d), {g3_nov} min in 1–10 Nov (CI {g3_nov_ci}, {w3_nov} d), {g3_prereg} min on pre-registered 12–15 Nov (CI {g3_prereg_ci}, {w3_prereg} d), and {g3_ext} min across all 14 blind days (CI {g3_ext_ci}, {w3_ext} d; {pw_ext_model} vs {pw_ext_rule} pairwise accuracy against {pw_ext_now} on the live board over {pw_ext_n} pairs). Of the large drop below the static historical mean ({ext_hm} min under November's shift), most is captured by one parameter, while boosting adds a consistent 2–4 min margin.
- **Twenty-four hours.** The same rule scores {r24_oct}, {r24_nov}, and {r24_ext}. Slow-history boosting fails to outperform it cleanly in any window ({s24_oct} in Oct, CI {s24_oct_ci}; {s24_nov} in 1–10 Nov, CI {s24_nov_ci}; {s24_prereg_diff} on 12–15 Nov, CI {s24_prereg_ci}; and {s24_ext_diff} across 14 blind days, CI {s24_ext_ci}, winning only {w24_ext} days), sealing our negative 24-hour conclusion.
- **A foundation model, zero-shot.** Chronos-Bolt \citep{{ansari2024chronos}} saw only the last 512 hourly waits of one hospital: no training on our data, no calendar, no hospital identity. At 3 hours it scores {fm3_oct}, {fm3_nov}, and {fm3_ext} minutes ({fm3_prereg} on 12–15 Nov)—matching our frozen boosting model across the 14 blind days and beating the rule in every window—and its 10–90% range achieves {fmc_oct}, {fmc_nov}, and {fmc3_ext} coverage, matching the nominal 80% far better than quantile boosting ({qc_oct}, {qc_nov}, and {qc3_ext} over a ~{qw}–{qw3_ext} min width). At 24 hours it lags the rule in Oct/early Nov ({fm24_oct}, {fm24_nov}) but adapts fastest to November's shift ({fm24_ext}, coverage {fmc24_ext}; Table 5, Supplementary Materials).

## Limitations

- **Test coverage.** October has {n_gap_days} missing days, 1–10 Nov has 8 days, and 11–25 Nov has 14 days (11 Nov absent; 22 November days total). The 3-hour advantage over both the historical mean and the one-parameter rule holds consistently across every window, including 14 days after freeze.
- **One year of minute-level data.** The target exists only since October 2025, so seasonal effects such as the winter influenza peak were seen once.
- **An error of ~40 minutes is still large** against a typical wait of two to three hours, and the target moves in 30-minute steps. A calibrated range is the better way to present the forecast.

# Conclusion

**Supported by the evidence.** Hong Kong A&E waits can be forecast 3 hours ahead: across October, early November, and a 14-day post-freeze blind test (`11–25 Nov`), gradient boosting cuts error 21%–35% below the same-slot historical average. Most of that gain comes from recent level and current deviation, which a one-parameter rule captures; gradient boosting adds a consistent ~2 minutes, and zero-shot Chronos-Bolt matches trained boosting at 3 hours while delivering better-calibrated 10–90% intervals than quantile boosting. Weather and air quality, as we measured them, added nothing.

**Not supported.** A day ahead, no supervised model we built beat the one-parameter rule across all windows (losing by 1.0 min over the 14 blind days). Our initial 24-hour gains against the historical mean reflected level drift, not forecasting skill.

**Next steps.** Schedule the live-feed collector to eliminate archive gaps; add slower epidemiological signals such as influenza surveillance for the 24-hour horizon; calibrate quantile forecast ranges; and test hourly district rainfall before ruling weather out completely.

\Needspace{{12\baselineskip}}

# Author Contributions {{.unnumbered}}

| Member name | Specific contributions | Evidence or output |
|-------------|------------------------------------------------|--------------------------------|
| CHEN Hongxi | Data sourcing and archive download; parsing and cleaning; feature pipeline; baselines and first models; November live and blind tests; post-hoc checks and Chronos-Bolt baseline (Section 5.4); integration of the report and slides | `download_ae_archive.py`, `parse_ae.py`, `build_features.py`, `train_models.py`, `live_test.py`, `posthoc_checks.py`, `foundation_baseline.py`, `blind_test_nov.py` |
| NI Haoran | 24-hour forecasting experiments: residual-target modelling, slower-history features and prediction shrinkage. Reviewed the experiment code and the validation and test outputs, compared the selected approach with the historical-mean baseline, and examined the limits of the observed improvement | `ni_24h_experiments.py`, `hn_24h_*.csv` |
| ZHU Yongkai | 3-hour model analysis: hyper-parameter tuning, probability calibration and error comparisons across hospitals and hours. Reviewed the analysis code and outputs, compared validation and test performance, and examined why tuning and calibration did not consistently improve generalisation | `zhu_3h_analysis.py`, `zy_3h_*.csv` |

# AI Tool Use {{.unnumbered}}

We used Claude Code (Anthropic) throughout the project to draft Python scripts, report text, and slides from our instructions, and Kimi to run the 4 November follow-up experiments. During final revision, Antigravity (Gemini) audited the submission, caught figure/commit/EDA-window mismatches, and executed the 11–25 November blind test (`blind_test_nov.py`). We verified all outputs via parser unit tests (`pytest`), chronological train/val/test/blind splits, an independent re-run of the 24 h slow-history experiment (val {v_slow}, Oct {slow_oct}), script-driven number injection (`build_final.py`), unmodified Chronos-Bolt evaluation, and an adversarial AI review that triggered `src/posthoc_checks.py` and `PREREGISTRATION.md`.


```{{=latex}}
\Needspace{{8\baselineskip}}
{{\small
\bibliographystyle{{plainnat}}
\bibliography{{references}}
}}
```

\clearpage

# Supplementary Materials {{.unnumbered}}

**Commands to reproduce the results**, run from a clone of the repository (`tag final-2026-11-26`), in this order. The PDF build needs pandoc and a TeX distribution; Chronos-Bolt runs in a separate environment (`requirements-foundation.txt`). Note: unzip `Group16-Project-data.zip` into `data/processed/` to reproduce the frozen 30 Oct (`hourly_features_full.csv`) and 11 Nov (`hourly_features_latest.csv`) snapshots alongside `hourly_features_blind.csv`.

```
pip install -r requirements.txt && python3 -m pytest -q tests/
# Data: unzip Group16-Project-data.zip into data/processed/, or rebuild through 25 Nov:
python3 src/download_ae_archive.py --start 20230101 --end 20261125 --every 4 --out ~/elec3544_data/raw/ae
python3 src/fetch_aux.py
python3 src/parse_ae.py --raw ~/elec3544_data/raw/ae --out data/processed/ae_snapshots_blind.csv
python3 src/build_features.py --ae data/processed/ae_snapshots_blind.csv \
        --out data/processed/hourly_features_blind.csv
python3 src/train_models.py --out results/models_2026-10-30                       # all features (ablation)
python3 src/train_models.py --drop weather air --out results/models_2026-10-30_v2 # Tables 1 & 4 (*_full.csv)
python3 src/ni_24h_experiments.py && python3 src/zhu_3h_analysis.py               # Fig 2b, 54-config, calib
python3 src/live_test.py && python3 src/live_predictions.py                       # 1-10 Nov (*_latest.csv)
python3 src/posthoc_checks.py && python3 src/posthoc_pairwise.py                  # Sec 5.4 post-hoc (Table 3)
python3 src/foundation_baseline.py                                                # Chronos-Bolt Oct/early Nov
python3 src/blind_test_nov.py && python3 src/blind_foundation_nov.py              # 11-25 Nov blind (*_blind.csv)
python3 src/plot_checkpoint.py && python3 src/plot_report.py
python3 report/build_final.py --official                                          # PDF: pandoc + pdfLaTeX
```

**November day-by-day error (1–10 Nov live, 12–15 Nov pre-registered, and 16–25 Nov extended blind).** Figure 3 shows daily MAE for the frozen models against the historical-mean baseline across all 22 available November days.

![Error on each November day (1–10 Nov live, 12–15 Nov pre-registered, 16–25 Nov extended blind) for the frozen models against the historical-mean baseline.]({fig_nov}){{width=86%}}

**Where the model fails.** The 3-hour error is largest around midnight (about {err_night} minutes at 00:00–02:00) and smallest in the morning ({err_morning} minutes at 07:00–08:00), following the level of the wait itself. By hospital, the model helps most where waits are long and volatile: {top_gain}. It adds almost nothing at the most stable departments: {low_gain}.

**Classification detail and the full model comparison.** Table 4 gives the October classification results summarised in Section 5.1; Figure 4 shows the October regression MAE and classifier ROC-AUC.

{table_cls}

![October 2026 test results for every model. The dotted line marks the historical-mean baseline at 3 hours.]({fig_models}){{width=86%}}

\Needspace{{10\baselineskip}}

**Pre-registration scorecard (11–15 Nov and extended 11–25 Nov).** Table 5 compares the five expectations fixed in `PREREGISTRATION.md` on 11 November (before the data existed) against both the 4-day pre-registered window and the 14-day extended blind window.

{table_prereg}
