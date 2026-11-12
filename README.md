# ELEC3544 Group 16: Predicting A&E Waiting Times in Hong Kong Public Hospitals

HKU ELEC3544 (Data Science with Foundation Models), Fall 2026. Chen Hongxi, Ni Haoran, Zhu Yongkai.

We forecast the waiting time for semi-urgent and non-urgent patients at Hong Kong's 18 public A&E departments,
3 hours and 24 hours ahead, from the Hospital Authority's open-data feed rebuilt through the data.gov.hk archive.

| Mean absolute error, minutes | 3 h, Oct 2026 | 3 h, 1–10 Nov | 24 h, Oct | 24 h, 1–10 Nov |
|---|---|---|---|---|
| Historical mean (hospital × weekday × hour) | 51.8 | 53.2 | 51.8 | 53.2 |
| One-line rule (28-day level + damped anomaly) | 43.4 | 41.2 | 50.4 | 47.9 |
| Chronos-Bolt, zero-shot | 42.8 | 40.5 | 52.0 | 50.3 |
| Gradient boosting (ours) | 41.1 | 39.5 | 50.2 | 47.4 |

Short version: the wait three hours ahead is predictable, most of the gain is available to a one-parameter rule,
gradient boosting adds about two minutes, and a day ahead nothing we built beats the rule.

- `EXPERIMENT_LOG.md`: every change, why, and what happened.
- `PREREGISTRATION.md`: the blind test on 11–15 November 2026, fixed before that data existed.
- `report/report_final.md`: the report source; `report/build_final.py` fills in every number from `results/`.
- Processed tables are in the release asset `Group16-Project-data.zip`; unzip it into `data/processed/` (see `data/processed/README.md`).
- The submitted report uses the official course LaTeX template: `python3 report/build_final.py --official` (text in `report/report_official.md`; needs pandoc, pdfLaTeX and the course style file `elec3544_report.sty` in `report/template/`). The experiments need only Python.
- AI tools were used heavily in this project; the report's "Use of AI tools" section says how and how the outputs were checked.

## Pipeline

```
src/download_ae_archive.py  -> data/raw/ae/{old,new}/<YYYYMMDD-HHMM>.json   (HA snapshots, 96/day, from 2023-01)
src/fetch_aux.py            -> data/raw/aux/{holidays.json, aqhi/*.csv, hko/*.csv}
src/parse_ae.py             -> data/processed/ae_snapshots.csv               (one row per snapshot x hospital)
src/build_features.py       -> data/processed/hourly_features.csv            (one row per hospital x hour, with features)
src/eda_baseline.py         -> prints EDA + persistence / historical-mean baselines
tests/test_parse.py         -> pytest for the waiting-time string parser
src/baselines_horizon.py    -> persistence / historical-mean baselines at 3 h and 24 h
src/train_models.py         -> linear, ridge, lasso, RF, grad boosting, logistic vs baselines; ablation; error by hospital/hour
src/plot_results.py         -> model-comparison and ablation figures
src/plot_checkpoint.py      -> daily-cycle / regime-shift figure
src/pull_live.py            -> save the live feed every 15 min (backup for archive gaps; never scheduled)
src/ni_24h_experiments.py   -> 24 h variants: residual target, shrinkage, slow history (results/hn_24h_*)
src/zhu_3h_analysis.py      -> 3 h 54-configuration search, calibration, error plots (results/zy_3h_*)
src/live_test.py            -> November 2026 live test of the frozen models (results/live_nov_*)
src/live_predictions.py     -> per-hour 3 h forecasts for November and the 'which hospital is quicker' check
src/posthoc_checks.py       -> stronger one-parameter baselines, decision tree, level features, quantile ranges (after the live test)
src/posthoc_pairwise.py     -> pairwise hospital check for the best simple rule
src/foundation_baseline.py  -> zero-shot Chronos-Bolt baseline (separate env: requirements-foundation.txt)
src/plot_report.py          -> figures for the final report
report/build_final.py       -> fills report/report_final.md from results and renders ../Assignments/Report/ELEC3544-Report-Group16.pdf
```

Latest results: `results/live_nov_summary.txt` (November live test) and `results/SUMMARY_2026-10-30.md`; every change and its effect: `EXPERIMENT_LOG.md`.

```bash
python3 src/train_models.py --drop weather air --out results/models_2026-10-30_v2   # ~10 min
python3 src/plot_results.py --prefix results/models_2026-10-30_v2
```

## Quick start

```bash
cd Project
python3 -m pytest -q tests/
python3 src/download_ae_archive.py --start 20240108 --end 20240114      # one sample week (old format)
python3 src/download_ae_archive.py --start 20260901 --end 20260907      # one sample week (new format)
python3 src/fetch_aux.py
python3 src/parse_ae.py
python3 src/build_features.py
python3 src/eda_baseline.py
```

Full history: `--start 20230101 --end 20261130`. About 130,000 snapshots; the script is resumable, so run it in chunks.
Use `--every 4` to keep hourly snapshots only if time is short.

## Data notes (verified 2026-10-07/08)

- Feed URL changed on 2025-10-13 09:30. Old: one categorical field `topWait` ("Around 1 hour", "Over 1 hour" ... "Over 8 hours").
  New: `t1wt, t2wt, t3p50, t3p95, t45p50, t45p95` as text ("less than 15 minutes", "23 minutes", "2.5 hours").
- `t45_min` is comparable across formats as the cat 4/5 wait (old = lower bound of the category, new = median). `t45_over2h` is defined for both.
- The regression target is `t45_min` (cat 4/5 median wait); in minutes it exists only in the new format, i.e. from 2025-10-13. `t3p50_min` (cat 3) is parsed but not modelled.
- `list-file-versions` truncates at ~35k results; the downloader queries one calendar month at a time.
- `python3` + `urllib` fails SSL verification on this Mac; the scripts shell out to `curl`.
- HKO daily series: `data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_{RF,RH,TEMP,MAXT,MINT}_ALL.csv`. Hourly weather is not in the archive.
- AQHI: `aqhi.gov.hk/epd/ddata/html/history/YYYY/YYYYMM_Eng.csv`, hour 24 = midnight of next day, `*` = provisional.
- HKO daily "ALL" files lag about one month (on 2026-10-08 they ended at 2026-08-31). For the live November test, fetch the latest days from the HKO daily extract page or drop the weather features for that window; the calendar, AQHI and lag features are available in near real time.
- `t45_min` for the old format is the lower bound of a category ("Over 3 hours" = 180), so old-format regression numbers are coarse; the classification label is the cleaner cross-format target.

## Full-history download (started 2026-10-08)

`./run_full_download.sh` launches 8 workers (hourly snapshots, `--every 4`) writing to `~/elec3544_data/raw/ae/` (outside iCloud on purpose:
33k small files would thrash iCloud sync). Re-run the script any time to fill gaps. Then:

```bash
python3 src/parse_ae.py --raw ~/elec3544_data/raw/ae --out data/processed/ae_snapshots_full.csv
python3 src/build_features.py --ae data/processed/ae_snapshots_full.csv --out data/processed/hourly_features_full.csv
python3 src/eda_baseline.py --in data/processed/ae_snapshots_full.csv
```
- list-file-versions rejects an end date >= today; the downloader caps the query at yesterday (fixed 2026-10-08).
