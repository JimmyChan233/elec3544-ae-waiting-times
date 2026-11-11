# Pre-registration: blind test on 11–15 November 2026

Written and committed on 11 November 2026, before any data for 11–15 November existed in our files
(the latest snapshot we hold is 10 November 23:15). The commit time of this file is the evidence.

## Why

Section 5.4 of our report (stronger one-parameter rules, the zero-shot foundation model, the forecast
ranges) was added after October and 1–10 November had been scored, so it is not a blind test.
Days 11–15 November are new to every method below. We fix the methods and the claims now and will
score them once, on or after 16 November.

## What is frozen

Everything is trained or fitted on target hours before 1 September 2026, exactly as in the report.
Nothing will be refitted, re-tuned or re-selected.

| Method | Definition |
|---|---|
| Historical mean | training-set mean for hospital × weekday × hour |
| One-line rule | 28-day mean for the target's clock hour + w × (wait now − 28-day mean for the current hour); w = 0.618 at 3 h, 0.205 at 24 h |
| Gradient boosting (3 h) | `train_models.py` v2 features, learning rate 0.05, 63 leaves, 400 iterations, seed 0 |
| Gradient boosting + slow history (24 h) | the same plus the 7-day same-hour mean and the 7-day rolling mean |
| Chronos-Bolt | `amazon/chronos-bolt-base`, zero-shot, last 512 hourly waits of the same hospital |
| Quantile boosting | 10 %, 50 %, 90 % levels, same settings as the regression model |

## What we will report, whatever the outcome

On target hours from 11 November 00:00 to 15 November 23:00 that have an archived snapshot:

1. MAE of every method above at 3 h and 24 h.
2. The difference between gradient boosting and the one-line rule, with a day-level bootstrap interval
   and the number of days each wins.
3. Coverage and width of the 10–90 % range for quantile boosting and for Chronos-Bolt.
4. The pairwise "which of two hospitals is quicker in 3 hours" accuracy for the board, the historical
   mean, the one-line rule and the model.

## What we expect (so the test can fail)

- **3 hours:** gradient boosting beats the one-line rule by 1–3 minutes of MAE and wins on most days.
- **24 hours:** slow-history boosting and the one-line rule differ by less than 1.5 minutes either way.
- **Chronos-Bolt at 3 hours** lands between the one-line rule and gradient boosting, or within 1 minute of either.
- **Ranges:** quantile boosting covers 70–80 % (too narrow); Chronos-Bolt covers 75–85 %.
- **Pairwise:** rule and model within 2 percentage points of each other, both at least 2 points above the board.

Five days is a small sample. If the archive is missing days in this window, we score the days that exist
and say how many there were.

## Amendment, 11 November 2026 (before any 11–15 November data existed)

The pairwise check now scores every method on the same pair-hours and counts a tie as half; an earlier version
counted ties on the board as misses. On 1–10 November this moves the board from 69% to 77%, so the pairwise
expectation above was changed from "at least 8 points above the board" to "at least 2 points". Nothing else changed.
