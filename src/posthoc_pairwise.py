"""Post-hoc: the 'which of two hospitals is quicker in 3 h' check for the best one-parameter rule
(level_anomaly from posthoc_checks.py), next to the frozen model. Usage: python src/posthoc_pairwise.py
Appends to results/posthoc_summary.txt and writes results/posthoc_pairwise.txt"""
import itertools, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import train_models as tm  # noqa: E402
from live_test import NOV, add_slow  # noqa: E402
from posthoc_checks import add_level  # noqa: E402

for g in ("weather", "air"):
    tm.GROUPS.pop(g)
grid, hol = tm.load_grid("data/processed/hourly_features_latest.csv", "data/raw/aux")
tm.END = grid.hour_ts.max() + pd.Timedelta(hours=1)
X = add_level(add_slow(tm.make_horizon(grid, hol, 3)), 3)
D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
D = tm.add_hist(D, "y_f", D.split == "train")
tr = D[D.split == "train"]
ok = tr[["r_f", "anom"]].notna().all(axis=1)
b = float(((tr.y_f - tr.r_f) * tr.anom)[ok].sum() / (tr.anom ** 2)[ok].sum())
nov = D[D.ts_f >= NOV].copy()
nov["rule"] = (nov.r_f + b * nov.anom).fillna(nov.r_f).fillna(nov["hist"])
p = pd.read_csv("results/live_nov_predictions_3h.csv", parse_dates=["target_hour"])
p = p.merge(nov[["hospital", "ts_f", "rule"]].rename(columns={"ts_f": "target_hour"}), on=["hospital", "target_hour"])
from live_predictions import pairwise  # noqa: E402
lines = [f"rule weight {b:.3f}; rule MAE {np.abs(p.actual - p.rule).mean():.2f}"] + pairwise(p, ("model", "rule", "hist_mean", "wait_now"))
open("results/posthoc_pairwise.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
