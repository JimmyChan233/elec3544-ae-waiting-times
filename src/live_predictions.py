"""Per-hour 3 h forecasts of the frozen model on November 2026, and a practical check built on them:
"of two hospitals, which will have the shorter wait in 3 hours?"

Usage: python src/live_predictions.py
Same frozen model as live_test.py (trained on target hours before 1 Sep 2026, v2 features, no tuning here).
Outputs: results/live_nov_predictions_3h.csv, results/live_nov_pairwise.txt
"""
import itertools, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import train_models as tm  # noqa: E402
from live_test import HGB_REG, NOV  # noqa: E402


def pairwise(out, methods, min_gap=30):
    W = {c: out.pivot(index="target_hour", columns="hospital", values=c) for c in ("actual",) + tuple(methods)}
    hit = {m: 0.0 for m in methods}; ties = {m: 0 for m in methods}; n = 0
    for a, b in itertools.combinations(W["actual"].columns, 2):
        d = W["actual"][a] - W["actual"][b]
        diffs = {m: W[m][a] - W[m][b] for m in methods}
        use = (d.abs() >= min_gap) & pd.concat(diffs, axis=1).notna().all(axis=1)
        n += int(use.sum())
        for m, dm in diffs.items():
            hit[m] += float(((np.sign(dm) == np.sign(d)) & use).sum()) + 0.5 * float(((dm == 0) & use).sum())
            ties[m] += int(((dm == 0) & use).sum())
    lines = [f"pairwise 'which hospital is quicker in 3 h' over {n:,} hospital-pair hours with an actual gap >= {min_gap} min "
             f"(all methods available; ties count as half):"]
    return lines + [f"  {m:10s} correct {hit[m] / n:.4f}   (ties {ties[m] / n:.1%})" for m in methods]


def main():
    for g in ("weather", "air"):
        tm.GROUPS.pop(g)
    grid, hol = tm.load_grid("data/processed/hourly_features_latest.csv", "data/raw/aux")
    tm.END = grid.hour_ts.max() + pd.Timedelta(hours=1)
    X = tm.make_horizon(grid, hol, 3)
    D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
    D = tm.add_hist(D, "y_f", D.split == "train")
    tr, nov = D[D.split == "train"], D[D.ts_f >= NOV].copy()
    feats = [f for g, fs in tm.GROUPS.items() if g != "era" for f in fs]
    p, _, _ = tm.fit_predict("hgb", tm.hgb_reg(**HGB_REG), feats, tr, {"nov": nov}, "y_f")
    nov["model"] = p["nov"]
    out = nov[["hospital", "ts", "ts_f", "y0", "y_f", "hist", "model"]].rename(
        columns={"ts": "forecast_made", "ts_f": "target_hour", "y0": "wait_now", "y_f": "actual", "hist": "hist_mean"})
    out.to_csv("results/live_nov_predictions_3h.csv", index=False)

    # Pairwise check: for every pair of hospitals and every target hour where the actual waits differ by >= 30 min,
    # does each method name the hospital that really turned out quicker? Scored on the pair-hours where every
    # method has a value; a tie (the method rates both hospitals the same) counts as half, like a coin flip.
    lines = [f"3 h ahead, 1-10 Nov 2026, frozen model | MAE model {np.abs(out.actual - out.model).mean():.2f}, "
             f"hist {np.abs(out.actual - out.hist_mean).mean():.2f}"] + pairwise(out, ("model", "hist_mean", "wait_now"))
    open("results/live_nov_pairwise.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
