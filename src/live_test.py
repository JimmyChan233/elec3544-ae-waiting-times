"""November 2026 live test: score the already-chosen models on data collected after every modelling choice was made.

Usage: python src/live_test.py [--in data/processed/hourly_features_latest.csv] [--out results/live_nov]

Nothing is tuned here. Features (v2: no weather / AQHI), hyper-parameters and the 24 h "slow history" variant were
all fixed on September validation before November existed. Two ways of using them are scored on November target hours:
  frozen : the exact October-protocol models (trained on target hours before 1 Sep 2026), applied unchanged
  refit  : same recipe, refitted on everything before 1 Nov 2026 (what one would deploy)
Baselines get the same treatment (the historical mean is recomputed from the same training rows).
The frozen models are also scored on the complete October (1-31 Oct; the 30 Oct report used 1-29 Oct).
Outputs: <out>_metrics.csv, <out>_by_day.csv, <out>_summary.txt
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import train_models as tm  # noqa: E402

NOV = pd.Timestamp("2026-11-01")
HGB_REG = dict(learning_rate=0.05, max_leaf_nodes=63)   # chosen on September validation (v2 run, both horizons)
HGB_CLS = dict(learning_rate=0.05, max_leaf_nodes=31)


def add_slow(X):
    """Slower history for the 24 h model: mean of the same clock hour over the last 7 days, and a 7-day rolling mean."""
    parts = []
    for _, g in X.sort_values(["hospital", "ts"]).groupby("hospital", sort=False):
        g = g.copy()
        same = pd.concat([g.y0.shift(k) for k in range(0, 168, 24)], axis=1)
        g["same_hour_7d"] = same.mean(axis=1).where(same.notna().sum(axis=1) >= 3)
        g["roll168"] = g.y0.rolling(168, min_periods=24).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def boot_ci(df, a, b, n=2000, seed=0):
    """Day-level bootstrap of mean(|err a|) - mean(|err b|): days are resampled, not hours, because hours in a day are correlated."""
    days = df.groupby(df.ts_f.dt.normalize())[[a, b]].agg(["sum", "count"])
    sa, sb, cnt = days[(a, "sum")].values, days[(b, "sum")].values, days[(a, "count")].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cnt), size=(n, len(cnt)))
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / cnt[idx].sum(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/hourly_features_latest.csv")
    ap.add_argument("--aux", default="data/raw/aux")
    ap.add_argument("--out", default="results/live_nov")
    a = ap.parse_args()
    for g in ("weather", "air"):
        tm.GROUPS.pop(g)
    grid, hol = tm.load_grid(a.inp, a.aux)
    tm.END = grid.hour_ts.max() + pd.Timedelta(hours=1)   # "test" now means every target hour from 1 Oct to the end
    lines, rows, by_day = [], [], []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"data to {grid.hour_ts.max()} | November days with data: "
        f"{sorted(set(grid[(grid.hour_ts >= NOV) & grid.t45_min.notna()].hour_ts.dt.day))}")
    reg_feats = [f for g, fs in tm.GROUPS.items() if g != "era" for f in fs]
    cls_feats = [f for fs in tm.GROUPS.values() for f in fs]
    for H in (3, 24):
        X = add_slow(tm.make_horizon(grid, hol, H))
        X["period"] = np.select([X.ts_f < tm.VAL, X.ts_f < tm.TEST, X.ts_f < NOV], ["train", "val", "oct"], "nov")
        variants = [("grad_boosting", reg_feats)] + ([("gb_slow_history", reg_feats + ["same_hour_7d", "roll168"])] if H == 24 else [])
        for mode, train_periods, eval_periods in (("frozen", ["train"], ["oct", "nov"]), ("refit", ["train", "val", "oct"], ["nov"])):
            # ---- regression (new feed format only)
            D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
            tr_mask = D.period.isin(train_periods)
            D = tm.add_hist(D, "y_f", tr_mask)
            tr = D[tr_mask]
            ev = {p: D[D.period == p].copy() for p in eval_periods}
            preds = {"persistence": {p: e.y0.fillna(e["hist"]).values for p, e in ev.items()},
                     "hist_mean": {p: e["hist"].values for p, e in ev.items()}}
            for name, feats in variants:
                preds[name], _, _ = tm.fit_predict("hgb", tm.hgb_reg(**HGB_REG), feats, tr, ev, "y_f")
            for p, e in ev.items():
                for name, pr in preds.items():
                    m = tm.reg_metrics(e.y_f.values, pr[p])
                    rows.append(dict(task="reg", H=H, mode=mode, period=p, model=name, **m))
                    e[f"ae_{name}"] = np.abs(e.y_f.values - pr[p])
                for name, _ in variants:
                    lo, hi = boot_ci(e, f"ae_{name}", "ae_hist_mean")
                    d = e[f"ae_{name}"].mean() - e["ae_hist_mean"].mean()
                    rows.append(dict(task="reg_diff_vs_hist", H=H, mode=mode, period=p, model=name, n=len(e), MAE=d, ci_lo=lo, ci_hi=hi))
                    log(f"reg H={H:2d} {mode:6s} {p}: n {len(e):,} | hist {e['ae_hist_mean'].mean():.2f} | persistence "
                        f"{e['ae_persistence'].mean():.2f} | {name} {e[f'ae_{name}'].mean():.2f} "
                        f"(diff {d:+.2f}, 95% day-bootstrap CI {lo:+.2f} .. {hi:+.2f})")
                if p == "nov":
                    cols = [c for c in e.columns if c.startswith("ae_")]
                    by_day.append(e.groupby(e.ts_f.dt.normalize())[cols].mean().assign(H=H, mode=mode, n=e.groupby(e.ts_f.dt.normalize()).size()).reset_index())
            # ---- classification (> 2 h), both formats with an era flag
            C = X[X.lbl_f.notna()].copy()
            C["lbl_f"] = C.lbl_f.astype(int)
            ctr_mask = C.period.isin(train_periods)
            C = tm.add_hist(C, "lbl_f", ctr_mask & (C.ts_f >= tm.SWITCH))
            ctr = C[ctr_mask]
            cev = {p: C[C.period == p] for p in eval_periods}
            cp, _, _ = tm.fit_predict("hgb", tm.hgb_cls(**HGB_CLS), cls_feats, ctr, cev, "lbl_f")
            for p, e in cev.items():
                for name, s in (("hist_mean", e["hist"].values), ("grad_boosting", cp[p])):
                    m = tm.cls_metrics(e.lbl_f.values, s)
                    rows.append(dict(task="cls", H=H, mode=mode, period=p, model=name, **m))
                    log(f"cls H={H:2d} {mode:6s} {p}: {name:14s} acc {m['acc']:.3f} AUC {m['AUC']:.3f} (n {m['n']:,}, pos {e.lbl_f.mean():.2f})")
    pd.DataFrame(rows).to_csv(f"{a.out}_metrics.csv", index=False)
    pd.concat(by_day).to_csv(f"{a.out}_by_day.csv", index=False)
    open(f"{a.out}_summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
