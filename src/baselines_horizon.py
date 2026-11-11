"""Baselines at the forecast horizons defined in the proposal (3 h and 24 h ahead).

Usage: python src/baselines_horizon.py [--in data/processed/hourly_features_full.csv] [--split 2026-09-01]

For each hospital and hour t we predict the value at t+h:
  persistence : y(t+h) ~ y(t)
  hist mean   : y(t+h) ~ mean over TRAIN of same hospital x weekday x hour-of-day of (t+h)
Regression target : t45_min (cat 4/5 wait, minutes; new format only, where it is a true median)
Classification    : t45_over2h (both formats)
Train = rows with target time < split date; test = the rest.
"""
import argparse
import numpy as np
import pandas as pd


def metrics_reg(y, p):
    m = ~(y.isna() | p.isna())
    y, p = y[m], p[m]
    return dict(n=int(m.sum()), MAE=float(np.mean(np.abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))))


def metrics_cls(y, p):
    m = ~(y.isna() | p.isna())
    y, p = y[m].astype(int), p[m]
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    # AUC via rank statistic
    pos, neg = p[y == 1], p[y == 0]
    auc = float((pos.values[:, None] > neg.values[None, :]).mean() + 0.5 * (pos.values[:, None] == neg.values[None, :]).mean()) if len(pos) and len(neg) and len(pos) * len(neg) < 4e7 else float("nan")
    return dict(n=len(y), base_rate=float(y.mean()), acc=acc, precision=prec, recall=rec, AUC=auc)


def run(h, horizon, target, split, cls=False):
    h = h.sort_values(["hospital", "hour_ts"]).copy()
    # shift target forward by `horizon` hours within hospital, only where the gap is exactly `horizon`
    g = h.groupby("hospital")
    h["y_future"] = g[target].shift(-horizon)
    h["ts_future"] = g["hour_ts"].shift(-horizon)
    ok = (h["ts_future"] - h["hour_ts"]) == pd.Timedelta(hours=horizon)
    h = h[ok & h["y_future"].notna()].copy()
    h["dow_f"] = h["ts_future"].dt.dayofweek; h["hr_f"] = h["ts_future"].dt.hour
    tr = h[h["ts_future"] < split]; te = h[h["ts_future"] >= split].copy()
    if len(tr) == 0 or len(te) == 0:
        return None
    te["pers"] = te[target]
    m1 = tr.groupby(["hospital", "dow_f", "hr_f"])["y_future"].mean().rename("m1")
    m2 = tr.groupby(["hospital", "hr_f"])["y_future"].mean().rename("m2")
    m3 = tr.groupby("hospital")["y_future"].mean().rename("m3")
    te = te.join(m1, on=["hospital", "dow_f", "hr_f"]).join(m2, on=["hospital", "hr_f"]).join(m3, on="hospital")
    te["hist"] = te["m1"].fillna(te["m2"]).fillna(te["m3"])
    f = metrics_cls if cls else metrics_reg
    return {"train": len(tr), "test": len(te), "pers": f(te["y_future"], te["pers"]), "hist": f(te["y_future"], te["hist"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/hourly_features_full.csv")
    ap.add_argument("--split", default="2026-09-01")
    a = ap.parse_args()
    h = pd.read_csv(a.inp, parse_dates=["hour_ts"])
    split = pd.Timestamp(a.split)
    print(f"rows {len(h):,}  range {h.hour_ts.min()} .. {h.hour_ts.max()}  split {split.date()}")
    for horizon in (3, 24):
        r = run(h[h.fmt == "new"], horizon, "t45_min", split)
        if r:
            print(f"\n[regression, cat 4/5 wait (min), new format] horizon {horizon}h  train {r['train']:,} test {r['test']:,}")
            for k in ("pers", "hist"):
                print(f"   {k:5s} n={r[k]['n']:6d}  MAE={r[k]['MAE']:6.1f}  RMSE={r[k]['RMSE']:6.1f}")
        r = run(h, horizon, "t45_over2h", split, cls=True)
        if r:
            print(f"[classification, wait > 2h, both formats] horizon {horizon}h  train {r['train']:,} test {r['test']:,}  base rate {r['pers']['base_rate']:.3f}")
            for k in ("pers", "hist"):
                v = r[k]; print(f"   {k:5s} n={v['n']:6d}  acc={v['acc']:.3f}  prec={v['precision']:.3f}  rec={v['recall']:.3f}  AUC={v['AUC']:.3f}")


if __name__ == "__main__":
    main()
