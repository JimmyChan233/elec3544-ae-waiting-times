"""Quick EDA + naive baselines on the parsed A&E table, to prove the pipeline end to end.

Usage: python src/eda_baseline.py [--in data/processed/ae_snapshots.csv]

Baselines (per hospital):
  persistence      : predict this hour's value with the previous hour's value
  hist mean (h,dow,hr): predict with the mean of the same hospital, weekday and hour over the TRAIN part
Chronological split: first 70% of hours = train, last 30% = test (within the sample).
"""
import argparse
import numpy as np
import pandas as pd


def hourly(df):
    df = df.copy()
    df["hour_ts"] = df["ts"].dt.floor("h")
    agg = (df.groupby(["hospital", "fmt", "hour_ts"], as_index=False)
             .agg(t45_min=("t45_min", "mean"), t45_over2h=("t45_over2h", "max"),
                  t3p50_min=("t3p50_min", "mean"), n_snap=("ts", "count")))
    agg["dow"] = agg["hour_ts"].dt.dayofweek
    agg["hr"] = agg["hour_ts"].dt.hour
    return agg.sort_values(["hospital", "hour_ts"])


def baselines(h, target):
    h = h.dropna(subset=[target]).copy()
    cut = h["hour_ts"].quantile(0.7)
    tr, te = h[h["hour_ts"] <= cut], h[h["hour_ts"] > cut].copy()
    # persistence
    te["pers"] = te.groupby("hospital")[target].shift(1)
    # historical mean by hospital x dow x hour, fall back to hospital x hour, then hospital
    m1 = tr.groupby(["hospital", "dow", "hr"])[target].mean().rename("m1")
    m2 = tr.groupby(["hospital", "hr"])[target].mean().rename("m2")
    m3 = tr.groupby("hospital")[target].mean().rename("m3")
    te = te.join(m1, on=["hospital", "dow", "hr"]).join(m2, on=["hospital", "hr"]).join(m3, on="hospital")
    te["hist"] = te["m1"].fillna(te["m2"]).fillna(te["m3"])
    out = {}
    for b in ("pers", "hist"):
        v = te.dropna(subset=[b])
        out[b] = dict(n=len(v), MAE=float(np.mean(np.abs(v[b] - v[target]))),
                      RMSE=float(np.sqrt(np.mean((v[b] - v[target]) ** 2))))
    return out, len(tr), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/ae_snapshots.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.inp, parse_dates=["ts"])
    print(f"snapshot-level rows: {len(df):,}  snapshots: {df['ts'].nunique():,}  hospitals: {df['hospital'].nunique()}")
    print(f"date range: {df['ts'].min()} .. {df['ts'].max()}")
    print("\nrows by format:\n", df.groupby("fmt").size().to_string())
    print("\nunparsed t45_min by format:\n", df.groupby("fmt")["t45_min"].apply(lambda s: int(s.isna().sum())).to_string())
    h = hourly(df)
    print(f"\nhourly rows: {len(h):,}  (>1,000 required)")
    print("\nshare of hours with cat 4/5 wait > 2h, by format:\n", h.groupby("fmt")["t45_over2h"].mean().round(3).to_string())
    print("\nmean cat 4/5 wait (min) by hour of day, new format:")
    print(h[h.fmt == "new"].groupby("hr")["t45_min"].mean().round(0).to_string())
    print("\nbusiest hospitals (mean cat 4/5 wait, min):")
    print(h.groupby("hospital")["t45_min"].mean().sort_values(ascending=False).head(5).round(0).to_string())
    for fmt in ("old", "new"):
        sub = h[h.fmt == fmt]
        if len(sub) < 100:
            continue
        res, ntr, nte = baselines(sub, "t45_min")
        print(f"\n[{fmt}] baselines on cat 4/5 wait (train hours {ntr}, test hours {nte}):")
        for k, v in res.items():
            print(f"   {k:5s} n={v['n']:5d}  MAE={v['MAE']:.1f} min  RMSE={v['RMSE']:.1f} min")
    if (h.fmt == "new").sum() > 100:
        res, ntr, nte = baselines(h[h.fmt == "new"], "t3p50_min")
        print(f"\n[new] baselines on cat 3 median wait (train {ntr}, test {nte}):")
        for k, v in res.items():
            print(f"   {k:5s} n={v['n']:5d}  MAE={v['MAE']:.1f} min  RMSE={v['RMSE']:.1f} min")


if __name__ == "__main__":
    main()
