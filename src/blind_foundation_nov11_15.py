"""Pre-registered zero-shot Chronos-Bolt baseline on the 11-15 November 2026 blind window.

Usage (separate environment, see requirements-foundation.txt):
  ~/elec3544_data/venv-chronos/bin/python src/blind_foundation_nov11_15.py
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

SWITCH = pd.Timestamp("2025-10-13 09:30")
BLIND_START = pd.Timestamp("2026-11-11")
BLIND_END = pd.Timestamp("2026-11-16")
CONTEXT, HORIZONS, Q = 512, (3, 24), [0.1, 0.5, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--inp", default="data/processed/hourly_features_blind.csv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    h = pd.read_csv(a.inp, usecols=["hospital", "fmt", "hour_ts", "t45_min"], parse_dates=["hour_ts"])
    h = h.sort_values(["hospital", "hour_ts", "fmt"]).drop_duplicates(["hospital", "hour_ts"], keep="first")
    h = h[h.hour_ts >= SWITCH]
    end = h.hour_ts.max()
    idx = pd.date_range(h.hour_ts.min().floor("h"), end, freq="h")
    series = {k: g.set_index("hour_ts").t45_min.reindex(idx).values.astype("float32") for k, g in h.groupby("hospital")}
    pos = {t: i for i, t in enumerate(idx)}
    origins = [t for t in idx if t >= BLIND_START - pd.Timedelta(hours=24) and t + pd.Timedelta(hours=3) <= end]
    jobs = [(hosp, t) for hosp in series for t in origins]
    pipe = BaseChronosPipeline.from_pretrained(a.model, device_map=a.device)
    out, t0 = [], time.time()
    for s in range(0, len(jobs), a.batch):
        chunk = jobs[s:s + a.batch]
        ctx = torch.tensor(np.stack([series[hp][pos[t] - CONTEXT + 1: pos[t] + 1] for hp, t in chunk]))
        q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=Q)
        q = q.numpy()
        for (hp, t), qq in zip(chunk, q):
            for H in HORIZONS:
                out.append((hp, t, t + pd.Timedelta(hours=H), H, *qq[H - 1]))
        if s == 0 or (s // a.batch) % 10 == 0:
            print(f"{s + len(chunk):,}/{len(jobs):,} forecasts, {time.time() - t0:.0f}s", flush=True)
    P = pd.DataFrame(out, columns=["hospital", "forecast_made", "target_hour", "H", "q10", "q50", "q90"])
    act = h.rename(columns={"hour_ts": "target_hour", "t45_min": "actual"})[["hospital", "target_hour", "actual"]]
    P = P.merge(act, on=["hospital", "target_hour"]).dropna(subset=["actual"])
    P = P[(P.target_hour >= BLIND_START) & (P.target_hour < BLIND_END)]
    P["period"] = "blind"
    P["ae"] = (P.actual - P.q50).abs()
    P["covered"] = (P.actual >= P.q10) & (P.actual <= P.q90)
    P.to_csv("results/blind_nov11_15_foundation_predictions.csv", index=False)
    lines = [f"model {a.model}, zero-shot, context {CONTEXT} h, device {a.device}, {len(jobs):,} forecasts in {time.time() - t0:.0f}s"]
    rows = []
    for H, g in P.groupby("H"):
        rows.append(dict(model=a.model, H=H, period="blind", n=len(g), MAE=g.ae.mean(),
                         coverage=g.covered.mean(), width=(g.q90 - g.q10).mean()))
        lines.append(f"H={H:2d} blind: n {len(g):,}  MAE {g.ae.mean():.2f}  10-90% range covers {g.covered.mean():.1%}, mean width {(g.q90 - g.q10).mean():.0f} min")
    pd.DataFrame(rows).to_csv("results/blind_nov11_15_foundation_metrics.csv", index=False)
    open("results/blind_nov11_15_foundation_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
