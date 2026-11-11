"""Post-hoc checks added on 11 Nov 2026, AFTER the November live test had been scored (so nothing here is a blind test).

Usage: python src/posthoc_checks.py            -> results/posthoc_{metrics.csv,summary.txt}

Why: an independent review asked whether the two baselines were too easy. This script adds
  1. stronger simple rules, each with at most one fitted number (fitted on training rows only):
       rolling28        mean of the same clock hour as the target over the previous 28 days
       damped_anomaly   hist(target) + b * (wait now - hist(now))
       level_anomaly    rolling28(target) + b * (wait now - rolling28(now))
  2. a single decision tree (promised in the proposal; depth chosen on September validation)
  3. gradient boosting with those level features added (chosen or not on September validation)
  4. quantile gradient boosting (10 / 50 / 90 %) and the coverage of its 10-90 % range
Models are trained on target hours before 1 Sep 2026, exactly like the frozen models, and scored on September
(validation), the complete October and 1-10 November. Differences come with a day-level bootstrap interval.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(__file__))
import train_models as tm  # noqa: E402
from live_test import HGB_REG, NOV, add_slow, boot_ci  # noqa: E402


def add_level(X, H):
    """Level features built only from waits observed up to the forecast hour t, new feed format only."""
    parts = []
    for _, g in X.sort_values(["hospital", "ts"]).groupby("hospital", sort=False):
        g = g.copy()
        y = g.y0.where(g.ts >= tm.SWITCH)
        f = pd.concat([y.shift(24 * k - H) for k in range(1, 29)], axis=1)   # values at target - 24k h, all <= t
        o = pd.concat([y.shift(24 * k) for k in range(1, 29)], axis=1)       # values at t - 24k h
        g["r_f"] = f.mean(axis=1).where(f.notna().sum(axis=1) >= 7)
        g["r_t"] = o.mean(axis=1).where(o.notna().sum(axis=1) >= 7)
        g["anom"] = g.y0 - g.r_t
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main():
    for g in ("weather", "air"):
        tm.GROUPS.pop(g)
    grid, hol = tm.load_grid("data/processed/hourly_features_latest.csv", "data/raw/aux")
    tm.END = grid.hour_ts.max() + pd.Timedelta(hours=1)
    lines, rows = [], []

    def log(s):
        print(s, flush=True); lines.append(s)

    new = grid[(grid.hour_ts >= tm.SWITCH) & grid.t45_min.notna()]
    log(f"new-format waits that are a multiple of 30 min: {(new.t45_min % 30 == 0).mean():.1%}; exactly 120 min: {(new.t45_min == 120).mean():.1%}")
    oct_ = grid[(grid.hour_ts >= "2026-10-01") & (grid.hour_ts < "2026-11-01")]
    log(f"October 2026 hours with no weather value (previous-day rainfall): {oct_.rf_mm_prev.isna().mean():.1%}")
    base = [f for g, fs in tm.GROUPS.items() if g != "era" for f in fs]
    for H in (3, 24):
        X = add_level(add_slow(tm.make_horizon(grid, hol, H)), H)
        X["period"] = np.select([X.ts_f < tm.VAL, X.ts_f < tm.TEST, X.ts_f < NOV], ["train", "val", "oct"], "nov")
        D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
        trm = D.period == "train"
        D = tm.add_hist(D, "y_f", trm)
        m = D[trm].groupby(["hosp", "dow_f", "hr_f"]).y_f.mean().rename("hist_t")
        D = D.join(m, on=["hosp", D.ts.dt.dayofweek.rename("dow_f"), D.ts.dt.hour.rename("hr_f")]) if False else D.assign(
            hist_t=pd.MultiIndex.from_arrays([D.hosp, D.ts.dt.dayofweek, D.ts.dt.hour]).map(m.to_dict()))
        D["hist_t"] = D.hist_t.fillna(D["hist"])
        tr = D[trm]
        ev = {p: D[D.period == p].copy() for p in ("val", "oct", "nov")}

        def beta(a, b):  # least squares through the origin on training rows: (y_f - a) ~ beta * b
            ok = tr[[a, b]].notna().all(axis=1)
            return float(((tr.y_f - tr[a]) * tr[b])[ok].sum() / (tr[b] ** 2)[ok].sum())

        tr = tr.assign(d_hist=tr.y0 - tr.hist_t)
        b1, b2 = beta("hist", "d_hist"), beta("r_f", "anom")
        log(f"\nH={H}: damped-anomaly weight {b1:.3f}, level-anomaly weight {b2:.3f} (fitted on training rows)")
        preds = {p: {} for p in ev}
        for p, e in ev.items():
            preds[p]["hist_mean"] = e["hist"].values
            preds[p]["persistence"] = e.y0.fillna(e["hist"]).values
            preds[p]["rolling28"] = e.r_f.fillna(e["hist"]).values
            preds[p]["damped_anomaly"] = (e["hist"] + b1 * (e.y0 - e.hist_t)).fillna(e["hist"]).values
            preds[p]["level_anomaly"] = (e.r_f + b2 * e.anom).fillna(e.r_f).fillna(e["hist"]).values
        # decision tree, depth on validation
        best = None
        for depth in (6, 10, 14):
            t = make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=0)).fit(tr[base], tr.y_f)
            s = np.abs(ev["val"].y_f - t.predict(ev["val"][base])).mean()
            if best is None or s < best[0]:
                best = (s, depth, t)
        log(f"  decision tree: depth {best[1]} chosen on validation (MAE {best[0]:.2f})")
        for p, e in ev.items():
            preds[p]["decision_tree"] = best[2].predict(e[base])
        level = base + ["r_f", "r_t", "anom", "same_hour_7d", "roll168"]
        slow = base + ["same_hour_7d", "roll168"]
        for name, feats in (("grad_boosting", base), ("gb_slow_history", slow), ("gb_level", level)):
            pr, secs, _ = tm.fit_predict("hgb", tm.hgb_reg(**HGB_REG), feats, tr, ev, "y_f")
            for p in ev:
                preds[p][name] = pr[p]
            log(f"  fitted {name} in {secs:.0f}s")
        # quantile boosting on the base features
        q = {}
        for a in (0.1, 0.5, 0.9):
            est = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=400, early_stopping=False, categorical_features=["hosp"],
                                                random_state=0, **HGB_REG).fit(tr[base], tr.y_f)
            q[a] = {p: est.predict(e[base]) for p, e in ev.items()}
        for p, e in ev.items():
            lo, hi = np.minimum(q[0.1][p], q[0.9][p]), np.maximum(q[0.1][p], q[0.9][p])
            cov, width = float(((e.y_f >= lo) & (e.y_f <= hi)).mean()), float(np.mean(hi - lo))
            rows.append(dict(H=H, period=p, model="quantile_10_90", n=len(e), coverage=cov, width=width,
                             below=float((e.y_f < lo).mean()), above=float((e.y_f > hi).mean()), MAE=float(np.abs(e.y_f - q[0.5][p]).mean())))
            log(f"  {p}: 10-90% range covers {cov:.1%} (below {(e.y_f < lo).mean():.1%}, above {(e.y_f > hi).mean():.1%}), mean width {width:.0f} min, median-model MAE {np.abs(e.y_f - q[0.5][p]).mean():.2f}")
        for p, e in ev.items():
            for name, pr in preds[p].items():
                e[f"ae_{name}"] = np.abs(e.y_f.values - pr)
                rows.append(dict(H=H, period=p, model=name, n=len(e), MAE=float(e[f"ae_{name}"].mean())))
            log(f"  {p}: " + "  ".join(f"{k} {e[f'ae_{k}'].mean():.2f}" for k in preds[p]))
            rules = ["rolling28", "damped_anomaly", "level_anomaly", "hist_mean"]
            best_rule = min(rules, key=lambda k: ev["val"][f"ae_{k}"].mean()) if "ae_hist_mean" in ev["val"] else None
            if p != "val":
                best_rule = min(rules, key=lambda k: ev["val"][f"ae_{k}"].mean())
                for mdl in ("grad_boosting", "gb_slow_history", "gb_level"):
                    lo, hi = boot_ci(e, f"ae_{mdl}", f"ae_{best_rule}")
                    d = e[f"ae_{mdl}"].mean() - e[f"ae_{best_rule}"].mean()
                    rows.append(dict(H=H, period=p, model=f"{mdl}_minus_{best_rule}", n=len(e), MAE=float(d), ci_lo=lo, ci_hi=hi))
                    log(f"    {mdl} vs best simple rule on validation ({best_rule}): {d:+.2f} (95% day-bootstrap {lo:+.2f} .. {hi:+.2f})")
                days = e.groupby(e.ts_f.dt.normalize())[["ae_grad_boosting", f"ae_{best_rule}"]].mean()
                log(f"    days grad_boosting beats {best_rule}: {(days.ae_grad_boosting < days[f'ae_{best_rule}']).sum()} of {len(days)}")
    pd.DataFrame(rows).to_csv("results/posthoc_metrics.csv", index=False)
    open("results/posthoc_summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
