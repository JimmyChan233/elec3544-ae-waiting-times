"""Post-freeze blind test on 11-25 November 2026, preserving the pre-registered 11-15 Nov subset.

Usage: python3 src/blind_test_nov.py

Strictly frozen: all models and one-parameter weights are fitted only on target hours before
1 September 2026 (identical to posthoc_checks.py and live_test.py). Nothing is refitted,
re-tuned, or re-selected on September, October, or November data.
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
from live_test import HGB_REG, add_slow, boot_ci  # noqa: E402
from live_predictions import pairwise  # noqa: E402
from posthoc_checks import add_level  # noqa: E402

PREREG_START = pd.Timestamp("2026-11-11")
PREREG_END = pd.Timestamp("2026-11-16")
EXT_END = pd.Timestamp("2026-11-26")


def main():
    for g in ("weather", "air"):
        tm.GROUPS.pop(g)
    grid, hol = tm.load_grid("data/processed/hourly_features_blind.csv", "data/raw/aux")
    tm.END = grid.hour_ts.max() + pd.Timedelta(hours=1)
    lines, rows, by_day, pw_lines = [], [], [], []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    blind_grid = grid[(grid.hour_ts >= PREREG_START) & (grid.hour_ts < EXT_END) & grid.t45_min.notna()]
    log(f"data to {grid.hour_ts.max()} | blind window (11-25 Nov) days with data: "
        f"{sorted(set(blind_grid.hour_ts.dt.day))}")

    base = [f for g, fs in tm.GROUPS.items() if g != "era" for f in fs]
    slow = base + ["same_hour_7d", "roll168"]
    level = base + ["r_f", "r_t", "anom", "same_hour_7d", "roll168"]

    for H in (3, 24):
        X = add_level(add_slow(tm.make_horizon(grid, hol, H)), H)
        X["period"] = np.select(
            [X.ts_f < tm.VAL, X.ts_f < tm.TEST, X.ts_f < PREREG_START, X.ts_f < EXT_END],
            ["train", "val", "pre_blind", "ext_11_25"],
            "after",
        )
        D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
        trm = D.period == "train"
        D = tm.add_hist(D, "y_f", trm)
        m = D[trm].groupby(["hosp", "dow_f", "hr_f"]).y_f.mean().rename("hist_t")
        D = D.assign(hist_t=pd.MultiIndex.from_arrays([D.hosp, D.ts.dt.dayofweek, D.ts.dt.hour]).map(m.to_dict()))
        D["hist_t"] = D["hist_t"].fillna(D["hist"])
        tr = D[trm].assign(d_hist=lambda d: d.y0 - d.hist_t)
        va = D[D.period == "val"].copy()
        bl_ext = D[D.period == "ext_11_25"].copy()

        def beta(a, b):
            ok = tr[[a, b]].notna().all(axis=1)
            return float(((tr.y_f - tr[a]) * tr[b])[ok].sum() / (tr[b] ** 2)[ok].sum())

        b1, b2 = beta("hist", "d_hist"), beta("r_f", "anom")
        expected_b2 = 0.618 if H == 3 else 0.205
        if abs(b2 - expected_b2) > 0.002:
            raise RuntimeError(f"Unexpected level-anomaly weight at H={H}: got {b2:.4f}, expected ~{expected_b2}")
        log(f"\nH={H}: damped-anomaly weight {b1:.3f}, level-anomaly weight {b2:.3f} (fitted on <2026-09-01, n={len(tr):,})")

        preds = {}
        preds["hist_mean"] = bl_ext["hist"].values
        preds["persistence"] = bl_ext.y0.fillna(bl_ext["hist"]).values
        preds["rolling28"] = bl_ext.r_f.fillna(bl_ext["hist"]).values
        preds["damped_anomaly"] = (bl_ext["hist"] + b1 * (bl_ext.y0 - bl_ext.hist_t)).fillna(bl_ext["hist"]).values
        preds["level_anomaly"] = (bl_ext.r_f + b2 * bl_ext.anom).fillna(bl_ext.r_f).fillna(bl_ext["hist"]).values

        best = None
        for depth in (6, 10, 14):
            t = make_pipeline(SimpleImputer(strategy="median"),
                              DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=0)).fit(tr[base], tr.y_f)
            s = np.abs(va.y_f - t.predict(va[base])).mean()
            if best is None or s < best[0]:
                best = (s, depth, t)
        preds["decision_tree"] = best[2].predict(bl_ext[base])
        log(f"  decision tree depth {best[1]} (val MAE {best[0]:.2f})")

        for name, feats in (("grad_boosting", base), ("gb_slow_history", slow), ("gb_level", level)):
            pr, secs, _ = tm.fit_predict("hgb", tm.hgb_reg(**HGB_REG), feats, tr, {"blind": bl_ext}, "y_f")
            preds[name] = pr["blind"]
            log(f"  fitted {name} in {secs:.0f}s")

        q = {}
        for a in (0.1, 0.5, 0.9):
            est = HistGradientBoostingRegressor(
                loss="quantile", quantile=a, max_iter=400, early_stopping=False,
                categorical_features=["hosp"], random_state=0, **HGB_REG,
            ).fit(tr[base], tr.y_f)
            q[a] = est.predict(bl_ext[base])
        lo_q, hi_q = np.minimum(q[0.1], q[0.9]), np.maximum(q[0.1], q[0.9])

        for name, pr in preds.items():
            bl_ext[f"ae_{name}"] = np.abs(bl_ext.y_f.values - pr)
        bl_ext["q_cov"] = (bl_ext.y_f >= lo_q) & (bl_ext.y_f <= hi_q)
        bl_ext["q_below"] = bl_ext.y_f < lo_q
        bl_ext["q_above"] = bl_ext.y_f > hi_q
        bl_ext["q_width"] = hi_q - lo_q
        bl_ext["q50_ae"] = np.abs(bl_ext.y_f - q[0.5])

        for pname, mask in (("prereg_11_15", bl_ext.ts_f < PREREG_END), ("ext_11_25", pd.Series(True, index=bl_ext.index))):
            sub = bl_ext[mask].copy()
            rows.append(dict(
                H=H, period=pname, model="quantile_10_90", n=len(sub),
                coverage=float(sub.q_cov.mean()), width=float(sub.q_width.mean()),
                below=float(sub.q_below.mean()), above=float(sub.q_above.mean()),
                MAE=float(sub.q50_ae.mean()),
            ))
            log(f"  [{pname}, n={len(sub):,}] 10-90% quantile: covers {sub.q_cov.mean():.1%} "
                f"(below {sub.q_below.mean():.1%}, above {sub.q_above.mean():.1%}), width {sub.q_width.mean():.0f} min")
            for name in preds:
                rows.append(dict(H=H, period=pname, model=name, n=len(sub), MAE=float(sub[f"ae_{name}"].mean())))
            log(f"  [{pname}] MAE: " + "  ".join(f"{k} {sub[f'ae_{k}'].mean():.2f}" for k in preds))

            for mdl in ("grad_boosting", "gb_slow_history", "gb_level"):
                for ref in ("level_anomaly", "hist_mean"):
                    lo, hi = boot_ci(sub, f"ae_{mdl}", f"ae_{ref}")
                    d = float(sub[f"ae_{mdl}"].mean() - sub[f"ae_{ref}"].mean())
                    days = sub.groupby(sub.ts_f.dt.normalize())[[f"ae_{mdl}", f"ae_{ref}"]].mean()
                    wins = int((days[f"ae_{mdl}"] < days[f"ae_{ref}"]).sum())
                    rows.append(dict(H=H, period=pname, model=f"{mdl}_minus_{ref}", n=len(sub),
                                     MAE=d, ci_lo=lo, ci_hi=hi, wins=wins, n_days=len(days)))
                    log(f"    {mdl} vs {ref}: {d:+.2f} (95% CI {lo:+.2f} .. {hi:+.2f}), wins {wins} of {len(days)} days")

        ae_cols = [c for c in bl_ext.columns if c.startswith("ae_")]
        by_day.append(bl_ext.groupby(bl_ext.ts_f.dt.normalize())[ae_cols].mean()
                      .assign(H=H, mode="blind", n=bl_ext.groupby(bl_ext.ts_f.dt.normalize()).size()).reset_index())

        if H == 3:
            out = pd.DataFrame({
                "hospital": bl_ext.hospital,
                "forecast_made": bl_ext.ts,
                "target_hour": bl_ext.ts_f,
                "wait_now": bl_ext.y0,
                "actual": bl_ext.y_f,
                "hist_mean": preds["hist_mean"],
                "rule": preds["level_anomaly"],
                "model": preds["grad_boosting"],
            })
            out.to_csv("results/blind_nov_predictions_3h.csv", index=False)
            for pname, mask in (("prereg_11_15", out.target_hour < PREREG_END), ("ext_11_25", pd.Series(True, index=out.index))):
                sub_o = out[mask]
                pw_lines.append(f"[{pname}] 3 h ahead blind test | MAE model {np.abs(sub_o.actual - sub_o.model).mean():.2f}, "
                                f"rule {np.abs(sub_o.actual - sub_o.rule).mean():.2f}, hist {np.abs(sub_o.actual - sub_o.hist_mean).mean():.2f}")
                pw_lines += pairwise(sub_o, ("model", "rule", "hist_mean", "wait_now"))
            open("results/blind_nov_pairwise.txt", "w").write("\n".join(pw_lines) + "\n")
            log("  " + "\n  ".join(pw_lines))

    pd.DataFrame(rows).to_csv("results/blind_nov_metrics.csv", index=False)
    pd.concat(by_day).to_csv("results/blind_nov_by_day.csv", index=False)
    open("results/blind_nov_summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
