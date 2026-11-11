"""Models vs baselines at 3 h and 24 h ahead (proposal weeks 3-4).

Usage: python src/train_models.py [--in data/processed/hourly_features_full.csv] [--out results/models]
                                  [--horizons 3 24] [--fast]

Setup
  Origin t = the hour we forecast from; target = value at t+h for the same hospital.
  The hourly table is re-indexed to a full hourly grid per hospital, so every shift is an exact number of hours.
  Only information known at t is used: waiting-time history up to t, AQHI at t, weather of the PREVIOUS day
  (HKO daily values are not known until the day ends), and the calendar of the target hour (known in advance).
  Split by target time: train < 2026-09-01, validation = Sep 2026 (all tuning), test = Oct 2026 (untouched,
  collected after the proposal). HKO has not published October weather yet, so weather is NaN in the test month.

Tasks
  reg : cat 4/5 median wait in minutes, new feed format only (from 2025-10-13)
  cls : wait > 2 h; regime A = train on both formats with an era flag, regime B = new format only

Outputs (prefix --out): _summary.txt, _metrics.csv, _ablation.csv, _by_hospital.csv, _by_hour.csv
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from build_features import load_holidays  # noqa: E402

SWITCH = pd.Timestamp("2025-10-13 09:30")
VAL, TEST, END = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")
WEATHER = ["rf_mm", "rh", "temp_mean", "temp_max", "temp_min"]

GROUPS = {
    "calendar": ["hr_f", "dow_f", "is_weekend_f", "is_holiday_f", "month_f"],
    "hospital": ["hosp"],
    "history": ["y0", "y_l1", "y_l3", "y_d", "y_w", "roll6", "roll24", "lbl0"],
    "hist_mean": ["hist"],
    "air": ["aqhi0"],
    "weather": [f"{w}_prev" for w in WEATHER],
    "era": ["era"],
}
CATEGORICAL = ["hosp", "hr_f", "dow_f", "month_f"]


def load_grid(path, aux):
    h = pd.read_csv(path, parse_dates=["hour_ts"])
    # the switch hour can appear twice (old + new snapshot); keep the new-format row
    h = h.sort_values(["hospital", "hour_ts", "fmt"]).drop_duplicates(["hospital", "hour_ts"], keep="first")
    daily = (h.assign(date=h.hour_ts.dt.normalize()).groupby("date")[WEATHER].first())
    parts = []
    for hosp, g in h.groupby("hospital"):
        idx = pd.date_range(g.hour_ts.min(), g.hour_ts.max(), freq="h", name="hour_ts")
        g = g.set_index("hour_ts").reindex(idx).reset_index()
        g["hospital"] = hosp
        parts.append(g)
    grid = pd.concat(parts, ignore_index=True)
    hol = load_holidays(aux)
    grid["fmt"] = np.where(grid.hour_ts >= SWITCH, "new", "old")
    grid["date"] = grid.hour_ts.dt.normalize()
    prev = daily.add_suffix("_prev")
    prev.index = prev.index + pd.Timedelta(days=1)  # weather of day D-1 joined to day D
    grid = grid.join(prev, on="date")
    return grid, hol


def make_horizon(grid, hol, H):
    """One row per (hospital, origin t) with features at t and targets at t+H."""
    rows = []
    for hosp, g in grid.groupby("hospital", sort=False):
        g = g.sort_values("hour_ts").reset_index(drop=True)
        y, lbl = g["t45_min"], g["t45_over2h"]
        d = pd.DataFrame({"hospital": hosp, "ts": g.hour_ts})
        d["ts_f"] = d.ts + pd.Timedelta(hours=H)
        d["y_f"] = y.shift(-H); d["lbl_f"] = lbl.shift(-H)
        d["y0"] = y; d["y_l1"] = y.shift(1); d["y_l3"] = y.shift(3)
        d["y_d"] = y.shift(24 - H) if H < 24 else y  # same clock hour as the target, one day earlier
        d["y_w"] = y.shift(168 - H)                    # same clock hour as the target, one week earlier
        d["roll6"] = y.rolling(6, min_periods=3).mean()
        d["roll24"] = y.rolling(24, min_periods=12).mean()
        d["lbl0"] = lbl
        d["aqhi0"] = g["aqhi"]
        for w in WEATHER:
            d[f"{w}_prev"] = g[f"{w}_prev"]
        rows.append(d)
    X = pd.concat(rows, ignore_index=True)
    X["hr_f"] = X.ts_f.dt.hour; X["dow_f"] = X.ts_f.dt.dayofweek; X["month_f"] = X.ts_f.dt.month
    X["is_weekend_f"] = (X.dow_f >= 5).astype(int)
    X["is_holiday_f"] = X.ts_f.dt.date.isin(hol).astype(int)
    X["era"] = (X.ts_f >= SWITCH).astype(int)
    X["hosp"] = X.hospital.astype("category").cat.codes
    X["split"] = np.select([X.ts_f < VAL, X.ts_f < TEST, X.ts_f < END], ["train", "val", "test"], "drop")
    return X[X.split != "drop"]


def add_hist(X, target, train_mask):
    m = X[train_mask].groupby(["hosp", "dow_f", "hr_f"])[target].mean().rename("hist")
    m2 = X[train_mask].groupby(["hosp", "hr_f"])[target].mean().rename("hist2")
    X = X.drop(columns=["hist", "hist2"], errors="ignore").join(m, on=["hosp", "dow_f", "hr_f"]).join(m2, on=["hosp", "hr_f"])
    X["hist"] = X["hist"].fillna(X["hist2"])
    return X


def linear_prep(feats):
    cat = [f for f in feats if f in CATEGORICAL]
    num = [f for f in feats if f not in CATEGORICAL]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ])


def reg_metrics(y, p):
    ok = ~np.isnan(p); y, p = y[ok], p[ok]
    return dict(n=len(y), MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))


def cls_metrics(y, s, thr=0.5):
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(y), acc=float((pred == y).mean()), precision=prec, recall=rec,
                F1=2 * prec * rec / max(prec + rec, 1e-9), AUC=roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan)


def best_thr(y, s):
    grid = np.linspace(0.2, 0.8, 61)
    return float(grid[np.argmax([((s >= t).astype(int) == y).mean() for t in grid])])


def fit_predict(kind, model, feats, tr, ev_sets, y_col):
    t0 = time.time()
    if kind in ("ols", "ridge", "lasso", "logit"):
        est = make_pipeline(linear_prep(feats), model)
    elif kind == "rf":
        est = make_pipeline(SimpleImputer(strategy="median"), model)
    else:
        est = model.set_params(categorical_features=["hosp"] if "hosp" in feats else None)
    Xtr = tr[feats]
    if kind == "rf" and len(Xtr) > 150_000:  # keep RF tractable
        tr = tr.sample(150_000, random_state=0); Xtr = tr[feats]
    est.fit(Xtr, tr[y_col])
    out = {}
    for name, ev in ev_sets.items():
        out[name] = est.predict_proba(ev[feats])[:, 1] if hasattr(est, "predict_proba") else est.predict(ev[feats])
    return out, time.time() - t0, est


def hgb_reg(**kw):
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, early_stopping=False,
                                         categorical_features=["hosp"], random_state=0, **kw)


def hgb_cls(**kw):
    return HistGradientBoostingClassifier(max_iter=400, early_stopping=False, categorical_features=["hosp"],
                                          random_state=0, **kw)


def run_reg(X, H, log, rows, abl, fast):
    D = X[(X.ts_f >= SWITCH) & X.y_f.notna()].copy()
    D = add_hist(D, "y_f", D.split == "train")
    tr, va, te = (D[D.split == s] for s in ("train", "val", "test"))
    sets = {"val": va, "test": te}
    log(f"\n=== REGRESSION cat 4/5 wait (min), H={H} h | train {len(tr):,} val {len(va):,} test {len(te):,}")
    feats_all = [f for g, fs in GROUPS.items() if g != "era" for f in fs]
    feats_lin = [f for f in feats_all if f != "lbl0"]
    preds = {"persistence": {k: v["y0"].fillna(v["hist"]).values for k, v in sets.items()},
             "hist_mean": {k: v["hist"].values for k, v in sets.items()}}
    best_alpha = {}
    for kind, cls_, grid in (("ridge", Ridge, [0.1, 1, 10, 100]), ("lasso", Lasso, [0.01, 0.1, 1])):
        scores = {}
        for a in grid:
            p, _, _ = fit_predict(kind, cls_(alpha=a, **({"max_iter": 5000} if kind == "lasso" else {})), feats_lin, tr, {"val": va}, "y_f")
            scores[a] = mean_absolute_error(va.y_f, p["val"])
        best_alpha[kind] = min(scores, key=scores.get)
        log(f"  {kind} alpha on val: " + ", ".join(f"{a}->{s:.2f}" for a, s in scores.items()))
    specs = [("ols_calendar_only", "ols", LinearRegression(), ["hosp"] + GROUPS["calendar"]),
             ("ols", "ols", LinearRegression(), feats_lin),
             ("ridge", "ridge", Ridge(alpha=best_alpha["ridge"]), feats_lin),
             ("lasso", "lasso", Lasso(alpha=best_alpha["lasso"], max_iter=5000), feats_lin),
             ("random_forest", "rf", RandomForestRegressor(n_estimators=150 if not fast else 40, min_samples_leaf=20,
                                                          max_features=0.5, n_jobs=-1, random_state=0), feats_all)]
    hgb_grid = [dict(learning_rate=0.05, max_leaf_nodes=31), dict(learning_rate=0.05, max_leaf_nodes=63),
                dict(learning_rate=0.1, max_leaf_nodes=31)]
    hgb_scores = {}
    for i, kw in enumerate(hgb_grid if not fast else hgb_grid[:1]):
        p, _, _ = fit_predict("hgb", hgb_reg(**kw), feats_all, tr, {"val": va}, "y_f")
        hgb_scores[i] = mean_absolute_error(va.y_f, p["val"])
    kw = hgb_grid[min(hgb_scores, key=hgb_scores.get)]
    log(f"  HGB grid on val (MAE): {[round(v, 2) for v in hgb_scores.values()]} -> {kw}")
    specs.append(("grad_boosting", "hgb", hgb_reg(**kw), feats_all))
    for name, kind, model, feats in specs:
        p, secs, est = fit_predict(kind, model, feats, tr, sets, "y_f")
        preds[name] = p
        if name == "lasso":
            coef = est[-1].coef_; log(f"  lasso kept {int((np.abs(coef) > 1e-6).sum())}/{len(coef)} coefficients")
        log(f"  fitted {name} in {secs:.0f}s")
    for name, p in preds.items():
        for split, ev in sets.items():
            m = reg_metrics(ev.y_f.values, p[split])
            rows.append(dict(task="reg", H=H, model=name, split=split, **m))
            log(f"  {name:18s} {split:4s} MAE {m['MAE']:6.1f}  RMSE {m['RMSE']:6.1f}  R2 {m['R2']:6.3f}")
    # feature-group ablation on validation (drop one group, refit HGB)
    base = mean_absolute_error(va.y_f, preds["grad_boosting"]["val"])
    for g, fs in GROUPS.items():
        if g == "era":
            continue
        keep = [f for f in feats_all if f not in fs]
        p, _, _ = fit_predict("hgb", hgb_reg(**kw), keep, tr, {"val": va}, "y_f")
        d = mean_absolute_error(va.y_f, p["val"]) - base
        abl.append(dict(task="reg", H=H, group=g, metric="val MAE increase (min)", delta=d))
        log(f"  ablation drop {g:10s}: val MAE {base:.2f} -> {base + d:.2f} ({d:+.2f})")
    te = te.assign(pred=preds["grad_boosting"]["test"], hist_pred=preds["hist_mean"]["test"])
    return te


def run_cls(X, H, log, rows, abl, fast):
    D = X[X.lbl_f.notna()].copy()
    D["lbl_f"] = D.lbl_f.astype(int)
    feats_all = [f for fs in GROUPS.values() for f in fs]
    for regime in ("A_all_eras", "B_new_only"):
        R = D if regime == "A_all_eras" else D[D.ts_f >= SWITCH]
        R = add_hist(R, "lbl_f", (R.split == "train") & (R.ts_f >= SWITCH))  # base rate of the current era
        tr, va, te = (R[R.split == s] for s in ("train", "val", "test"))
        sets = {"val": va, "test": te}
        log(f"\n=== CLASSIFICATION wait > 2 h, H={H} h, {regime} | train {len(tr):,} (pos {tr.lbl_f.mean():.2f}) "
            f"val {len(va):,} (pos {va.lbl_f.mean():.2f}) test {len(te):,} (pos {te.lbl_f.mean():.2f})")
        feats = feats_all if regime == "A_all_eras" else [f for f in feats_all if f != "era"]
        preds = {}
        if regime == "A_all_eras":
            preds["persistence"] = {k: v.lbl0.fillna(v["hist"]).values for k, v in sets.items()}
            preds["hist_mean"] = {k: v["hist"].values for k, v in sets.items()}
        specs = [("logistic", "logit", LogisticRegression(max_iter=2000, C=1.0), feats),
                 ("random_forest", "rf", RandomForestClassifier(n_estimators=150 if not fast else 40, min_samples_leaf=20,
                                                               max_features=0.5, n_jobs=-1, random_state=0), feats),
                 ("grad_boosting", "hgb", hgb_cls(learning_rate=0.05, max_leaf_nodes=31), feats)]
        for name, kind, model, fs in specs:
            p, secs, _ = fit_predict(kind, model, fs, tr, sets, "lbl_f")
            preds[name] = p
            log(f"  fitted {name} in {secs:.0f}s")
        for name, p in preds.items():
            thr = best_thr(va.lbl_f.values, p["val"]) if name not in ("persistence",) else 0.5
            for split, ev in sets.items():
                for t_name, t in (("0.5", 0.5), ("val-tuned", thr)):
                    if name == "persistence" and t_name == "val-tuned":
                        continue
                    m = cls_metrics(ev.lbl_f.values, p[split], t)
                    rows.append(dict(task="cls", H=H, regime=regime, model=name, split=split, thr=t_name, thr_value=t, **m))
                    log(f"  {name:14s} {split:4s} thr {t_name:9s}({t:.2f}) acc {m['acc']:.3f} prec {m['precision']:.3f} "
                        f"rec {m['recall']:.3f} F1 {m['F1']:.3f} AUC {m['AUC']:.3f}")
        if regime == "A_all_eras":
            base = roc_auc_score(va.lbl_f, preds["grad_boosting"]["val"])
            for g, fs in GROUPS.items():
                keep = [f for f in feats if f not in fs]
                p, _, _ = fit_predict("hgb", hgb_cls(learning_rate=0.05, max_leaf_nodes=31), keep, tr, {"val": va}, "lbl_f")
                d = roc_auc_score(va.lbl_f, p["val"]) - base
                abl.append(dict(task="cls", H=H, group=g, metric="val AUC change", delta=d))
                log(f"  ablation drop {g:10s}: val AUC {base:.4f} -> {base + d:.4f} ({d:+.4f})")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/hourly_features_full.csv")
    ap.add_argument("--aux", default="data/raw/aux")
    ap.add_argument("--out", default="results/models")
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 24])
    ap.add_argument("--fast", action="store_true", help="smaller forests / grid, for a quick check")
    ap.add_argument("--drop", nargs="*", default=[], choices=list(GROUPS), help="feature groups to leave out")
    a = ap.parse_args()
    for g in a.drop:
        GROUPS.pop(g)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    grid, hol = load_grid(a.inp, a.aux)
    log(f"feature groups: {list(GROUPS)} (dropped: {a.drop or 'none'})")
    log(f"grid rows {len(grid):,} ({grid.t45_min.isna().mean():.1%} hours missing a snapshot), "
        f"range {grid.hour_ts.min()} .. {grid.hour_ts.max()}")
    rows, abl, by_h, by_hr = [], [], [], []
    for H in a.horizons:
        X = make_horizon(grid, hol, H)
        te = run_reg(X, H, log, rows, abl, a.fast)
        run_cls(X, H, log, rows, abl, a.fast)
        te["abs_err"] = (te.pred - te.y_f).abs(); te["abs_err_hist"] = (te.hist_pred - te.y_f).abs()
        bh = te.groupby("hospital")[["abs_err", "abs_err_hist", "y_f"]].mean().assign(H=H)
        by_h.append(bh.reset_index()); by_hr.append(te.groupby("hr_f")[["abs_err", "abs_err_hist", "y_f"]].mean().assign(H=H).reset_index())
        log(f"\n  test-month MAE by hospital, H={H} (grad boosting vs hist mean):")
        for hosp, r in bh.sort_values("abs_err", ascending=False).iterrows():
            log(f"    {hosp[:42]:42s} {r.abs_err:6.1f} vs {r.abs_err_hist:6.1f}  (mean wait {r.y_f:5.0f})")
    pd.DataFrame(rows).to_csv(f"{a.out}_metrics.csv", index=False)
    pd.DataFrame(abl).to_csv(f"{a.out}_ablation.csv", index=False)
    pd.concat(by_h).to_csv(f"{a.out}_by_hospital.csv", index=False)
    pd.concat(by_hr).to_csv(f"{a.out}_by_hour.csv", index=False)
    open(f"{a.out}_summary.txt", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}_*.csv / _summary.txt")


if __name__ == "__main__":
    main()
