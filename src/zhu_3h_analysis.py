import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_models as tm

OUT = Path("results")
PREFIX = OUT / "zy_3h"


def fit_one(kw, tr, va, feats):
    with threadpool_limits(limits=3):
        p, secs, _ = tm.fit_predict("hgb", tm.hgb_reg(**kw), feats, tr, {"val": va}, "y_f")
    return dict(**kw, val_MAE=mean_absolute_error(va.y_f, p["val"]), fit_seconds=secs)


def key(row):
    return (float(row["learning_rate"]), int(row["max_leaf_nodes"]),
            int(row["min_samples_leaf"]), float(row["l2_regularization"]))


def main():
    grid, hol = tm.load_grid("data/processed/hourly_features_full.csv", "data/raw/aux")
    X = tm.make_horizon(grid, hol, 3)
    reg = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
    reg = tm.add_hist(reg, "y_f", reg.split == "train")
    tr, va, te = (reg[reg.split == s].copy() for s in ("train", "val", "test"))
    feats = [f for g, fs in tm.GROUPS.items() if g not in ("era", "weather", "air") for f in fs]
    lines = [f"3 h regression tuning | train {len(tr):,} val {len(va):,} test {len(te):,}",
             "All 54 configurations are ranked on September validation; only the selected configuration is evaluated on October."]

    tuning_path = Path(f"{PREFIX}_tuning_validation.csv")
    records = pd.read_csv(tuning_path).to_dict("records") if tuning_path.exists() else []
    done = {key(r) for r in records}
    search = []
    for i, (lr, leaves, min_leaf, l2) in enumerate(itertools.product(
            [0.03, 0.05, 0.1], [31, 63, 127], [20, 50, 100], [0, 1]), 1):
        kw = dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf, l2_regularization=l2)
        if key(kw) not in done:
            search.append((i, kw))
    print(f"Resuming with {len(records)} completed and {len(search)} remaining configurations", flush=True)
    for start in range(0, len(search), 3):
        batch = search[start:start + 3]
        results = Parallel(n_jobs=len(batch))(delayed(fit_one)(kw, tr, va, feats) for _, kw in batch)
        records.extend(results)
        tuning = pd.DataFrame(records).sort_values(
            ["learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]).reset_index(drop=True)
        tuning.to_csv(tuning_path, index=False)
        for (i, _), r in zip(batch, results):
            print(f"[{i:02d}/54] lr={r['learning_rate']} leaves={r['max_leaf_nodes']} "
                  f"min_leaf={r['min_samples_leaf']} l2={r['l2_regularization']} val MAE={r['val_MAE']:.3f}", flush=True)
    tuning = pd.DataFrame(records)
    best = tuning.loc[tuning.val_MAE.idxmin()]
    kw = {k: (float(best[k]) if k in ("learning_rate", "l2_regularization") else int(best[k]))
          for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}
    lines.append(f"Selected parameters: {json.dumps(kw)}; val MAE {best.val_MAE:.3f}")
    p, _, _ = tm.fit_predict("hgb", tm.hgb_reg(**kw), feats, tr, {"val": va, "test": te}, "y_f")
    val_mae = mean_absolute_error(va.y_f, p["val"])
    test_mae = mean_absolute_error(te.y_f, p["test"])
    hist_val = mean_absolute_error(va.y_f, va["hist"])
    hist_test = mean_absolute_error(te.y_f, te["hist"])
    pd.DataFrame([
        dict(model="hist_mean", split="val", MAE=hist_val),
        dict(model="tuned_hgb", split="val", MAE=val_mae),
        dict(model="hist_mean", split="test", MAE=hist_test),
        dict(model="tuned_hgb", split="test", MAE=test_mae),
    ]).to_csv(f"{PREFIX}_regression_metrics.csv", index=False)
    lines.append(f"Historical mean: val {hist_val:.3f}, test {hist_test:.3f}")
    lines.append(f"Tuned HGB: val {val_mae:.3f}, test {test_mae:.3f}")

    err = te[["hospital", "hr_f", "y_f", "hist"]].copy()
    err["abs_err"] = np.abs(te.y_f.values - p["test"])
    err["abs_err_hist"] = np.abs(te.y_f.values - te["hist"].values)
    by_hospital = err.groupby("hospital")[["abs_err", "abs_err_hist", "y_f"]].mean().reset_index()
    by_hour = err.groupby("hr_f")[["abs_err", "abs_err_hist", "y_f"]].mean().reset_index()
    by_hospital.to_csv(f"{PREFIX}_by_hospital.csv", index=False)
    by_hour.to_csv(f"{PREFIX}_by_hour.csv", index=False)

    plot_h = by_hospital.sort_values("abs_err_hist")
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=180)
    y = np.arange(len(plot_h))
    ax.barh(y - 0.2, plot_h.abs_err_hist, height=0.38, label="Historical mean", color="#9aa3ad")
    ax.barh(y + 0.2, plot_h.abs_err, height=0.38, label="Tuned HGB", color="#386b8c")
    ax.set_yticks(y, plot_h.hospital, fontsize=7)
    ax.set_xlabel("October test MAE (minutes)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_error_by_hospital.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=180)
    ax.plot(by_hour.hr_f, by_hour.abs_err_hist, marker="o", ms=3, label="Historical mean", color="#7f8892")
    ax.plot(by_hour.hr_f, by_hour.abs_err, marker="o", ms=3, label="Tuned HGB", color="#386b8c")
    ax.set(xticks=range(0, 24, 2), xlabel="Target hour", ylabel="October test MAE (minutes)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_error_by_hour.png", bbox_inches="tight")
    plt.close(fig)

    cls = X[X.lbl_f.notna()].copy()
    cls["lbl_f"] = cls.lbl_f.astype(int)
    cls = tm.add_hist(cls, "lbl_f", (cls.split == "train") & (cls.ts_f >= tm.SWITCH))
    ctr, cva, cte = (cls[cls.split == s].copy() for s in ("train", "val", "test"))
    cls_feats = [f for g, fs in tm.GROUPS.items() if g not in ("weather", "air") for f in fs]
    probs, secs, _ = tm.fit_predict("hgb", tm.hgb_cls(learning_rate=0.05, max_leaf_nodes=31),
                                    cls_feats, ctr, {"val": cva, "test": cte}, "lbl_f")
    iso = IsotonicRegression(out_of_bounds="clip").fit(probs["val"], cva.lbl_f)
    calibrated = {s: iso.transform(probs[s]) for s in ("val", "test")}
    cal_rows = []
    curve_rows = []
    for split, ev in (("val", cva), ("test", cte)):
        for method, score in (("uncalibrated", probs[split]), ("isotonic", calibrated[split])):
            cal_rows.append(dict(split=split, method=method, Brier=brier_score_loss(ev.lbl_f, score),
                                 AUC=roc_auc_score(ev.lbl_f, score), accuracy=((score >= 0.5) == ev.lbl_f).mean()))
            frac, mean_pred = calibration_curve(ev.lbl_f, score, n_bins=10, strategy="quantile")
            curve_rows.extend(dict(split=split, method=method, mean_predicted=mp, observed_fraction=of)
                              for mp, of in zip(mean_pred, frac))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(f"{PREFIX}_calibration_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(f"{PREFIX}_calibration_curve.csv", index=False)
    val_cal = cal[cal.split == "val"].set_index("method")
    chosen = "isotonic" if val_cal.loc["isotonic", "Brier"] < val_cal.loc["uncalibrated", "Brier"] else "uncalibrated"
    lines.append(f"Calibration selected on validation Brier score: {chosen}")
    for r in cal_rows:
        lines.append(f"  {r['split']:4s} {r['method']:12s} Brier {r['Brier']:.4f} AUC {r['AUC']:.4f} accuracy {r['accuracy']:.4f}")

    curves = pd.DataFrame(curve_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=180)
    for ax, split in zip(axes, ("val", "test")):
        for method, color in (("uncalibrated", "#7f8892"), ("isotonic", "#386b8c")):
            c = curves[(curves.split == split) & (curves.method == method)]
            ax.plot(c.mean_predicted, c.observed_fraction, marker="o", ms=3, label=method, color=color)
        ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction > 2 h",
               title="September validation" if split == "val" else "October test", xlim=(0, 1), ylim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_calibration.png", bbox_inches="tight")

    lines.append(f"Largest test MAE hour: {int(by_hour.loc[by_hour.abs_err.idxmax(), 'hr_f']):02d}:00 "
                 f"({by_hour.abs_err.max():.2f} min)")
    gains = by_hospital.assign(gain=by_hospital.abs_err_hist - by_hospital.abs_err).sort_values("gain", ascending=False)
    lines.append("Largest hospital gains: " + "; ".join(f"{r.hospital} {r.gain:.1f} min" for _, r in gains.head(3).iterrows()))
    (OUT / "zy_3h_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
