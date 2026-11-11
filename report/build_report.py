"""Fill report_template.md with the current numbers and render .docx + .pdf.

Usage: python3 report/build_report.py [--prefix results/models_2026-10-30_v2] [--v1 results/models_2026-10-30]
                                      [--out ../Assignments/Report]
After new data: rerun train_models.py (and plot_results.py), then this script. Every number in the report comes
from the files below, so nothing has to be edited by hand.
  <prefix>_metrics.csv / _summary.txt / _by_hour.csv / _by_hospital.csv / _fig_models.png   (main run, v2)
  <v1>_metrics.csv / _ablation.csv / _fig_ablation.png                                      (all-feature run)
  data/processed/hourly_features_full.csv, results/baselines_horizon_full_2026-10-08.txt, results/checkpoint_fig.png
"""
import argparse, datetime as dt, re, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"persistence": "Persistence (baseline)", "hist_mean": "Historical mean (baseline)",
         "ols_calendar_only": "OLS, calendar + hospital only", "ols": "OLS", "ridge": "Ridge", "lasso": "Lasso",
         "random_forest": "Random forest", "grad_boosting": "Gradient boosting", "logistic": "Logistic regression"}
SHORT = {"Pamela Youde Nethersole Eastern Hospital": "PYNEH", "Alice Ho Miu Ling Nethersole Hospital": "AHNH"}


def f1(x):
    return f"{x:.1f}"


def f3(x):
    return f"{x:.3f}"


def short(h):
    return SHORT.get(h, h.replace(" Hospital", "").replace(" Medical Centre", ""))


def md_table(header, rows):
    # pandoc sizes pipe-table columns by the dash counts: give the model-name column ~2.5x the others
    out = ["| " + " | ".join(header) + " |", "|" + "-" * 30 + "|" + ("-" * 12 + "|") * (len(header) - 1)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def numbers(a):
    N = {"build_date": dt.date.today().strftime("%d %b %Y")}
    m = pd.read_csv(ROOT / f"{a.prefix}_metrics.csv")
    m1 = pd.read_csv(ROOT / f"{a.v1}_metrics.csv")
    abl = pd.read_csv(ROOT / f"{a.v1}_ablation.csv")
    summ = (ROOT / f"{a.prefix}_summary.txt").read_text()

    def reg(model, H, split="test", df=m, col="MAE"):
        return df[(df.task == "reg") & (df.model == model) & (df.H == H) & (df.split == split)][col].iloc[0]

    def cls(model, H, col, split="test", regime="A_all_eras", df=m):
        q = df[(df.task == "cls") & (df.model == model) & (df.H == H) & (df.split == split) & (df.thr == "0.5")]
        if model not in ("persistence", "hist_mean"):
            q = q[q.regime == regime]
        return q[col].iloc[0]

    # --- data facts
    h = pd.read_csv(ROOT / "data/processed/hourly_features_full.csv",
                    usecols=["hospital", "fmt", "hour_ts", "t45_min", "t45_over2h", "hr", "is_weekend", "is_holiday"],
                    parse_dates=["hour_ts"])
    N.update(n_rows=f"{len(h):,}", n_rows_new=f"{(h.fmt == 'new').sum():,}", n_hosp=h.hospital.nunique(),
             n_snapshots=f"{h.hour_ts.nunique():,}", date_from=h.hour_ts.min().strftime("%d %b %Y"),
             date_to=h.hour_ts.max().strftime("%d %b %Y"), n_holiday_hours=f"{int(h.is_holiday.sum()):,}")
    last = h.hour_ts.max().normalize()
    test_days = pd.date_range("2026-10-01", min(last, pd.Timestamp("2026-10-31")), freq="D")
    have = set(h[h.hour_ts >= "2026-10-01"].hour_ts.dt.normalize())
    gaps = [d for d in test_days if d not in have]
    N.update(n_gap_days=len(gaps), gap_days=", ".join(f"{d.day} Oct" for d in gaps) or "none")
    new = h[h.fmt == "new"]
    off = new.is_weekend.astype(bool) | new.is_holiday.astype(bool)
    wk, we = new[~off].groupby("hr").t45_min.mean(), new[off].groupby("hr").t45_min.mean()
    N.update(trough_min=round(wk.min()), trough_hr=f"{wk.idxmin():02d}", peak_min=round(wk.max()), peak_hr=f"{wk.idxmax():02d}",
             wkend_gap=round(wk[21] - we[21]))
    by_h = new.groupby("hospital").t45_min.mean()
    N.update(busiest=short(by_h.idxmax()), busiest_min=round(by_h.max()), least_busy=short(by_h.idxmin()),
             least_busy_min=round(by_h.min()))
    o = h[(h.hour_ts >= "2025-10-01") & (h.hour_ts < "2025-11-01")].groupby("fmt").t45_over2h.mean()
    N.update(oct25_old=round(100 * o["old"]), oct25_new=round(100 * o["new"]),
             pos_old=round(100 * h[h.fmt == "old"].t45_over2h.mean()),
             pos_new=round(100 * h[(h.fmt == "new") & (h.hour_ts < "2026-09-01")].t45_over2h.mean()))

    # --- split sizes
    r3 = re.search(r"REGRESSION.*H=3 h \| train ([\d,]+) val ([\d,]+) test ([\d,]+)", summ)
    c3 = re.search(r"CLASSIFICATION.*H=3 h, A_all_eras \| train ([\d,]+)", summ)
    N.update(n_train_reg=r3.group(1), n_val=r3.group(2), n_test=r3.group(3), n_train_cls=c3.group(1))

    # --- headline numbers
    for H in (3, 24):
        N[f"hm_mae{H}"], N[f"gb_mae{H}"] = f1(reg("hist_mean", H)), f1(reg("grad_boosting", H))
        N[f"hm_val{H}"], N[f"gb_val{H}"] = f1(reg("hist_mean", H, "val")), f1(reg("grad_boosting", H, "val"))
        N[f"hm_auc{H}"], N[f"gb_auc{H}"] = f3(cls("hist_mean", H, "AUC")), f3(cls("grad_boosting", H, "AUC"))
        N[f"hm_acc{H}"], N[f"gb_acc{H}"] = f3(cls("hist_mean", H, "acc")), f3(cls("grad_boosting", H, "acc"))
    N["gain_mae3"] = round(100 * (1 - reg("grad_boosting", 3) / reg("hist_mean", 3)))
    N["ols_mae3"], N["rf_mae3"] = f1(reg("ols", 3)), f1(reg("random_forest", 3))
    N["gb_v1_mae3"] = f1(reg("grad_boosting", 3, df=m1))
    N["auc_A3"] = f3(cls("grad_boosting", 3, "AUC"))
    N["auc_B3"] = f3(cls("grad_boosting", 3, "AUC", regime="B_new_only"))
    N["hm_acc_val"] = f3(cls("hist_mean", 3, "acc", split="val"))
    old = (ROOT / "results/baselines_horizon_full_2026-10-08.txt").read_text()
    N["hm_acc_old"] = re.search(r"horizon 3h.*?hist\s+n=\s*\d+\s+acc=([\d.]+)", old.split("[classification")[1], re.S).group(1)

    # --- ablation (v1)
    def ab(g, H):
        return abl[(abl.task == "reg") & (abl.group == g) & (abl.H == H)].delta.iloc[0]
    N.update(abl_hist3=f"+{ab('history', 3):.1f}", abl_cal3=f"+{ab('calendar', 3):.1f}", abl_hosp3=f"+{ab('hospital', 3):.1f}",
             abl_wx3=f"{ab('weather', 3):+.1f}", abl_air3=f"{ab('air', 3):+.1f}", abl_wx24_abs=f"{abs(ab('weather', 24)):.1f}")

    # --- error analysis (main run, 3 h)
    bh = pd.read_csv(ROOT / f"{a.prefix}_by_hour.csv"); bh = bh[bh.H == 3].set_index("hr_f")
    N.update(err_night=round(bh.loc[[0, 1, 2], "abs_err"].mean()), err_morning=round(bh.loc[[7, 8], "abs_err"].mean()))
    bp = pd.read_csv(ROOT / f"{a.prefix}_by_hospital.csv"); bp = bp[bp.H == 3].copy()
    bp["gain"] = bp.abs_err_hist - bp.abs_err
    bp = bp.sort_values("gain", ascending=False)
    N["top_gain"] = ", ".join(f"{short(r.hospital)} (−{r.gain:.0f} min)" for r in bp.head(3).itertuples())
    N["low_gain"] = ", ".join(f"{short(r.hospital)} (−{r.gain:.0f})" for r in bp.tail(3).itertuples())

    # --- tables
    rows = []
    for k in ["persistence", "hist_mean", "ols_calendar_only", "ols", "ridge", "lasso", "random_forest", "grad_boosting"]:
        name = f"**{NAMES[k]}**" if k == "grad_boosting" else NAMES[k]
        rows.append([name, f1(reg(k, 3, "val")), f1(reg(k, 3)), f3(reg(k, 3, col="R2")), f1(reg(k, 24, "val")), f1(reg(k, 24))])
    N["table_reg"] = md_table(["Model", "3 h val MAE", "3 h test MAE", "3 h test R²", "24 h val MAE", "24 h test MAE"], rows)
    rows = []
    for k in ["persistence", "hist_mean", "logistic", "random_forest", "grad_boosting"]:
        rows.append([NAMES[k], f3(cls(k, 3, "acc")), f3(cls(k, 3, "F1")), f3(cls(k, 3, "AUC")), f3(cls(k, 24, "acc")), f3(cls(k, 24, "AUC"))])
    rows.append(["Gradient boosting, new format only", f3(cls("grad_boosting", 3, "acc", regime="B_new_only")),
                 f3(cls("grad_boosting", 3, "F1", regime="B_new_only")), f3(cls("grad_boosting", 3, "AUC", regime="B_new_only")),
                 f3(cls("grad_boosting", 24, "acc", regime="B_new_only")), f3(cls("grad_boosting", 24, "AUC", regime="B_new_only"))])
    N["table_cls"] = md_table(["Model (test month)", "3 h acc", "3 h F1", "3 h AUC", "24 h acc", "24 h AUC"], rows)

    N.update(fig_eda=str(ROOT / "results/checkpoint_fig.png"), fig_models=str(ROOT / f"{a.prefix}_fig_models.png"),
             fig_ablation=str(ROOT / f"{a.v1}_fig_ablation.png"))
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/models_2026-10-30_v2")
    ap.add_argument("--v1", default="results/models_2026-10-30")
    ap.add_argument("--out", default=str(ROOT.parent / "Assignments" / "Report"))
    a = ap.parse_args()
    N = numbers(a)
    text = (ROOT / "report/report_template.md").read_text().format_map(N)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    md = out / "ELEC3544-Report-Group16-DRAFT.md"
    md.write_text(text)
    docx = md.with_suffix(".docx")
    subprocess.run(["pandoc", str(md), "-o", str(docx), "--reference-doc", str(ROOT / "report/reference.docx")]
                   if (ROOT / "report/reference.docx").exists() else ["pandoc", str(md), "-o", str(docx)], check=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(out), str(docx)],
                   check=True, capture_output=True)
    print(f"wrote {md.name}, {docx.name}, {docx.with_suffix('.pdf').name} in {out}")


if __name__ == "__main__":
    main()
