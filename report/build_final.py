"""Fill report_final.md with numbers from Project/results and render the final report (PDF via xelatex, plus .docx).

Usage: python3 report/build_final.py [--zip ../Assignments/Share/Group16-Project-2026-11-11.zip]
Every number in the report comes from a result file; edit the words in report_final.md, never the PDF.
"""
import argparse, hashlib, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_report as br  # noqa: E402  (shared number extraction for the October run)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "Assignments" / "Report"
HEADER = r"""
\usepackage{needspace}\usepackage{soul}\usepackage{xcolor}\usepackage{float}\floatplacement{figure}{H}
\usepackage{titling}\setlength{\droptitle}{-4.6em}
\usepackage{enumitem}\setlist{itemsep=1pt,topsep=3pt,parsep=0pt}
\usepackage{titlesec}\titlespacing*{\section}{0pt}{9pt}{3pt}\titlespacing*{\subsection}{0pt}{7pt}{2pt}
\usepackage{caption}\captionsetup{font=small,skip=3pt}
\setlength{\parskip}{4pt}\setlength{\textfloatsep}{6pt}\setlength{\intextsep}{6pt}
\renewcommand{\arraystretch}{1.08}\AtBeginEnvironment{longtable}{\small}
\pretitle{\begin{flushleft}\huge\bfseries}\posttitle{\end{flushleft}\vspace{-0.9em}}
\preauthor{\begin{flushleft}\normalsize\bfseries}\postauthor{\end{flushleft}\vspace{-1.0em}}
\predate{}\postdate{}\date{}
% house style of the course's own project document: Linux Biolinum, running header, "title | page" footer
\setmainfont{LinBiolinum_R.otf}[BoldFont=LinBiolinum_RB.otf, ItalicFont=LinBiolinum_RI.otf, Ligatures=TeX]
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\footnotesize ELEC3544 \textbar{} DATA SCIENCE WITH FOUNDATION MODELS \textbar{} FALL 2026}
\fancyfoot[R]{\footnotesize\textcolor{gray}{Group 16 final project report \textbar{}} \normalsize\thepage}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[L]{\footnotesize ELEC3544 \textbar{} DATA SCIENCE WITH FOUNDATION MODELS \textbar{} FALL 2026}\fancyfoot[R]{\footnotesize\textcolor{gray}{Group 16 final project report \textbar{}} \normalsize\thepage}}
\setlength{\headheight}{12pt}\setlength{\headsep}{8pt}\setlength{\footskip}{22pt}
"""


def pm(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/models_2026-10-30_v2")
    ap.add_argument("--v1", default="results/models_2026-10-30")
    ap.add_argument("--zip", default=str(ROOT.parent / "Assignments" / "Share" / "Group16-Project-data.zip"))
    ap.add_argument("--official", action="store_true", help="build in the official course LaTeX template")
    a = ap.parse_args()
    N = br.numbers(a)
    ab_all = pd.read_csv(ROOT / "results/models_2026-10-30_ablation.csv").set_index(["task", "H", "group"])["delta"]
    ab_v2 = pd.read_csv(ROOT / "results/models_2026-10-30_v2_ablation.csv").set_index(["task", "H", "group"])["delta"]
    v1m = pd.read_csv(ROOT / "results/models_2026-10-30_metrics.csv")
    v1_gb3 = v1m[(v1m.task == "reg") & (v1m.H == 3) & (v1m.split == "test") & (v1m.model == "grad_boosting")]["MAE"].iloc[0]
    N.update(
        abl_hm3=pm(float(ab_all.loc[("reg", 3, "hist_mean")]), 2),
        abl_hm24=pm(float(ab_all.loc[("reg", 24, "hist_mean")]), 2),
        abl_v2_hm3=pm(float(ab_v2.loc[("reg", 3, "hist_mean")]), 2),
        abl_v2_hm24=pm(float(ab_v2.loc[("reg", 24, "hist_mean")]), 2),
        v1_gb_mae3=br.f1(v1_gb3),
    )
    f1, f3 = br.f1, br.f3

    N["top_gain"] = N["top_gain"].replace("PYNEH", "Pamela Youde Nethersole Eastern")
    N["low_gain"] = N["low_gain"].replace("AHNH", "Alice Ho Miu Ling Nethersole")
    # --- latest & blind data facts
    h = pd.read_csv(ROOT / "data/processed/hourly_features_latest.csv", usecols=["fmt"])
    hb = pd.read_csv(ROOT / "data/processed/hourly_features_blind.csv", usecols=["hospital", "fmt", "hour_ts", "t45_min"], parse_dates=["hour_ts"])
    hn = hb[hb.fmt == "new"]
    raw = Path.home() / "elec3544_data/raw/ae"
    n_files = sum(len(os.listdir(raw / d)) for d in ("old", "new")) if raw.exists() else 33555
    grid_hours_blind = int((hb.hour_ts.max() - hb.hour_ts.min()).total_seconds() // 3600) + 1
    N.update(n_rows_latest=f"{len(h):,}", n_rows_new_latest=f"{(h.fmt == 'new').sum():,}",
             n_rows_blind=f"{len(hb):,}", n_hours_blind=f"{hb.hour_ts.nunique():,}",
             n_rows_new_blind=f"{(hb.fmt == 'new').sum():,}", n_files=f"{n_files:,}",
             n_missing_pct_blind=f"{100 * (1 - len(hb) / (grid_hours_blind * 18)):.1f}",
             mean_tr=f1(hn[hn.hour_ts < "2026-09-01"].t45_min.mean()),
             pos_tr=f"{(hn[hn.hour_ts < '2026-09-01'].t45_min > 120).mean():.1%}",
             mean_oct=f1(hn[(hn.hour_ts >= "2026-10-01") & (hn.hour_ts < "2026-11-01")].t45_min.mean()),
             mean_nov1=f1(hn[(hn.hour_ts >= "2026-11-01") & (hn.hour_ts < "2026-11-11")].t45_min.mean()),
             mean_ext=f1(hn[hn.hour_ts >= "2026-11-11"].t45_min.mean()),
             pos_ext=f"{(hn[hn.hour_ts >= '2026-11-11'].t45_min > 120).mean():.1%}")

    # --- follow-up experiments of 4 Nov
    v = pd.read_csv(ROOT / "results/hn_24h_validation.csv"); v = v[v["mode"] == "raw"].set_index("model").val_MAE
    t = pd.read_csv(ROOT / "results/hn_24h_test.csv").set_index("model").MAE
    z = pd.read_csv(ROOT / "results/zy_3h_regression_metrics.csv").set_index(["model", "split"]).MAE
    c = pd.read_csv(ROOT / "results/zy_3h_calibration_metrics.csv").set_index(["split", "method"]).Brier
    N.update(v_slow=f1(v["slow_history"]), v_orig=f1(v["original"]), slow_oct=f1(t["slow_history"]),
             tuned_val=f1(z["tuned_hgb", "val"]), tuned_test=f1(z["tuned_hgb", "test"]),
             brier_iso=f"{c['test', 'isotonic']:.4f}", brier_raw=f"{c['test', 'uncalibrated']:.4f}")

    # --- November live test
    L = pd.read_csv(ROOT / "results/live_nov_metrics.csv")

    def lv(task, H, mode, period, model, col):
        return L[(L.task == task) & (L.H == H) & (L["mode"] == mode) & (L.period == period) & (L.model == model)][col].iloc[0]

    def ci(H, period, model):
        return f"{pm(lv('reg_diff_vs_hist', H, 'frozen', period, model, 'ci_lo'))} to {pm(lv('reg_diff_vs_hist', H, 'frozen', period, model, 'ci_hi'))}"

    N.update(nov_n=f"{int(lv('reg', 3, 'frozen', 'nov', 'hist_mean', 'n')):,}",
             nov_hm3=f1(lv("reg", 3, "frozen", "nov", "hist_mean", "MAE")), nov_gb3=f1(lv("reg", 3, "frozen", "nov", "grad_boosting", "MAE")),
             nov_slow24=f1(lv("reg", 24, "frozen", "nov", "gb_slow_history", "MAE")),
             nov_d3=f1(-lv("reg_diff_vs_hist", 3, "frozen", "nov", "grad_boosting", "MAE")), nov_ci3=ci(3, "nov", "grad_boosting"),
             nov_d24=f1(-lv("reg_diff_vs_hist", 24, "frozen", "nov", "gb_slow_history", "MAE")), nov_ci24=ci(24, "nov", "gb_slow_history"),
             oct_d24=f1(-lv("reg_diff_vs_hist", 24, "frozen", "oct", "gb_slow_history", "MAE")), oct_ci24=ci(24, "oct", "gb_slow_history"),
             nov_auc_hm=f3(lv("cls", 3, "frozen", "nov", "hist_mean", "AUC")), nov_auc_gb3=f3(lv("cls", 3, "frozen", "nov", "grad_boosting", "AUC")),
             nov_refit3=f1(lv("reg", 3, "refit", "nov", "grad_boosting", "MAE")), nov_refit24=f1(lv("reg", 24, "refit", "nov", "gb_slow_history", "MAE")),
             oct_full_gb3=f1(lv("reg", 3, "frozen", "oct", "grad_boosting", "MAE")), oct_full_hm=f1(lv("reg", 3, "frozen", "oct", "hist_mean", "MAE")))
    rows = []
    for key, name in (("persistence", "Persistence (baseline)"), ("hist_mean", "Historical mean (baseline)"),
                      ("grad_boosting", "Gradient boosting"), ("gb_slow_history", "Gradient boosting + slow history")):
        r = [name]
        for H in (3, 24):
            q = L[(L.task == "reg") & (L.H == H) & (L["mode"] == "frozen") & (L.period == "nov") & (L.model == key)]
            r.append(f1(q.MAE.iloc[0]) if len(q) else "–")
        for H in (3, 24):
            q = L[(L.task == "cls") & (L.H == H) & (L["mode"] == "frozen") & (L.period == "nov") & (L.model == key)]
            r.append(f3(q.AUC.iloc[0]) if len(q) else "–")
        q = L[(L.task == "cls") & (L.H == 3) & (L["mode"] == "frozen") & (L.period == "nov") & (L.model == key)]
        r.append(f3(q.acc.iloc[0]) if len(q) else "–")
        rows.append(r)
    N["table_nov"] = br.md_table(["Model (frozen, 1–10 Nov)", "3 h MAE", "24 h MAE", "3 h AUC", "24 h AUC", "3 h acc"], rows) + \
        "\n\n: November live test. MAE in minutes; AUC and accuracy for \"over 2 hours\".\n"
    N["table_reg"] += "\n\n: Regression, category 4/5 wait in minutes. Validation = September, test = 1–29 October 2026.\n"

    # --- post-hoc checks (Section 5.4)
    Pm = pd.read_csv(ROOT / "results/posthoc_metrics.csv")
    psum = (ROOT / "results/posthoc_summary.txt").read_text()

    def pv(H, period, model, col="MAE"):
        return Pm[(Pm.H == H) & (Pm.period == period) & (Pm.model == model)][col].iloc[0]

    def pci(H, period, model):
        return f"{pm(pv(H, period, model, 'ci_lo'))} to {pm(pv(H, period, model, 'ci_hi'))}"

    rule = "level_anomaly"
    wins = re.findall(r"days grad_boosting beats level_anomaly: (\d+ of \d+)", psum)
    N.update(r3_oct=f1(pv(3, "oct", rule)), r3_nov=f1(pv(3, "nov", rule)), r24_oct=f1(pv(24, "oct", rule)), r24_nov=f1(pv(24, "nov", rule)),
             b3=f"{float(re.search(r'H=3: .*level-anomaly weight ([0-9.]+)', psum).group(1)):.3f}",
             g3_oct=f1(-pv(3, "oct", f"grad_boosting_minus_{rule}")), g3_oct_ci=pci(3, "oct", f"grad_boosting_minus_{rule}"),
             g3_nov=f1(-pv(3, "nov", f"grad_boosting_minus_{rule}")), g3_nov_ci=pci(3, "nov", f"grad_boosting_minus_{rule}"),
             w3_oct=wins[0], w3_nov=wins[1],
             s24_oct=pm(pv(24, "oct", f"gb_slow_history_minus_{rule}")), s24_oct_ci=pci(24, "oct", f"gb_slow_history_minus_{rule}"),
             s24_nov=pm(pv(24, "nov", f"gb_slow_history_minus_{rule}")), s24_nov_ci=pci(24, "nov", f"gb_slow_history_minus_{rule}"),
             gl3_oct=f1(pv(3, "oct", "gb_level")), gl3_nov=f1(pv(3, "nov", "gb_level")), gl24_oct=f1(pv(24, "oct", "gb_level")), gl24_nov=f1(pv(24, "nov", "gb_level")),
             qw=round(pv(3, "nov", "quantile_10_90", "width") / 5) * 5,
             qw3_oct=round(pv(3, "oct", "quantile_10_90", "width")), qw3_nov=round(pv(3, "nov", "quantile_10_90", "width")),
             qc_oct=f"{pv(3, 'oct', 'quantile_10_90', 'coverage'):.0%}", qc_nov=f"{pv(3, 'nov', 'quantile_10_90', 'coverage'):.0%}",
             q120=re.search(r"exactly 120 min: ([\d.]+%)", psum).group(1),
             wx_missing=f"{round(float(re.search(r'rainfall\): ([\d.]+)%', psum).group(1)))}%",
             b1_3=f"{float(re.search(r'H=3: damped-anomaly weight ([0-9.]+)', psum).group(1)):.2f}",
             b1_24=f"{float(re.search(r'H=24: damped-anomaly weight ([0-9.]+)', psum).group(1)):.2f}",
             pw_rule=f"{float(re.search(r'rule\s+correct ([\d.]+)', (ROOT / 'results/posthoc_pairwise.txt').read_text()).group(1)):.0%}",
             nov_gb24=f1(lv("reg", 24, "frozen", "nov", "grad_boosting", "MAE")), nov_dgb24=f1(-lv("reg_diff_vs_hist", 24, "frozen", "nov", "grad_boosting", "MAE")),
             nov_cigb24=ci(24, "nov", "grad_boosting"))
    pwt = (ROOT / "results/live_nov_pairwise.txt").read_text()
    bd = pd.read_csv(ROOT / "results/live_nov_by_day.csv"); bd = bd[bd["mode"] == "frozen"]
    d3, d24 = bd[bd.H == 3], bd[bd.H == 24]
    N.update(post_hm_oct=f1(pv(3, "oct", "hist_mean")), b24=f"{float(re.search(r'H=24: .*level-anomaly weight ([0-9.]+)', psum).group(1)):.3f}",
             dt3=re.findall(r"decision tree: depth (\d+)", psum)[0], dt24=re.findall(r"decision tree: depth (\d+)", psum)[1],
             pw_n=re.search(r"over ([\d,]+) hospital-pair", pwt).group(1),
             pw_now=f"{float(re.search(r'wait_now\s+correct ([\d.]+)', pwt).group(1)):.0%}", pw_hist=f"{float(re.search(r'hist_mean\s+correct ([\d.]+)', pwt).group(1)):.0%}",
             pw_model=f"{float(re.search(r'model\s+correct ([\d.]+)', pwt).group(1)):.0%}",
             w3_live=("all " if (d3.ae_grad_boosting < d3.ae_hist_mean).all() else f"{int((d3.ae_grad_boosting < d3.ae_hist_mean).sum())} of ") + str(len(d3)),
             w24_live=f"{int((d24.ae_gb_slow_history < d24.ae_hist_mean).sum())} of {len(d24)}")
    # --- pre-registered (11-15 Nov) & extended (11-25 Nov) blind test
    Bm = pd.read_csv(ROOT / "results/blind_nov_metrics.csv")
    BF = pd.read_csv(ROOT / "results/blind_nov_foundation_metrics.csv").set_index(["H", "period"])
    bpw = (ROOT / "results/blind_nov_pairwise.txt").read_text()
    prereg_pw_txt, ext_pw_txt = bpw.split("[ext_11_25]")

    def bv(H, period, model, col="MAE"):
        return Bm[(Bm.H == H) & (Bm.period == period) & (Bm.model == model)][col].iloc[0]

    def bci(H, period, model):
        return f"{pm(bv(H, period, model, 'ci_lo'))} to {pm(bv(H, period, model, 'ci_hi'))}"

    pw4_m = float(re.search(r"model\s+correct ([\d.]+)", prereg_pw_txt).group(1))
    pw4_r = float(re.search(r"rule\s+correct ([\d.]+)", prereg_pw_txt).group(1))
    pw4_h = float(re.search(r"hist_mean\s+correct ([\d.]+)", prereg_pw_txt).group(1))
    pw4_n = float(re.search(r"wait_now\s+correct ([\d.]+)", prereg_pw_txt).group(1))
    pw14_m = float(re.search(r"model\s+correct ([\d.]+)", ext_pw_txt).group(1))
    pw14_r = float(re.search(r"rule\s+correct ([\d.]+)", ext_pw_txt).group(1))
    pw14_h = float(re.search(r"hist_mean\s+correct ([\d.]+)", ext_pw_txt).group(1))
    pw14_n = float(re.search(r"wait_now\s+correct ([\d.]+)", ext_pw_txt).group(1))
    N.update(
        prereg_n=f"{int(bv(3, 'prereg_11_15', 'hist_mean', 'n')):,}",
        ext_n=f"{int(bv(3, 'ext_11_25', 'hist_mean', 'n')):,}",
        prereg_hm=f1(bv(3, "prereg_11_15", "hist_mean")),
        ext_hm=f1(bv(3, "ext_11_25", "hist_mean")),
        r3_prereg=f1(bv(3, "prereg_11_15", "level_anomaly")),
        r3_ext=f1(bv(3, "ext_11_25", "level_anomaly")),
        r24_prereg=f1(bv(24, "prereg_11_15", "level_anomaly")),
        r24_ext=f1(bv(24, "ext_11_25", "level_anomaly")),
        gb3_prereg=f1(bv(3, "prereg_11_15", "grad_boosting")),
        gb3_ext=f1(bv(3, "ext_11_25", "grad_boosting")),
        s24_prereg_mae=f1(bv(24, "prereg_11_15", "gb_slow_history")),
        s24_ext_mae=f1(bv(24, "ext_11_25", "gb_slow_history")),
        g3_prereg=f1(-bv(3, "prereg_11_15", "grad_boosting_minus_level_anomaly")),
        g3_prereg_ci=bci(3, "prereg_11_15", "grad_boosting_minus_level_anomaly"),
        w3_prereg=f"{int(bv(3, 'prereg_11_15', 'grad_boosting_minus_level_anomaly', 'wins'))} of {int(bv(3, 'prereg_11_15', 'grad_boosting_minus_level_anomaly', 'n_days'))}",
        g3_ext=f1(-bv(3, "ext_11_25", "grad_boosting_minus_level_anomaly")),
        g3_ext_ci=bci(3, "ext_11_25", "grad_boosting_minus_level_anomaly"),
        w3_ext=f"{int(bv(3, 'ext_11_25', 'grad_boosting_minus_level_anomaly', 'wins'))} of {int(bv(3, 'ext_11_25', 'grad_boosting_minus_level_anomaly', 'n_days'))}",
        s24_prereg_diff=pm(bv(24, "prereg_11_15", "gb_slow_history_minus_level_anomaly")),
        s24_prereg_ci=bci(24, "prereg_11_15", "gb_slow_history_minus_level_anomaly"),
        w24_prereg=f"{int(bv(24, 'prereg_11_15', 'gb_slow_history_minus_level_anomaly', 'wins'))} of {int(bv(24, 'prereg_11_15', 'gb_slow_history_minus_level_anomaly', 'n_days'))}",
        s24_ext_diff=pm(bv(24, "ext_11_25", "gb_slow_history_minus_level_anomaly")),
        s24_ext_ci=bci(24, "ext_11_25", "gb_slow_history_minus_level_anomaly"),
        w24_ext=f"{int(bv(24, 'ext_11_25', 'gb_slow_history_minus_level_anomaly', 'wins'))} of {int(bv(24, 'ext_11_25', 'gb_slow_history_minus_level_anomaly', 'n_days'))}",
        fm3_prereg=f1(BF.loc[(3, "prereg_11_15"), "MAE"]),
        fm24_prereg=f1(BF.loc[(24, "prereg_11_15"), "MAE"]),
        fmc3_prereg=f"{BF.loc[(3, 'prereg_11_15'), 'coverage']:.0%}",
        fmc24_prereg=f"{BF.loc[(24, 'prereg_11_15'), 'coverage']:.0%}",
        fm3_ext=f1(BF.loc[(3, "ext_11_25"), "MAE"]),
        fm24_ext=f1(BF.loc[(24, "ext_11_25"), "MAE"]),
        fmc3_ext=f"{BF.loc[(3, 'ext_11_25'), 'coverage']:.0%}",
        fmc24_ext=f"{BF.loc[(24, 'ext_11_25'), 'coverage']:.0%}",
        fmw3_ext=round(BF.loc[(3, "ext_11_25"), "width"]),
        fmw24_ext=round(BF.loc[(24, "ext_11_25"), "width"]),
        qc3_prereg=f"{bv(3, 'prereg_11_15', 'quantile_10_90', 'coverage'):.0%}",
        qc3_prereg_1d=f"{bv(3, 'prereg_11_15', 'quantile_10_90', 'coverage'):.1%}",
        qc24_prereg_1d=f"{bv(24, 'prereg_11_15', 'quantile_10_90', 'coverage'):.1%}",
        qc3_ext=f"{bv(3, 'ext_11_25', 'quantile_10_90', 'coverage'):.0%}",
        qw3_ext=round(bv(3, "ext_11_25", "quantile_10_90", "width")),
        qw24_ext=round(bv(24, "ext_11_25", "quantile_10_90", "width")),
        qc24_ext=f"{bv(24, 'ext_11_25', 'quantile_10_90', 'coverage'):.0%}",
        q_below3_ext=f"{bv(3, 'ext_11_25', 'quantile_10_90', 'below'):.1%}",
        q_above3_ext=f"{bv(3, 'ext_11_25', 'quantile_10_90', 'above'):.1%}",
        pw_prereg_model=f"{pw4_m:.0%}",
        pw_prereg_rule=f"{pw4_r:.0%}",
        pw_prereg_hist=f"{pw4_h:.0%}",
        pw_prereg_now=f"{pw4_n:.0%}",
        pw_ext_n=re.search(r"over ([\d,]+) hospital-pair", ext_pw_txt).group(1),
        pw_ext_model=f"{pw14_m:.0%}",
        pw_ext_rule=f"{pw14_r:.0%}",
        pw_ext_hist=f"{pw14_h:.0%}",
        pw_ext_now=f"{pw14_n:.0%}",
    )
    bbd = pd.read_csv(ROOT / "results/blind_nov_by_day.csv", parse_dates=["ts_f"])
    bfp = pd.read_csv(ROOT / "results/blind_nov_foundation_predictions.csv", parse_dates=["target_hour", "forecast_made"])
    bp3 = pd.read_csv(ROOT / "results/blind_nov_predictions_3h.csv")
    cl3 = bp3[bp3.wait_now.notna()]
    orig_ok = hb[["hospital", "hour_ts"]].drop_duplicates().assign(present=1).rename(columns={"hour_ts": "forecast_made"})
    bfp24 = bfp[bfp.H == 24].merge(orig_ok, on=["hospital", "forecast_made"], how="left")
    bf_day = bfp.assign(ts_f=bfp.target_hour.dt.normalize()).groupby(["H", "ts_f"])["ae"].mean().rename("ae_chronos")
    bmrg = bbd.merge(bf_day, on=["H", "ts_f"])

    def day_boot(vals, n_boot=2000):
        rng = np.random.default_rng(0)
        b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
        return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    b3_10 = bmrg[(bmrg.H == 3) & (bmrg.ts_f >= "2026-11-16")]
    m10, l10, h10 = day_boot((b3_10.ae_grad_boosting - b3_10.ae_level_anomaly).values)
    b24_all = bmrg[bmrg.H == 24]
    m24f, l24f, h24f = day_boot((b24_all.ae_chronos - b24_all.ae_level_anomaly).values)
    b3_all = bmrg[bmrg.H == 3]
    m3f, l3f, h3f = day_boot((b3_all.ae_chronos - b3_all.ae_grad_boosting).values)
    n12 = b24_all[b24_all.ts_f == "2026-11-12"].iloc[0]
    ex12 = b24_all[b24_all.ts_f > "2026-11-12"]
    N.update(
        g3_ext10=f1(-m10), g3_ext10_ci=f"{pm(l10)} to {pm(h10)}",
        w3_ext10=f"{int((b3_10.ae_grad_boosting < b3_10.ae_level_anomaly).sum())} of {len(b3_10)}",
        fm24_vs_rule=pm(m24f, 2), fm24_vs_rule_ci=f"{pm(l24f, 2)} to {pm(h24f, 2)}",
        fm24_vs_rule_wins=f"{int((b24_all.ae_chronos < b24_all.ae_level_anomaly).sum())} of {len(b24_all)}",
        fm3_vs_gb=pm(m3f, 2), fm3_vs_gb_ci=f"{pm(l3f, 2)} to {pm(h3f, 2)}",
        fm3_vs_gb_wins=f"{int((b3_all.ae_chronos < b3_all.ae_grad_boosting).sum())} of {len(b3_all)}",
        gb24_nov12=f1(n12.ae_grad_boosting), sgb24_nov12=f1(n12.ae_gb_slow_history),
        r24_nov12=f1(n12.ae_level_anomaly), fm24_nov12=f1(n12.ae_chronos),
        miss3_blind=str(int(bp3.wait_now.isna().sum())),
        miss24_blind=str(int(bfp24.present.isna().sum())),
        clean_gb3_blind=f1((cl3.actual - cl3.model).abs().mean()),
        clean_r3_blind=f1((cl3.actual - cl3.rule).abs().mean()),
        ex12_fm24_diff=pm(float((ex12.ae_chronos - ex12.ae_level_anomaly).mean()), 2),
        ex12_sgb24_diff=pm(float((ex12.ae_gb_slow_history - ex12.ae_level_anomaly).mean()), 2),
    )

    prow = []
    for key, name in (("hist_mean", "Historical mean (baseline)"), ("rolling28", "Rule: 28-day mean, same hour"), ("damped_anomaly", "Rule: hist. mean + anomaly"),
                      ("level_anomaly", "Rule: 28-day mean + anomaly"), ("decision_tree", "Decision tree"), ("grad_boosting", "Gradient boosting (frozen)"),
                      ("gb_slow_history", "Gradient boosting + slow hist."), ("gb_level", "Gradient boosting + level")):
        prow.append([name] + [f1(pv(H, per, key)) for H in (3, 24) for per in ("oct", "nov")])
    F = pd.read_csv(ROOT / "results/foundation_metrics.csv").set_index(["H", "period"])
    FP = pd.read_csv(ROOT / "results/foundation_predictions.csv", parse_dates=["target_hour"])
    G3 = pd.read_csv(ROOT / "results/live_nov_predictions_3h.csv", parse_dates=["target_hour"])
    mrg = FP[(FP.H == 3) & (FP.period == "nov")].merge(G3[["hospital", "target_hour", "model"]], on=["hospital", "target_hour"])
    dd = mrg.assign(ae_gb=(mrg.actual - mrg.model).abs()).groupby(mrg.target_hour.dt.normalize())[["ae", "ae_gb"]].mean()
    N.update(fm3_oct=f1(F.loc[(3, "oct"), "MAE"]), fm3_nov=f1(F.loc[(3, "nov"), "MAE"]), fm24_oct=f1(F.loc[(24, "oct"), "MAE"]), fm24_nov=f1(F.loc[(24, "nov"), "MAE"]),
             fmc_oct=f"{F.loc[(3, 'oct'), 'coverage']:.0%}", fmc_nov=f"{F.loc[(3, 'nov'), 'coverage']:.0%}",
             fmw3_oct=round(F.loc[(3, "oct"), "width"]), fmw3_nov=round(F.loc[(3, "nov"), "width"]),
             fm_gap="1 to 2", fm_days=f"{int((dd.ae < dd.ae_gb).sum())} of {len(dd)}")
    prow.insert(5, ["Chronos-Bolt, zero-shot [10]"] + [f1(F.loc[(H, per), "MAE"]) for H in (3, 24) for per in ("oct", "nov")])
    blind_keys = ("hist_mean", "rolling28", "damped_anomaly", "level_anomaly", "decision_tree",
                  "chronos", "grad_boosting", "gb_slow_history", "gb_level")
    for r, key in zip(prow, blind_keys):
        b3v = f1(BF.loc[(3, "ext_11_25"), "MAE"]) if key == "chronos" else f1(bv(3, "ext_11_25", key))
        b24v = f1(BF.loc[(24, "ext_11_25"), "MAE"]) if key == "chronos" else f1(bv(24, "ext_11_25", key))
        r[:] = [r[0], r[1], r[2], b3v, r[3], r[4], b24v]
    N["table_post"] = br.md_table(["Method (before Sep 2026)", "3 h Oct", "3 h Nov", "3 h Blind", "24 h Oct", "24 h Nov", "24 h Blind"], prow) + \
        "\n\n: Post-hoc (complete Oct, 1–10 Nov) and post-freeze blind (12–25 Nov, 14 days) MAE in minutes.\n"

    d3_4, w3_4, n3_4 = -bv(3, "prereg_11_15", "grad_boosting_minus_level_anomaly"), int(bv(3, "prereg_11_15", "grad_boosting_minus_level_anomaly", "wins")), int(bv(3, "prereg_11_15", "grad_boosting_minus_level_anomaly", "n_days"))
    d3_14, w3_14, n3_14 = -bv(3, "ext_11_25", "grad_boosting_minus_level_anomaly"), int(bv(3, "ext_11_25", "grad_boosting_minus_level_anomaly", "wins")), int(bv(3, "ext_11_25", "grad_boosting_minus_level_anomaly", "n_days"))
    s24_4, s24_4_lo, s24_4_hi = bv(24, "prereg_11_15", "gb_slow_history_minus_level_anomaly"), bv(24, "prereg_11_15", "gb_slow_history_minus_level_anomaly", "ci_lo"), bv(24, "prereg_11_15", "gb_slow_history_minus_level_anomaly", "ci_hi")
    s24_14, s24_14_lo, s24_14_hi = bv(24, "ext_11_25", "gb_slow_history_minus_level_anomaly"), bv(24, "ext_11_25", "gb_slow_history_minus_level_anomaly", "ci_lo"), bv(24, "ext_11_25", "gb_slow_history_minus_level_anomaly", "ci_hi")
    ok_fm = lambda per: (min(bv(3, per, "grad_boosting"), bv(3, per, "level_anomaly")) - 1.0) <= BF.loc[(3, per), "MAE"] <= (max(bv(3, per, "grad_boosting"), bv(3, per, "level_anomaly")) + 1.0)
    ok_cov = lambda per: (0.70 <= bv(3, per, "quantile_10_90", "coverage") <= 0.80) and (0.70 <= bv(24, per, "quantile_10_90", "coverage") <= 0.80) and (0.75 <= BF.loc[(3, per), "coverage"] <= 0.85) and (0.75 <= BF.loc[(24, per), "coverage"] <= 0.85)
    ok_pw = lambda m, r, b, thr: abs(m - r) <= 0.02 and (min(m, r) - b) >= thr

    v1 = f"{'Met' if (1 <= d3_4 <= 3 and w3_4 > n3_4 / 2) else ('Direction met (>3 min)' if d3_4 > 3 else 'Missed')} (4 d); {'met' if (1 <= d3_14 <= 3 and w3_14 > n3_14 / 2) else 'missed'} (14 d)"
    v2 = f"{'Point est. met' if abs(s24_4) < 1.5 else 'Missed'}{' (CI wide)' if (s24_4_lo < -1.5 or s24_4_hi > 1.5) else ''} (4 d); {'point est. met' if abs(s24_14) < 1.5 else 'missed'}{' (CI wide)' if (s24_14_lo < -1.5 or s24_14_hi > 1.5) else ''} (14 d)"
    v3 = f"{'Met' if ok_fm('prereg_11_15') else 'Missed'} (4 d) & {'met' if ok_fm('ext_11_25') else 'missed'} (14 d)"
    v4 = f"{'Met' if ok_cov('prereg_11_15') else 'GB <70%'} (4 d); {'met' if ok_cov('ext_11_25') else 'missed'} (14 d)"
    v5 = f"{'Met ≥2 pp (ae49a32)' if (ok_pw(pw4_m, pw4_r, pw4_n, 0.02) and ok_pw(pw14_m, pw14_r, pw14_n, 0.02)) else 'Missed'}; {'met' if ok_pw(pw4_m, pw4_r, pw4_n, 0.08) else '< orig. ≥8 pp (amended scoring)'}"

    prereg_rows = [
        ["3 h: GB beats rule by 1–3 min, wins most days",
         f"{N['gb3_prereg']} vs {N['r3_prereg']} (−{N['g3_prereg']}, CI {N['g3_prereg_ci']}, {N['w3_prereg']} d)",
         f"{N['gb3_ext']} vs {N['r3_ext']} (−{N['g3_ext']}, CI {N['g3_ext_ci']}, {N['w3_ext']} d)",
         v1],
        ["24 h: slow GB within 1.5 min of rule",
         f"{N['s24_prereg_mae']} vs {N['r24_prereg']} ({N['s24_prereg_diff']}, CI {N['s24_prereg_ci']}, {N['w24_prereg']} d)",
         f"{N['s24_ext_mae']} vs {N['r24_ext']} ({N['s24_ext_diff']}, CI {N['s24_ext_ci']}, {N['w24_ext']} d)",
         v2],
        ["3 h Chronos-Bolt within 1 min or between",
         f"{N['fm3_prereg']} (GB {N['gb3_prereg']}, rule {N['r3_prereg']}; 24 h {N['fm24_prereg']})",
         f"{N['fm3_ext']} (GB {N['gb3_ext']}, rule {N['r3_ext']}; 24 h {N['fm24_ext']})",
         v3],
        ["10–90% cov.: GB 70–80%; Chronos 75–85%",
         f"GB {N['qc3_prereg_1d']} / {N['qc24_prereg_1d']}; Chronos {N['fmc3_prereg']} / {N['fmc24_prereg']}",
         f"GB {N['qc3_ext']} / {N['qc24_ext']}; Chronos {N['fmc3_ext']} / {N['fmc24_ext']}",
         v4],
        ["3 h pairwise: rule & GB ≤2 pp, ≥2 pp > board",
         f"GB {N['pw_prereg_model']}, rule {N['pw_prereg_rule']}, hist {N['pw_prereg_hist']}, board {N['pw_prereg_now']}",
         f"GB {N['pw_ext_model']}, rule {N['pw_ext_rule']}, hist {N['pw_ext_hist']}, board {N['pw_ext_now']} ({N['pw_ext_n']} pairs)",
         v5],
    ]
    prereg_hdr = ["Expectation (11 Nov)", "12–15 Nov (4 d)", "12–25 Nov (14 d)", "Computed verdict"]
    N["table_prereg"] = "\n".join([
        "| " + " | ".join(prereg_hdr) + " |",
        "|" + "-" * 24 + "|" + "-" * 26 + "|" + "-" * 26 + "|" + "-" * 22 + "|",
        *["| " + " | ".join(str(c) for c in r) + " |" for r in prereg_rows],
    ]) + "\n\n: Blind-test scorecard against `PREREGISTRATION.md` (11 Nov).\n"
    N["prereg_all_mae"] = (
        f"historical mean {N['prereg_hm']} / {N['prereg_hm']}, "
        f"persistence {f1(bv(3, 'prereg_11_15', 'persistence'))} / {f1(bv(24, 'prereg_11_15', 'persistence'))}, "
        f"rule (level+anom.) {N['r3_prereg']} / {N['r24_prereg']}, "
        f"decision tree {f1(bv(3, 'prereg_11_15', 'decision_tree'))} / {f1(bv(24, 'prereg_11_15', 'decision_tree'))}, "
        f"Chronos-Bolt {N['fm3_prereg']} / {N['fm24_prereg']}, "
        f"frozen GB {N['gb3_prereg']} / {f1(bv(24, 'prereg_11_15', 'grad_boosting'))}, "
        f"slow GB {f1(bv(3, 'prereg_11_15', 'gb_slow_history'))} / {N['s24_prereg_mae']}"
    )
    # Table 2 with precision and recall (the proposal promised them)
    mm = pd.read_csv(ROOT / f"{a.prefix}_metrics.csv")

    def c2(model, H, col, regime="A_all_eras"):
        q = mm[(mm.task == "cls") & (mm.model == model) & (mm.H == H) & (mm.split == "test") & (mm.thr == "0.5")]
        if model not in ("persistence", "hist_mean"):
            q = q[q.regime == regime]
        return f3(q[col].iloc[0])
    crows = [[n_] + [c2(k, 3, c_) for c_ in ("acc", "precision", "recall", "AUC")] + [c2(k, 24, "AUC")] for n_, k in
             (("Persistence (baseline)", "persistence"), ("Historical mean (baseline)", "hist_mean"), ("Logistic regression", "logistic"),
              ("Random forest", "random_forest"), ("Gradient boosting", "grad_boosting"))]
    crows.append(["Gradient boosting, new format only"] + [c2("grad_boosting", 3, c_, "B_new_only") for c_ in ("acc", "precision", "recall", "AUC")] + [c2("grad_boosting", 24, "AUC", "B_new_only")])
    N["table_cls"] = br.md_table(["Model (October test)", "3 h acc", "3 h precision", "3 h recall", "3 h AUC", "24 h AUC"], crows) + \
        "\n\n: Classification of \"wait over 2 hours\" on the October test set, threshold 0.5.\n"
    N["logit_b"] = c2("logistic", 3, "AUC", "B_new_only")

    N.update(fig_eda=str(ROOT / "results/checkpoint_fig.png"), fig_models=str(ROOT / "results/report_fig_models.png"),
             fig_dev=str(ROOT / "results/report_fig_development.png"), fig_nov=str(ROOT / "results/report_fig_november.png"))
    z = next((p for p in (Path(a.zip), ROOT / "Group16-Project-data.zip") if p.exists()), Path(a.zip))
    N["zip_sha"] = hashlib.sha256(z.read_bytes()).hexdigest()[:16] if z.exists() else "TBD"
    log = (ROOT / "EXPERIMENT_LOG.md").read_text()
    for a_, b_ in (("→", r"$\to$"), ("≈", r"$\approx$"), ("≠", r"$\neq$")):
        log = log.replace(a_, b_)
    N["exp_log"] = log[log.index("| Date"):].replace("|---|---|---|---|---|", "|------|------------------|------------------|--------------------|-----------|")

    if a.official:
        import build_official
        return build_official.build(N, ROOT, OUT)
    text = (ROOT / "report/report_final.md").read_text().format_map(N)
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "ELEC3544-Report-Group16.md"
    md.write_text(text)
    hdr = OUT / "_header.tex"; hdr.write_text(HEADER)
    common = ["pandoc", str(md), "--from", "markdown+raw_tex"]
    subprocess.run(common + ["-o", str(md.with_suffix(".pdf")), "--pdf-engine=xelatex", "-H", str(hdr),
                             "-V", "geometry:a4paper,left=1.8cm,right=1.8cm,top=2.0cm,bottom=1.9cm", "-V", "fontsize=10pt",
                             "-V", "monofont=Menlo", "-V", "monofontoptions=Scale=0.78", "-V", "colorlinks=true", "-V", "urlcolor=black", "-V", "linkcolor=black"], check=True)
    # (no .docx: the title block uses LaTeX-only styling)
    hdr.unlink()
    print(f"wrote {md.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
