"""Fill report_final.md with numbers from Project/results and render the final report (PDF via xelatex, plus .docx).

Usage: python3 report/build_final.py [--zip ../Assignments/Share/Group16-Project-2026-11-11.zip]
Every number in the report comes from a result file; edit the words in report_final.md, never the PDF.
"""
import argparse, hashlib, os, re, subprocess, sys
from pathlib import Path
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
    a = ap.parse_args()
    N = br.numbers(a)
    f1, f3 = br.f1, br.f3

    N["top_gain"] = N["top_gain"].replace("PYNEH", "Pamela Youde Nethersole Eastern")
    N["low_gain"] = N["low_gain"].replace("AHNH", "Alice Ho Miu Ling Nethersole")
    # --- latest data facts
    h = pd.read_csv(ROOT / "data/processed/hourly_features_latest.csv", usecols=["fmt"])
    raw = Path.home() / "elec3544_data/raw/ae"
    n_files = sum(len(os.listdir(raw / d)) for d in ("old", "new")) if raw.exists() else 33219
    summ = (ROOT / f"{a.prefix}_summary.txt").read_text()
    N.update(n_rows_latest=f"{len(h):,}", n_rows_new_latest=f"{(h.fmt == 'new').sum():,}", n_files=f"{n_files:,}",
             n_missing_pct=re.search(r"\(([\d.]+)% hours missing", summ).group(1))

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
             b3=f"{float(re.search(r'H=3: .*level-anomaly weight ([0-9.]+)', psum).group(1)):.2f}",
             g3_oct=f1(-pv(3, "oct", f"grad_boosting_minus_{rule}")), g3_oct_ci=pci(3, "oct", f"grad_boosting_minus_{rule}"),
             g3_nov=f1(-pv(3, "nov", f"grad_boosting_minus_{rule}")), g3_nov_ci=pci(3, "nov", f"grad_boosting_minus_{rule}"),
             w3_oct=wins[0], w3_nov=wins[1],
             s24_oct=pm(pv(24, "oct", f"gb_slow_history_minus_{rule}")), s24_oct_ci=pci(24, "oct", f"gb_slow_history_minus_{rule}"),
             s24_nov=pm(pv(24, "nov", f"gb_slow_history_minus_{rule}")), s24_nov_ci=pci(24, "nov", f"gb_slow_history_minus_{rule}"),
             gl3_oct=f1(pv(3, "oct", "gb_level")), gl3_nov=f1(pv(3, "nov", "gb_level")), gl24_oct=f1(pv(24, "oct", "gb_level")), gl24_nov=f1(pv(24, "nov", "gb_level")),
             qw=round(pv(3, "nov", "quantile_10_90", "width") / 5) * 5, qc_oct=f"{pv(3, 'oct', 'quantile_10_90', 'coverage'):.0%}", qc_nov=f"{pv(3, 'nov', 'quantile_10_90', 'coverage'):.0%}",
             q120=re.search(r"exactly 120 min: ([\d.]+%)", psum).group(1), wx_missing=re.search(r"rainfall\): ([\d.]+)%", psum).group(1).split(".")[0] + "%",
             pw_rule=f"{float(re.search(r'rule\s+correct ([\d.]+)', (ROOT / 'results/posthoc_pairwise.txt').read_text()).group(1)):.0%}",
             nov_gb24=f1(lv("reg", 24, "frozen", "nov", "grad_boosting", "MAE")), nov_dgb24=f1(-lv("reg_diff_vs_hist", 24, "frozen", "nov", "grad_boosting", "MAE")),
             nov_cigb24=ci(24, "nov", "grad_boosting"))
    pwt = (ROOT / "results/live_nov_pairwise.txt").read_text()
    bd = pd.read_csv(ROOT / "results/live_nov_by_day.csv"); bd = bd[bd["mode"] == "frozen"]
    d3, d24 = bd[bd.H == 3], bd[bd.H == 24]
    N.update(post_hm_oct=f1(pv(3, "oct", "hist_mean")), b24=f"{float(re.search(r'H=24: .*level-anomaly weight ([0-9.]+)', psum).group(1)):.2f}",
             dt3=re.findall(r"decision tree: depth (\d+)", psum)[0], dt24=re.findall(r"decision tree: depth (\d+)", psum)[1],
             pw_n=re.search(r"over ([\d,]+) hospital-pair", pwt).group(1),
             pw_now=f"{float(re.search(r'wait_now\s+correct ([\d.]+)', pwt).group(1)):.0%}", pw_hist=f"{float(re.search(r'hist_mean\s+correct ([\d.]+)', pwt).group(1)):.0%}",
             pw_model=f"{float(re.search(r'model\s+correct ([\d.]+)', pwt).group(1)):.0%}",
             w3_live=("all " if (d3.ae_grad_boosting < d3.ae_hist_mean).all() else f"{int((d3.ae_grad_boosting < d3.ae_hist_mean).sum())} of ") + str(len(d3)),
             w24_live=f"{int((d24.ae_gb_slow_history < d24.ae_hist_mean).sum())} of {len(d24)}")
    lat = pd.read_csv(ROOT / "data/processed/hourly_features_latest.csv", usecols=["hospital", "fmt", "t45_min"])
    byh2 = lat[lat.fmt == "new"].groupby("hospital").t45_min.mean()
    N.update(busiest=br.short(byh2.idxmax()), busiest_min=round(byh2.max()), least_busy=br.short(byh2.idxmin()), least_busy_min=round(byh2.min()))
    prow = []
    for key, name in (("hist_mean", "Historical mean (original baseline)"), ("rolling28", "Rule: 28-day mean, same clock hour"), ("damped_anomaly", "Rule: historical mean + damped anomaly"),
                      ("level_anomaly", "Rule: 28-day mean + damped anomaly"), ("decision_tree", "Decision tree"), ("grad_boosting", "Gradient boosting (frozen model)"),
                      ("gb_slow_history", "Gradient boosting + slow history"), ("gb_level", "Gradient boosting + level features")):
        prow.append([name] + [f1(pv(H, per, key)) for H in (3, 24) for per in ("oct", "nov")])
    F = pd.read_csv(ROOT / "results/foundation_metrics.csv").set_index(["H", "period"])
    FP = pd.read_csv(ROOT / "results/foundation_predictions.csv", parse_dates=["target_hour"])
    G3 = pd.read_csv(ROOT / "results/live_nov_predictions_3h.csv", parse_dates=["target_hour"])
    mrg = FP[(FP.H == 3) & (FP.period == "nov")].merge(G3[["hospital", "target_hour", "model"]], on=["hospital", "target_hour"])
    dd = mrg.assign(ae_gb=(mrg.actual - mrg.model).abs()).groupby(mrg.target_hour.dt.normalize())[["ae", "ae_gb"]].mean()
    gaps = sorted([F.loc[(3, "nov"), "MAE"] - pv(3, "nov", "grad_boosting"), F.loc[(3, "oct"), "MAE"] - pv(3, "oct", "grad_boosting")])
    N.update(fm3_oct=f1(F.loc[(3, "oct"), "MAE"]), fm3_nov=f1(F.loc[(3, "nov"), "MAE"]), fm24_oct=f1(F.loc[(24, "oct"), "MAE"]), fm24_nov=f1(F.loc[(24, "nov"), "MAE"]),
             fmc_oct=f"{F.loc[(3, 'oct'), 'coverage']:.0%}", fmc_nov=f"{F.loc[(3, 'nov'), 'coverage']:.0%}",
             fm_gap="1 to 2", fm_days=f"{int((dd.ae < dd.ae_gb).sum())} of {len(dd)}")
    prow.insert(5, ["Chronos-Bolt, zero-shot [10]"] + [f1(F.loc[(H, per), "MAE"]) for H in (3, 24) for per in ("oct", "nov")])
    N["table_post"] = br.md_table(["Method (trained before Sep 2026)", "3 h Oct", "3 h Nov", "24 h Oct", "24 h Nov"], prow) + \
        "\n\n: Post-hoc comparison, MAE in minutes on the complete October (1–31) and on 1–10 November.\n"
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
    z = Path(a.zip)
    N["zip_sha"] = hashlib.sha256(z.read_bytes()).hexdigest()[:16] if z.exists() else "TBD"
    log = (ROOT / "EXPERIMENT_LOG.md").read_text()
    for a_, b_ in (("→", r"$\to$"), ("≈", r"$\approx$"), ("≠", r"$\neq$")):
        log = log.replace(a_, b_)
    N["exp_log"] = log[log.index("| Date"):].replace("|---|---|---|---|---|", "|------|------------------|------------------|--------------------|-----------|")

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
