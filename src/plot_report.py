"""Figures for the final report, drawn from the result files only (no model is fitted here).

Usage: python src/plot_report.py
Writes results/report_fig_{models,development,november}.png
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import train_models as tm  # noqa: E402
from live_test import NOV, add_slow  # noqa: E402
from posthoc_checks import add_level  # noqa: E402

GREY, BLUE, TAN = "#9aa3ad", "#2f6690", "#d08c3f"
CHARCOAL, CRIMSON = "#1f2421", "#b8332a"
NAMES = {"persistence": "Persistence (baseline)", "hist_mean": "Historical mean (baseline)",
         "ols_calendar_only": "OLS, calendar + hospital", "ols": "OLS, all features", "ridge": "Ridge", "lasso": "Lasso",
         "random_forest": "Random forest", "grad_boosting": "Gradient boosting", "logistic": "Logistic regression"}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.title.set_fontsize(10.5)
    ax.xaxis.label.set_fontsize(9.5)
    ax.yaxis.label.set_fontsize(9.5)


def fig_models():
    m = pd.read_csv("results/models_2026-10-30_v2_metrics.csv")
    reg = m[(m.task == "reg") & (m.split == "test")].pivot(index="model", columns="H", values="MAE")
    reg = reg.reindex(["persistence", "hist_mean", "ols_calendar_only", "ols", "ridge", "lasso", "random_forest", "grad_boosting"])
    cls = m[(m.task == "cls") & (m.split == "test") & (m.regime == "A_all_eras") & (m.thr == "0.5")]
    cls = cls.pivot(index="model", columns="H", values="AUC").reindex(["persistence", "hist_mean", "logistic", "random_forest", "grad_boosting"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=220, gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, d, fmt in ((a1, reg, "{:.1f}"), (a2, cls, "{:.3f}")):
        y = np.arange(len(d))
        for off, H, c in ((-0.2, 3, BLUE), (0.2, 24, TAN)):
            ax.barh(y + off, d[H], height=0.38, color=c, label=f"{H} h ahead")
            for yi, v in zip(y + off, d[H]):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8)
        ax.set_yticks(y, [NAMES[k] for k in d.index])
        ax.invert_yaxis()
        tidy(ax)
    a1.axvline(reg.loc["hist_mean", 3], color="grey", lw=0.8, ls=":")
    a1.set(xlabel="October 2026 test MAE (minutes; lower is better)", title="Category 4/5 wait, regression", xlim=(0, 74))
    a2.set(xlabel="October test ROC-AUC (higher is better)", title="Wait over 2 hours, classification", xlim=(0.6, 0.96))
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.01), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=2.0)
    fig.savefig("results/report_fig_models.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def fig_development():
    abl = pd.read_csv("results/models_2026-10-30_ablation.csv")
    r = abl[abl.task == "reg"].pivot(index="group", columns="H", values="delta").sort_values(3, ascending=False)
    lab = {"history": "Recent waits", "calendar": "Calendar", "hospital": "Hospital", "hist_mean": "Historical-mean feature",
           "air": "Air quality (AQHI)", "weather": "Weather"}
    v = pd.read_csv("results/hn_24h_validation.csv")
    v = v[v["mode"] == "raw"].set_index("model").val_MAE
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.3, 2.7), dpi=220)
    y = np.arange(len(r))
    a1.barh(y - 0.2, r[3], height=0.38, color=BLUE, label="3 h ahead")
    a1.barh(y + 0.2, r[24], height=0.38, color=TAN, label="24 h ahead")
    a1.axvline(0, color="grey", lw=0.8)
    a1.set_yticks(y, [lab.get(g, g) for g in r.index])
    a1.invert_yaxis()
    a1.set(xlabel="Increase in validation MAE when removed (minutes)", title="(a) Which feature group matters")
    a1.legend(fontsize=8.5, frameon=False, loc="lower right")
    tidy(a1)
    vals = [51.549, v["original"], v["residual"], v["residual_slow"], v["slow_history"]]
    names = ["Historical mean (baseline)", "Gradient boosting, original", "+ residual target", "+ residual and slow history", "+ slow history (selected)"]
    a2.barh(names, vals, color=[GREY, BLUE, BLUE, BLUE, TAN])
    a2.invert_yaxis()
    for i, x in enumerate(vals):
        a2.text(x, i, f" {x:.2f}", va="center", fontsize=8.5)
    a2.set(xlabel="September validation MAE (minutes)", title="(b) Four attempts at the 24 h horizon", xlim=(44, 53.8))
    tidy(a2)
    fig.tight_layout(w_pad=2.2)
    fig.savefig("results/report_fig_development.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _nov_early_rule():
    """Daily MAE of frozen level_anomaly on 1-10 Nov (identical formula to posthoc_checks.py)."""
    for g in ("weather", "air"):
        tm.GROUPS.pop(g, None)
    grid, hol = tm.load_grid("data/processed/hourly_features_latest.csv", "data/raw/aux")
    tm.END = grid.hour_ts.max() + pd.Timedelta(hours=1)
    out = {}
    for H in (3, 24):
        X = add_level(add_slow(tm.make_horizon(grid, hol, H)), H)
        D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
        D = tm.add_hist(D, "y_f", D.ts_f < tm.VAL)
        tr = D[D.ts_f < tm.VAL]
        ok = tr[["r_f", "anom"]].notna().all(axis=1)
        b2 = float(((tr.y_f - tr["r_f"]) * tr["anom"])[ok].sum() / (tr["anom"] ** 2)[ok].sum())
        nov = D[D.ts_f >= NOV].copy()
        pred = (nov.r_f + b2 * nov.anom).fillna(nov.r_f).fillna(nov["hist"])
        nov["ae_rule"] = np.abs(nov.y_f - pred)
        assert abs(nov["ae_rule"].mean() - (41.18 if H == 3 else 47.85)) < 0.05
        d = nov.assign(day=nov.ts_f.dt.floor("D")).groupby("day")["ae_rule"].mean()
        out[H] = d
    return out


def fig_november():
    live = pd.read_csv("results/live_nov_by_day.csv", parse_dates=["ts_f"])
    live = live[live["mode"] == "frozen"]
    blind = pd.read_csv("results/blind_nov_by_day.csv", parse_dates=["ts_f"])
    blind = blind[blind["mode"] == "blind"]
    rule_early = _nov_early_rule()
    fm_early = pd.read_csv("results/foundation_predictions.csv", parse_dates=["target_hour"])
    fm_early = fm_early[fm_early["period"] == "nov"].assign(day=lambda x: x.target_hour.dt.floor("D"))
    fm_blind = pd.read_csv("results/blind_nov_foundation_predictions.csv", parse_dates=["target_hour"])
    fm_blind = fm_blind.assign(day=lambda x: x.target_hour.dt.floor("D"))
    fm_all = pd.concat([fm_early, fm_blind], ignore_index=True).groupby(["H", "day"])["ae"].mean()

    d = pd.concat([live, blind], ignore_index=True).sort_values(["H", "ts_f"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.1), dpi=220, sharey=True)
    for ax, H, col, name in ((axes[0], 3, "ae_grad_boosting", "Gradient boosting"), (axes[1], 24, "ae_gb_slow_history", "GB + slow history")):
        g = d[d.H == H].reset_index(drop=True)
        rule_series = pd.concat([rule_early[H], blind[blind.H == H].set_index("ts_f")["ae_level_anomaly"]])
        g["ae_rule"] = g["ts_f"].map(rule_series)
        g["ae_chronos"] = [fm_all.loc[(H, t)] for t in g["ts_f"]]

        x = np.arange(len(g))
        ax.bar(x - 0.2, g.ae_hist_mean, width=0.38, color=GREY, alpha=0.85, label="Historical mean")
        ax.bar(x + 0.2, g[col], width=0.38, color=BLUE if H == 3 else TAN, alpha=0.9, label=name)
        ax.plot(x, g.ae_rule, color=CHARCOAL, ls="--", lw=1.35, marker="o", ms=2.6, label="Rule: level + anomaly")
        ax.plot(x, g.ae_chronos, color=CRIMSON, ls="-", lw=1.45, marker="s", ms=2.5, label="Chronos-Bolt (zero-shot)")

        s1 = len(live[live.H == H]) - 0.5
        s2 = s1 + int((g.ts_f < "2026-11-16").sum()) - len(live[live.H == H])
        ax.axvline(s1, color="grey", ls="--", lw=0.9)
        ax.axvline(s2, color="grey", ls=":", lw=0.9)
        ax.text(s1 / 2 - 0.2, 86, "Live", ha="center", va="top", fontsize=7.8, color="#444444")
        ax.text((s1 + s2) / 2, 86, "Pre-reg", ha="center", va="top", fontsize=7.8, color="#444444")
        ax.text((s2 + len(g) - 1) / 2 + 0.2, 86, "Extended blind", ha="center", va="top", fontsize=7.8, color="#444444")
        ax.set_xticks(x, [t.strftime("%-d") for t in g.ts_f])
        ax.set_xlabel("Day of November 2026")
        tidy(ax)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set(ylim=(0, 90))
        ax.set_title(f"{H} h ahead", pad=30)
        ax.legend(fontsize=7.4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.00), ncol=2, handlelength=1.8, columnspacing=1.0)
    axes[0].set_ylabel("MAE on that day (minutes)")
    fig.tight_layout(w_pad=1.6)
    fig.savefig("results/report_fig_november.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    fig_models()
    fig_development()
    fig_november()
    print("wrote results/report_fig_{models,development,november}.png")
