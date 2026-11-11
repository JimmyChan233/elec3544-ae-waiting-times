"""Figures for the final report, drawn from the result files only (no model is fitted here).

Usage: python src/plot_report.py
Writes results/report_fig_{models,development,november}.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GREY, BLUE, TAN = "#9aa3ad", "#2f6690", "#d08c3f"
NAMES = {"persistence": "Persistence (baseline)", "hist_mean": "Historical mean (baseline)",
         "ols_calendar_only": "OLS, calendar + hospital", "ols": "OLS, all features", "ridge": "Ridge", "lasso": "Lasso",
         "random_forest": "Random forest", "grad_boosting": "Gradient boosting", "logistic": "Logistic regression"}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.title.set_fontsize(10.5)
    ax.xaxis.label.set_fontsize(9.5); ax.yaxis.label.set_fontsize(9.5)


def fig_models():
    m = pd.read_csv("results/models_2026-10-30_v2_metrics.csv")
    reg = m[(m.task == "reg") & (m.split == "test")].pivot(index="model", columns="H", values="MAE")
    reg = reg.reindex(["persistence", "hist_mean", "ols_calendar_only", "ols", "ridge", "lasso", "random_forest", "grad_boosting"])
    cls = m[(m.task == "cls") & (m.split == "test") & (m.regime == "A_all_eras") & (m.thr == "0.5")]
    cls = cls.pivot(index="model", columns="H", values="AUC").reindex(["persistence", "hist_mean", "logistic", "random_forest", "grad_boosting"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.0), dpi=220, gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, d, fmt in ((a1, reg, "{:.1f}"), (a2, cls, "{:.3f}")):
        y = np.arange(len(d))
        for off, H, c in ((-0.2, 3, BLUE), (0.2, 24, TAN)):
            b = ax.barh(y + off, d[H], height=0.38, color=c, label=f"{H} h ahead")
            for yi, v in zip(y + off, d[H]):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8)
        ax.set_yticks(y, [NAMES[k] for k in d.index]); ax.invert_yaxis(); tidy(ax)
    a1.axvline(reg.loc["hist_mean", 3], color="grey", lw=0.8, ls=":")
    a1.set(xlabel="October 2026 test MAE (minutes; lower is better)", title="Category 4/5 wait, regression", xlim=(0, 84))
    a2.set(xlabel="October 2026 test ROC-AUC (higher is better)", title="Wait over 2 hours, classification", xlim=(0.6, 0.95))
    a1.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("results/report_fig_models.png", bbox_inches="tight"); plt.close(fig)


def fig_development():
    abl = pd.read_csv("results/models_2026-10-30_ablation.csv")
    r = abl[abl.task == "reg"].pivot(index="group", columns="H", values="delta").sort_values(3, ascending=False)
    lab = {"history": "Recent waits", "calendar": "Calendar", "hospital": "Hospital", "hist_mean": "Historical-mean feature",
           "air": "Air quality (AQHI)", "weather": "Weather"}
    v = pd.read_csv("results/hn_24h_validation.csv"); v = v[v["mode"] == "raw"].set_index("model").val_MAE
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.6), dpi=220)
    y = np.arange(len(r))
    a1.barh(y - 0.2, r[3], height=0.38, color=BLUE, label="3 h ahead"); a1.barh(y + 0.2, r[24], height=0.38, color=TAN, label="24 h ahead")
    a1.axvline(0, color="grey", lw=0.8); a1.set_yticks(y, [lab.get(g, g) for g in r.index]); a1.invert_yaxis()
    a1.set(xlabel="Increase in validation MAE when the group is removed (minutes)", title="(a) Which feature group matters")
    a1.legend(fontsize=8.5, frameon=False, loc="lower right"); tidy(a1)
    vals = [51.549, v["original"], v["residual"], v["residual_slow"], v["slow_history"]]
    names = ["Historical mean (baseline)", "Gradient boosting, original", "+ residual target", "+ residual and slow history", "+ slow history (selected)"]
    a2.barh(names, vals, color=[GREY, BLUE, BLUE, BLUE, TAN]); a2.invert_yaxis()
    for i, x in enumerate(vals):
        a2.text(x, i, f" {x:.2f}", va="center", fontsize=8.5)
    a2.set(xlabel="September validation MAE, 24 h ahead (minutes)", title="(b) Four attempts at the 24 h horizon", xlim=(44, 53.5)); tidy(a2)
    fig.tight_layout(); fig.savefig("results/report_fig_development.png", bbox_inches="tight"); plt.close(fig)


def fig_november():
    d = pd.read_csv("results/live_nov_by_day.csv", parse_dates=["ts_f"])
    d = d[d["mode"] == "frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), dpi=220, sharey=True)
    for ax, H, col, name in ((axes[0], 3, "ae_grad_boosting", "Gradient boosting"), (axes[1], 24, "ae_gb_slow_history", "Gradient boosting + slow history")):
        g = d[d.H == H]
        x = np.arange(len(g))
        ax.bar(x - 0.2, g.ae_hist_mean, width=0.38, color=GREY, label="Historical mean (baseline)")
        ax.bar(x + 0.2, g[col], width=0.38, color=BLUE if H == 3 else TAN, label=name)
        ax.set_xticks(x, [t.strftime("%-d") for t in g.ts_f]); ax.set_xlabel("Day of November 2026"); tidy(ax)
        ax.set(title=f"{H} h ahead", ylim=(0, 75)); ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    axes[0].set_ylabel("MAE on that day (minutes)")
    fig.tight_layout(); fig.savefig("results/report_fig_november.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_models(); fig_development(); fig_november()
    print("wrote results/report_fig_{models,development,november}.png")
