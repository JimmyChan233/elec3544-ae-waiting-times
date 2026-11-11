"""Figures from train_models.py outputs.

Usage: python src/plot_results.py [--prefix results/models_2026-10-30]
Writes <prefix>_fig_models.png (test-month MAE / AUC per model) and <prefix>_fig_ablation.png (drop-one-group).
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REG_ORDER = ["persistence", "hist_mean", "ols_calendar_only", "ols", "ridge", "lasso", "random_forest", "grad_boosting"]
CLS_ORDER = ["persistence", "hist_mean", "logistic", "random_forest", "grad_boosting"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/models_2026-10-30")
    a = ap.parse_args()
    m = pd.read_csv(f"{a.prefix}_metrics.csv")
    abl = pd.read_csv(f"{a.prefix}_ablation.csv")

    reg = m[(m.task == "reg") & (m.split == "test")].pivot(index="model", columns="H", values="MAE").reindex(REG_ORDER)
    cls = m[(m.task == "cls") & (m.split == "test") & (m.regime == "A_all_eras") & (m.thr == "0.5")]
    cls = cls.pivot(index="model", columns="H", values="AUC").reindex(CLS_ORDER)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=200, gridspec_kw={"width_ratios": [1.5, 1]})
    reg.plot.barh(ax=ax1, width=0.75)
    ax1.set(xlabel="Test MAE, Oct 2026 (min)", ylabel="", title="Cat 4/5 wait regression")
    ax1.invert_yaxis()
    cls.plot.barh(ax=ax2, width=0.75, legend=False)
    ax2.set(xlabel="Test AUC, Oct 2026", ylabel="", title="Wait > 2 h classifier", xlim=(0.6, 0.95))
    ax2.invert_yaxis()
    ax1.set_xlim(0, 78)
    ax1.legend([f"{h} h ahead" for h in reg.columns], fontsize=7, frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.title.set_fontsize(9); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.prefix}_fig_models.png", bbox_inches="tight")

    r = abl[abl.task == "reg"].pivot(index="group", columns="H", values="delta").sort_values(abl.H.min(), ascending=False)
    fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=200)
    r.plot.barh(ax=ax, width=0.75)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(xlabel="Val MAE increase when group removed (min)", ylabel="", title="Which feature group matters (grad boosting)")
    ax.invert_yaxis(); ax.title.set_fontsize(9); ax.tick_params(labelsize=7)
    ax.legend(ax.containers, [f"{h} h ahead" for h in r.columns], fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.prefix}_fig_ablation.png", bbox_inches="tight")
    print(reg.round(1).to_string()); print(cls.round(3).to_string()); print(r.round(2).to_string())


if __name__ == "__main__":
    main()
