"""Evidence figure for the project checkpoint (Nov 2026).

Usage: python src/plot_checkpoint.py [--in data/processed/hourly_features_full.csv] [--out results/checkpoint_fig.png]

Left : mean cat 4/5 median wait by hour of day, weekday vs weekend/holiday (new format only, from 2025-10-13).
Right: monthly share of hospital-hours with cat 4/5 wait > 2 h, both formats, with the 2025-10-13 format change marked.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/hourly_features_full.csv")
    ap.add_argument("--out", default="results/checkpoint_fig.png")
    a = ap.parse_args()

    h = pd.read_csv(a.inp, usecols=["hospital", "fmt", "hour_ts", "t45_min", "t45_over2h",
                                    "hr", "is_weekend", "is_holiday"], parse_dates=["hour_ts"])
    new = h[h["fmt"] == "new"].dropna(subset=["t45_min"])
    off = new["is_weekend"].astype(bool) | new["is_holiday"].astype(bool)
    by_hr = pd.DataFrame({"Weekday": new[~off].groupby("hr")["t45_min"].mean(),
                          "Weekend / holiday": new[off].groupby("hr")["t45_min"].mean()})
    monthly = h.dropna(subset=["t45_over2h"]).set_index("hour_ts").groupby(
        [pd.Grouper(freq="MS"), "fmt"])["t45_over2h"].mean().unstack("fmt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.0), dpi=200)
    by_hr.plot(ax=ax1, marker="o", ms=3, lw=1.4)
    ax1.axhline(120, color="grey", ls=":", lw=1)
    ax1.set(xlabel="Hour of day", ylabel="Cat 4/5 median wait (min)", xticks=range(0, 24, 3),
            title="Daily cycle, 18 hospitals (Oct 2025 - Sep 2026)")
    ax1.legend(fontsize=7, frameon=False)
    for f, lab in (("old", "Old format (category lower bound)"), ("new", "New format (median)")):
        if f in monthly:
            ax2.plot(monthly.index, monthly[f], marker="o", ms=2.5, lw=1.2, label=lab)
    ax2.axvline(pd.Timestamp("2025-10-13"), color="grey", ls="--", lw=1)
    ax2.set(ylabel="Share of hours > 2 h", ylim=(0, 1), title="Wait > 2 h by month (label, 2023-2026)")
    ax2.legend(fontsize=7, frameon=False, loc="lower left")
    ax2.tick_params(axis="x", labelsize=7)
    for ax in (ax1, ax2):
        ax.title.set_fontsize(9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"saved {a.out}")
    print(by_hr.round(0).T.to_string())
    print(monthly.round(3).tail(14).to_string())


if __name__ == "__main__":
    main()
