import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_models as tm

OUT = Path("results")
PREFIX = OUT / "hn_24h"


def metrics(y, pred):
    return dict(MAE=mean_absolute_error(y, pred), RMSE=float(np.sqrt(np.mean((y - pred) ** 2))))


def main():
    grid, hol = tm.load_grid("data/processed/hourly_features_full.csv", "data/raw/aux")
    X = tm.make_horizon(grid, hol, 24)
    parts = []
    for _, g in X.sort_values(["hospital", "ts"]).groupby("hospital", sort=False):
        g = g.copy()
        same_hour = pd.concat([g.y0.shift(k) for k in range(0, 168, 24)], axis=1)
        g["same_hour_7d"] = same_hour.mean(axis=1).where(same_hour.notna().sum(axis=1) >= 3)
        g["roll168"] = g.y0.rolling(168, min_periods=24).mean()
        parts.append(g)
    X = pd.concat(parts, ignore_index=True)
    D = X[(X.ts_f >= tm.SWITCH) & X.y_f.notna()].copy()
    D = add_hist = tm.add_hist(D, "y_f", D.split == "train")
    tr, va, te = (D[D.split == s].copy() for s in ("train", "val", "test"))
    base_feats = [f for g, fs in tm.GROUPS.items() if g not in ("era", "weather", "air") for f in fs]
    slow_feats = base_feats + ["same_hour_7d", "roll168"]

    lines = [f"24 h regression | train {len(tr):,} val {len(va):,} test {len(te):,}",
             "All model and shrinkage choices use September validation only. October is evaluated for the selected models afterwards."]
    hgb_grid = [dict(learning_rate=0.05, max_leaf_nodes=31),
                dict(learning_rate=0.05, max_leaf_nodes=63),
                dict(learning_rate=0.1, max_leaf_nodes=31)]
    grid_scores = []
    for kw in hgb_grid:
        p, secs, _ = tm.fit_predict("hgb", tm.hgb_reg(**kw), base_feats, tr, {"val": va}, "y_f")
        score = mean_absolute_error(va.y_f, p["val"])
        grid_scores.append(score)
        lines.append(f"Initial HGB {kw}: val MAE {score:.3f} ({secs:.0f}s)")
    best_kw = hgb_grid[int(np.argmin(grid_scores))]
    lines.append(f"Selected HGB parameters: {json.dumps(best_kw)}")

    variants = [
        ("original", base_feats, "y_f", False),
        ("residual", base_feats, "y_resid", True),
        ("slow_history", slow_feats, "y_f", False),
        ("residual_slow", slow_feats, "y_resid", True),
    ]
    D["y_resid"] = D.y_f - D["hist"]
    tr, va, te = (D[D.split == s].copy() for s in ("train", "val", "test"))
    records = []
    fitted_val = {}
    alphas = np.linspace(0, 1, 21)
    for name, feats, target, is_residual in variants:
        p, secs, _ = tm.fit_predict("hgb", tm.hgb_reg(**best_kw), feats, tr, {"val": va}, target)
        raw = p["val"] + (va["hist"].values if is_residual else 0)
        raw_mae = mean_absolute_error(va.y_f, raw)
        blend_scores = [mean_absolute_error(va.y_f, va["hist"].values + a * (raw - va["hist"].values)) for a in alphas]
        alpha = float(alphas[int(np.argmin(blend_scores))])
        shrink_mae = float(min(blend_scores))
        fitted_val[name] = dict(feats=feats, target=target, is_residual=is_residual, raw=raw, alpha=alpha)
        records.append(dict(model=name, mode="raw", val_MAE=raw_mae, alpha=1.0, fit_seconds=secs))
        records.append(dict(model=name, mode="shrunk", val_MAE=shrink_mae, alpha=alpha, fit_seconds=secs))
        lines.append(f"{name:16s} raw val MAE {raw_mae:.3f}; shrunk val MAE {shrink_mae:.3f} (alpha={alpha:.2f})")
        pd.DataFrame(records).to_csv(f"{PREFIX}_validation.csv", index=False)

    val_table = pd.DataFrame(records)
    best_row = val_table.loc[val_table.val_MAE.idxmin()]
    best_model = best_row["model"]
    best_mode = best_row["mode"]
    selected = fitted_val[best_model]
    lines.append(f"Selected on validation: {best_model} / {best_mode}, alpha={best_row.alpha:.2f}")

    test_rows = [dict(model="hist_mean", mode="baseline", **metrics(te.y_f.values, te["hist"].values))]
    original = fitted_val["original"]
    p, _, _ = tm.fit_predict("hgb", tm.hgb_reg(**best_kw), original["feats"], tr, {"val": va, "test": te}, original["target"])
    original_test = p["test"]
    test_rows.append(dict(model="original", mode="raw", **metrics(te.y_f.values, original_test)))
    if best_model == "original" and best_mode == "raw":
        selected_test = original_test
    else:
        p, _, _ = tm.fit_predict("hgb", tm.hgb_reg(**best_kw), selected["feats"], tr, {"test": te}, selected["target"])
        selected_test = p["test"] + (te["hist"].values if selected["is_residual"] else 0)
        if best_mode == "shrunk":
            selected_test = te["hist"].values + best_row.alpha * (selected_test - te["hist"].values)
    test_rows.append(dict(model=str(best_model), mode=str(best_mode), alpha=float(best_row.alpha),
                          **metrics(te.y_f.values, selected_test)))
    pd.DataFrame(test_rows).to_csv(f"{PREFIX}_test.csv", index=False)
    lines.append("October test results (after selection):")
    for r in test_rows:
        lines.append(f"  {r['model']:16s} {r['mode']:8s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f}")

    pred = pd.DataFrame({"hospital": te.hospital, "ts_f": te.ts_f, "actual": te.y_f,
                         "hist_pred": te["hist"], "selected_pred": selected_test})
    pred.to_csv(f"{PREFIX}_predictions.csv", index=False)

    plot = val_table[val_table["mode"] == "raw"].set_index("model").loc[[v[0] for v in variants]]
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=180)
    vals = [mean_absolute_error(va.y_f, va["hist"])] + plot.val_MAE.tolist()
    labels = ["Historical mean"] + ["Original HGB", "Residual target", "Slower history", "Residual + slower"]
    ax.barh(labels, vals, color=["#8b95a1"] + ["#386b8c"] * 4)
    ax.invert_yaxis()
    ax.set_xlabel("September validation MAE (minutes), 24 h ahead")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.13)
    fig.tight_layout()
    fig.savefig(f"{PREFIX}_validation.png", bbox_inches="tight")
    (OUT / "hn_24h_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
