"""Generate all charts for the extended analysis from saved outputs.

Run:  python run_charts.py
Reads outputs_extended/*.csv|json, writes outputs_extended/*.png.
The Person-residual chart refits the Poisson on train (0.4s with fast IRLS).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.charts_extended import (
    plot_calibration_curves,
    plot_cv_gini_boxplot,
    plot_dispersion_scatter,
    plot_relativity_comparison,
    plot_tail_comparison,
    plot_tweedie_grid,
)
from src.data import load_frequency, load_severity
from src.features import add_rating_factors, glm_formula
from src.models import FrequencyModel, build_design, drop_empty_levels
from run_extended import load_data, split_train_test

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_extended"

_NAVY = "#1f4e79"
_TEAL = "#2a9d8f"
_PURPLE = "#7d6ba0"
_GRAY = "#888888"


def main() -> None:
    t1 = json.loads((OUT / "task1_diagnostics.json").read_text())
    grid = pd.read_csv(OUT / "task2_summary.csv")
    cv = pd.read_csv(OUT / "task3_cv_folds.csv")
    tails = pd.read_csv(OUT / "holdout_tails.csv")
    rel = pd.read_csv(OUT / "holdout_relativities.csv")
    cal_tables = {
        "poisson+gamma": pd.read_csv(OUT / "holdout_calibration_poisson_gamma.csv"),
        "negbin+gamma": pd.read_csv(OUT / "holdout_calibration_negbin_gamma.csv"),
        "tweedie": pd.read_csv(OUT / "holdout_calibration_tweedie.csv"),
    }
    tw_vs_fs = pd.read_csv(OUT / "holdout_tweedie_vs_fs_deciles.csv")

    # --- Task 1: person residuals (refit poisson on train) ---
    freq, sev = load_data()
    tr_mask, _ = split_train_test(freq)
    y, X, names = build_design(glm_formula("ClaimNb"), freq)
    X, names = drop_empty_levels(X, names)
    off = np.log(freq["Exposure"].clip(lower=1e-6).to_numpy())
    pois = FrequencyModel("poisson").fit(X[tr_mask], y[tr_mask], off[tr_mask])
    p1 = plot_dispersion_scatter(pois.mu, y[tr_mask], OUT)
    # re-draw with actual alpha from json
    print("wrote", p1)

    # --- relativities comparison ---
    p2 = plot_relativity_comparison(rel, OUT)
    print("wrote", p2)

    # --- tweedie grid ---
    p3 = plot_tweedie_grid(grid, OUT)
    print("wrote", p3)

    # --- CV gini boxplot ---
    p4 = plot_cv_gini_boxplot(cv, OUT)
    print("wrote", p4)

    # --- calibration curves ---
    p5 = plot_calibration_curves(cal_tables, OUT)
    print("wrote", p5)

    # --- tails ---
    p6 = plot_tail_comparison(tails, OUT)
    print("wrote", p6)

    # --- tweedie vs freq*sev premium ratio by decile (tail pricing story) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(tw_vs_fs["decile_fs"], tw_vs_fs["ratio_tw_over_fs"], "o-", color=_PURPLE, label="Tweedie / freq-sev premium")
    ax.plot(tw_vs_fs["decile_fs"], tw_vs_fs["actual_over_fs"], "o-", color=_NAVY, label="Actual / freq-sev premium")
    ax.axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("Decile of freq-sev premium (1 = highest)")
    ax.set_ylabel("Ratio")
    ax.set_title("Tweedie prices the top risks ~1.6-1.8x the frequency-severity model")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p7 = OUT / "holdout_tweedie_vs_fs.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    print("wrote", p7)

    # --- train vs valid gini (overfitting gap) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = cv["model"].unique()
    w = 0.35
    x = np.arange(len(models))
    tr = [-cv.groupby("model")["train_gini"].mean()[m] for m in models]
    va = [-cv.groupby("model")["gini"].mean()[m] for m in models]
    ax.bar(x - w / 2, tr, width=w, color=_TEAL, label="Train (in-sample)")
    ax.bar(x + w / 2, va, width=w, color=_NAVY, label="Validation (CV folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("|Gini|")
    ax.set_title("Overfitting gap: train vs validation Gini (repeated CV)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p8 = OUT / "task3_overfitting_gap.png"
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    print("wrote", p8)


if __name__ == "__main__":
    main()
