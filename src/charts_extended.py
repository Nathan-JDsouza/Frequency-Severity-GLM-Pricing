"""Charts for the extended analysis (Tasks 1-3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_NAVY = "#1f4e79"
_TEAL = "#2a9d8f"
_ORANGE = "#e76f51"
_GRAY = "#888888"
_GREEN = "#6a994e"
_PURPLE = "#7d6ba0"
_COLORS = {
    "poisson": _NAVY,
    "poisson+gamma": _NAVY,
    "quasipoisson+gamma": _NAVY,
    "negbin+gamma": _TEAL,
    "tweedie": _PURPLE,
}

_FACTOR_ORDER = {
    "VehPowerBand": ["4-", "5", "6", "7", "8-9", "10-15", "16+"],
    "VehAgeBand": ["0", "1", "2-4", "5-10", "11+"],
    "DrivAgeBand": ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"],
    "BonusMalusBand": ["50", "51-60", "61-80", "81-100", "101-150"],
}

TERM_RE = None  # populated lazily


def _term_split(term: str):
    """'C(Factor)[T.level]' -> ('Factor', 'level'); 'LogDensity' -> (None, 'LogDensity')."""
    import re

    global TERM_RE
    if TERM_RE is None:
        TERM_RE = re.compile(r"^C\(([^)]+)\)\[T\.(.+)\]$")
    m = TERM_RE.match(term)
    if m:
        return m.group(1), m.group(2)
    return None, term


def _order_levels(factor, levels):
    order = _FACTOR_ORDER.get(factor)
    if not order:
        return list(levels)
    return [lv for lv in order if lv in set(levels)]


def plot_dispersion_scatter(poisson_mu, y, out_dir, n_sample=20000):
    """Person residual diagnostics: (y - mu)/sqrt(mu) vs mu, log-x."""
    rng = np.random.default_rng(7)
    idx = rng.choice(len(y), min(n_sample, len(y)), replace=False)
    mu = poisson_mu[idx]
    r = (y[idx] - mu) / np.sqrt(mu)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(mu, r, s=2, alpha=0.25, color=_NAVY, edgecolors="none")
    axes[0].set_xscale("log")
    axes[0].axhline(0, color=_GRAY, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Poisson fitted mu")
    axes[0].set_ylabel("Person (Pearson) residual")
    axes[0].set_title("Person residuals vs fitted mean (log x)")

    # expected residual variance by mu bucket under Poisson, NB2, observed
    edges = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    mu_full = poisson_mu
    y_full = y
    centers, obs, po, nb = [], [], [], []
    alpha = 0.791352
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mu_full >= lo) & (mu_full < hi)
        if not m.any():
            continue
        centers.append(np.sqrt(lo * hi))
        obs.append(((y_full[m] - mu_full[m]) ** 2 / mu_full[m]).mean())
        po.append(1.0)
        nb.append(1.0 + alpha * mu_full[m].mean())
    axes[1].plot(centers, obs, "o-", color=_NAVY, label="Observed (y-mu)^2/mu")
    axes[1].plot(centers, po, "--", color=_GRAY, label="Poisson: Var = mu")
    axes[1].plot(centers, nb, "--", color=_TEAL, label=f"NB2: Var = mu + {alpha:.2f} mu^2")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sqrt(mu bucket lo*hi), log scale")
    axes[1].set_ylabel("mean squared Pearson residual")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Where the extra variance lives")
    fig.tight_layout()
    path = out_dir / "task1_person_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_relativity_comparison(rel: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bars: exp(beta) under Poisson vs NB2 vs Tweedie for main factors."""
    factors = ["BonusMalusBand", "DrivAgeBand", "VehAgeBand", "Area"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, factor in zip(axes.ravel(), factors):
        sub = rel[rel["term"].str.contains(factor) | (rel["term"] == "Intercept")].copy() if "term" in rel.columns else rel
        if "factor" not in rel.columns:
            rel2 = rel.copy()
            rel2[["factor", "level"]] = rel2["term"].apply(lambda t: pd.Series(_term_split(t)))
            sub = rel2[rel2["factor"] == factor].copy()
        else:
            sub = rel[rel["factor"] == factor].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        levels = _order_levels(factor, sub["level"].unique())
        sub = sub.set_index("level").loc[levels].reset_index()
        x = np.arange(len(sub))
        w = 0.35
        cols, labels = [], []
        for col, lab in [
            ("rel_poisson", "Poisson"),
            ("rel_nb2", "NB2"),
            ("rel_tweedie", "Tweedie"),
        ]:
            if col in sub.columns:
                cols.append(col)
                labels.append(lab)
        for i, (col, lab) in enumerate(zip(cols, labels)):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, sub[col], width=w * 0.9, label=lab,
                   color=[_NAVY, _TEAL, _PURPLE][i])
        ax.axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        stats_labels = [str(l) for l in sub["level"]]
        ax.set_xticklabels(stats_labels, rotation=45, ha="right")
        ax.set_title(factor)
        ax.set_ylabel("Relativity vs base")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Rating relativities: frequency (Poisson vs NB2) and pure premium (Tweedie)", y=1.0)
    fig.tight_layout()
    path = out_dir / "relativities_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tweedie_grid(summary: pd.DataFrame, out_dir: Path) -> Path:
    """CV deviance vs power, with Gini and loss ratio panels."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    s = summary.sort_values("power")
    axes[0].plot(s["power"], s["mean_deviance"], "o-", color=_NAVY)
    best = s.loc[s["mean_deviance"].idxmin()]
    axes[0].axvline(best["power"], color=_ORANGE, linestyle="--", label=f"best p = {best['power']:.1f}")
    axes[0].set_xlabel("Tweedie power p")
    axes[0].set_ylabel("mean CV deviance per policy")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[0].set_title("CV deviance vs power")

    axes[1].plot(s["power"], -s["mean_gini"], "o-", color=_NAVY, label="Gini (abs)")
    gbest = s.loc[(-s["mean_gini"]).idxmax()]
    axes[1].axvline(gbest["power"], color=_ORANGE, linestyle="--", label=f"best Gini p = {gbest['power']:.1f}")
    axes[1].set_xlabel("Tweedie power p")
    axes[1].set_ylabel("|Gini| (validation)")
    axes[1].legend(frameon=False)
    axes[1].set_title("Ranking skill vs power")

    axes[2].plot(s["power"], s["mean_loss_ratio"], "o-", color=_NAVY)
    axes[2].axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
    axes[2].set_xlabel("Tweedie power p")
    axes[2].set_ylabel("validation loss ratio (A/P)")
    axes[2].set_title("Level bias vs power")
    fig.tight_layout()
    path = out_dir / "task2_tweedie_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cv_gini_boxplot(cv: pd.DataFrame, out_dir: Path) -> Path:
    """Gini by model across folds/repeats with fold jitter."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = cv["model"].unique().tolist()
    data = [-cv.loc[cv["model"] == m, "gini"] for m in models]
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True, widths=0.5)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(_COLORS.get(m, _GRAY))
        patch.set_alpha(0.6)
    ax.set_ylabel("|Gini| (validation, higher = better)")
    ax.set_title("Repeated CV: Gini by model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out_dir / "task3_gini_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration_curves(cal_tables: dict, out_dir: Path, best_power=None) -> Path:
    """A/P by decile for the holdout models."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, cal in cal_tables.items():
        color = _PURPLE if "tweedie" in label else _TEAL if "negbin" in label else _NAVY
        ax.plot(cal["decile"], cal["actual_over_pred"], "o-", label=label, color=color)
    ax.axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("Decile of predicted premium (1 = highest)")
    ax.set_ylabel("Actual / predicted loss")
    ax.set_title("Holdout calibration by decile")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "holdout_calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_tail_comparison(tails: pd.DataFrame, out_dir: Path) -> Path:
    """Tail lift by model: top 10% / 1% / 0.1% slices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for model, sub in tails.groupby("model"):
        sub = sub.sort_values("slice")
        color = _PURPLE if "tweedie" in model else _TEAL if "negbin" in model else _NAVY
        axes[0].plot(sub["slice"], sub["lift"], "o-", label=model, color=color)
    axes[0].set_xlabel("Tail slice")
    axes[0].set_ylabel("Loss-rate lift vs portfolio")
    axes[0].set_title("Top-tail lift (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].tick_params(axis="x", rotation=20)

    # right panel: A/P by slice
    for model, sub in tails.groupby("model"):
        sub = sub.sort_values("slice")
        color = _PURPLE if "tweedie" in model else _TEAL if "negbin" in model else _NAVY
        axes[1].plot(sub["slice"], sub["actual_over_pred"], "o-", label=model, color=color)
    axes[1].axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Tail slice")
    axes[1].set_ylabel("Actual / predicted loss")
    axes[1].set_title("Tail calibration (A/P)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out_dir / "holdout_tails.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
