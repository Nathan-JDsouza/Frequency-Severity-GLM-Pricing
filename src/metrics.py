"""Metrics for pricing model comparison: Gini, lift, calibration, deviance.

Gini convention follows the repo's premium.gini_coefficient: policies ordered
by DECREASING score, more negative = more actual loss concentrated at the top
of the ranking. For reporting we use the absolute value (higher = better
separation) and keep the sign consistent with the original analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gini(loss: np.ndarray, score: np.ndarray) -> float:
    """Gini of actual loss under decreasing-score ordering (repo convention, negative)."""
    order = np.argsort(-np.asarray(score), kind="stable")
    y = np.asarray(loss, dtype=float)[order]
    if y.sum() <= 0:
        return 0.0
    n = y.size
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * y) / np.sum(y) - (n + 1)) / n)


def gini_se(pred_a: np.ndarray, pred_b: np.ndarray, loss: np.ndarray, n_boot: int = 200, seed: int = 7):
    """Paired bootstrap SE of the Gini difference between two scores on the same losses."""
    rng = np.random.default_rng(seed)
    n = len(loss)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(gini(loss[idx], pred_a[idx]) - gini(loss[idx], pred_b[idx]))
    diffs = np.array(diffs)
    return float(np.std(diffs)), float(np.mean(diffs))


def lift_table(test: pd.DataFrame, score_col: str = "pure_premium", n_deciles: int = 10) -> pd.DataFrame:
    """Decile lift of actual loss / portfolio average, ordered by decreasing score."""
    work = test.sort_values(score_col, ascending=False).copy()
    work["decile"] = pd.qcut(np.arange(len(work)), n_deciles, labels=False) + 1
    grouped = work.groupby("decile", as_index=False).agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        actual_loss=("actual_loss", "sum"),
        model_premium=(score_col, "sum"),
    )
    overall = work["actual_loss"].sum() / work["Exposure"].sum()
    grouped["actual_rate"] = grouped["actual_loss"] / grouped["exposure"]
    grouped["lift"] = grouped["actual_rate"] / overall
    grouped["loss_ratio"] = grouped["actual_loss"] / grouped["model_premium"]
    grouped["cum_loss_share"] = grouped["actual_loss"].cumsum() / grouped["actual_loss"].sum()
    grouped["cum_exposure_share"] = grouped["exposure"].cumsum() / grouped["exposure"].sum()
    return grouped


def calibration_deciles(test: pd.DataFrame, score_col: str = "pure_premium", n: int = 10) -> pd.DataFrame:
    """Actual/predicted loss by decile of predicted premium (descending)."""
    work = test.copy()
    work["decile"] = pd.qcut(work[score_col].rank(method="first", ascending=False), n, labels=False) + 1
    g = work.groupby("decile", as_index=False).agg(
        exposure=("Exposure", "sum"),
        actual_loss=("actual_loss", "sum"),
        model_premium=(score_col, "sum"),
    )
    g["actual_over_pred"] = g["actual_loss"] / g["model_premium"]
    g["actual_rate"] = g["actual_loss"] / g["exposure"]
    g["pred_rate"] = g["model_premium"] / g["exposure"]
    return g


def mean_abs_error(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.asarray(y) - np.asarray(yhat)
    return float(np.sqrt(np.mean(d ** 2)))


def out_of_sample_deviance(y, mu, family: str, alpha=None, power=None) -> float:
    """Family-appropriate deviance on validation data (used in CV)."""
    from src.fastirls import _deviance, tweedie_deviance

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if family == "poisson":
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        return float(2.0 * (t - (y - mu)).sum())
    if family == "nb2":
        return _deviance(y, mu, "nb2", alpha or 1.0)
    if family == "gamma":
        return _deviance(y, mu, "gamma")
    if family == "tweedie":
        return tweedie_deviance(y, mu, power or 1.5)
    raise ValueError(family)
