"""Overdispersion diagnostics for the Poisson frequency GLM (Task 1).

Statistics
----------
- Deviance ratio:            D / (n - p)
- Pearson chi-square ratio:  X² / df = mean of squared Pearson residuals
- Person / standardized Pearson residuals: r_P,i = (y_i - mu_i)/sqrt(mu_i)

On sparse count data (mean mu ~ 0.05) the deviance ratio is a biased scale
estimator -- its null distribution concentrates near 1 - k/n - E[mu] (see
run_extended.py: parametric bootstrap and analytic approximation), so the
Pearson ratio is the decisive statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dispersion_statistics(y, mu, df_resid: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    deviance = float(2.0 * (t - (y - mu)).sum())
    pearson_chi2 = float((((y - mu) ** 2) / mu).sum())
    return {
        "deviance": deviance,
        "pearson_chi2": pearson_chi2,
        "df_resid": int(df_resid),
        "deviance_ratio": deviance / df_resid,
        "pearson_ratio": pearson_chi2 / df_resid,
    }


def pearson_residuals(y, mu) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return (y - mu) / np.sqrt(mu)


def pearson_ratio_by_mu_bucket(y, mu, edges=(0.0, 0.02, 0.05, 0.1, 0.2, 1.0)) -> pd.DataFrame:
    """Where the Pearson chi-square comes from: mean squared Pearson residual by mu band."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mu >= lo) & (mu < hi)
        if not m.any():
            continue
        p = ((y[m] - mu[m]) ** 2 / mu[m]).sum()
        rows.append(
            {
                "mu_band": f"[{lo:g},{hi:g})",
                "n": int(m.sum()),
                "pearson_chi2": float(p),
                "mean_sq_pearson": float(p / m.sum()),
                "mean_mu": float(mu[m].mean()),
            }
        )
    return pd.DataFrame(rows)
