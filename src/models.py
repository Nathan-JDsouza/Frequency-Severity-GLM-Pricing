"""Model zoo for cross-validation: Poisson / quasi-Poisson / NB2 frequency + Gamma severity + Tweedie.

All frequency and Tweedie fits use the fast IRLS in fastirls.py (validated
against statsmodels to ~1e-4 on this dataset). Quasi-Poisson shares the
Poisson mean fit and differs only through phi-scaled standard errors, so in
CV it is scored by the same out-of-sample deviance as Poisson (its predicted
means are identical); its role in the report is inference (SEs, p-values).

NB2 fits estimate alpha by profile MLE on the fit fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy

from src.fastirls import (
    _irls,
    glm_cov,
    nb2_loglike,
    poisson_loglike,
    profile_nb2_alpha,
    tweedie_deviance,
    _deviance,
)

FREQ_FAMILIES = ["poisson", "quasipoisson", "negbin"]


def build_design(formula: str, frame: pd.DataFrame):
    y, X = patsy.dmatrices(formula, frame, return_type="dataframe")
    return np.asarray(y).ravel(), np.asarray(X), list(X.columns)


def drop_empty_levels(X: np.ndarray, names: list[str]):
    """Drop all-zero dummy columns (factor levels absent from the fit frame)."""
    keep = np.where(np.abs(X).max(axis=0) > 0)[0]
    return X[:, keep], [names[i] for i in keep]


class FrequencyModel:
    """Log-link count GLM with log-exposure offset. family in {poisson, nb2}."""

    def __init__(self, family: str, alpha: float | None = None):
        self.family = family
        self.alpha = alpha

    def fit(self, X, y, offset):
        if self.family == "poisson":
            self.beta, self.mu, self.dev, self.n_iter = _irls(X, y, offset, "poisson")
        elif self.family == "nb2":
            alpha, beta, mu, ll, n = profile_nb2_alpha(X, y, offset)
            self.alpha = alpha
            self.beta, self.mu, self.dev = beta, mu, _deviance(y, mu, "nb2", alpha)
            self.loglike = ll
        else:
            raise ValueError(self.family)
        # phi = Pearson ratio on the fit sample
        self.phi = float((((y - self.mu) ** 2) / self.mu).sum() / (len(y) - X.shape[1]))
        return self

    def predict(self, X, offset):
        return np.exp(X @ self.beta + offset)

    def cov(self, X, scale=None):
        if self.family == "poisson":
            w = self.mu
        elif self.family == "nb2":
            w = self.mu / (1.0 + self.alpha * self.mu)
        else:
            raise ValueError(self.family)
        return glm_cov(X, w) * (scale if scale is not None else 1.0)


class GammaSeverityModel:
    """Gamma GLM, log link, claim amounts; no offset (weights = 1)."""

    def fit(self, X, y, offset=None):
        offset = np.zeros(len(y)) if offset is None else offset
        self.beta, self.mu, self.dev, self.n_iter = _irls(X, y, offset, "gamma")
        self.phi = float((2.0 * (-np.log(y / self.mu) + (y - self.mu) / self.mu)).sum() / (len(y) - X.shape[1]))
        return self

    def predict(self, X, offset=None):
        return np.exp(X @ self.beta)


class TweedieModel:
    """Tweedie GLM, log link, offset = log(Exposure). Response: PurePremium."""

    def __init__(self, power: float):
        self.power = float(power)

    def fit(self, X, y, offset):
        self.beta, self.mu, self.dev, self.n_iter = _irls(X, y, offset, "tweedie", power=self.power)
        # Pearson-style dispersion: sum((y-mu)^2 / mu^p) / df
        w = np.power(self.mu, self.power)
        self.phi = float((((y - self.mu) ** 2) / w).sum() / (len(y) - X.shape[1]))
        return self

    def predict(self, X, offset):
        return np.exp(X @ self.beta + offset)

    def cov(self, X, scale=None):
        w = np.power(self.mu, 2.0 - self.power)
        return glm_cov(X, w) * (scale if scale is not None else 1.0)
