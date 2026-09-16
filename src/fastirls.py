"""Fast IRLS for log-link GLMs used in cross-validation loops.

statsmodels' GLM IRLS re-does a full pinv/SVD each iteration (~50s on 542k
rows). For the NB2 profile likelihood over alpha and the Tweedie power grid
we need many fits, so this module implements plain Fisher scoring via normal
equations (X'W X) beta = X'W z with a Cholesky solve, warm-started. Only log
link, single-parameter families, no weights (exposure enters as offset).

Families:
  poisson : V(mu) = mu
  nb2     : V(mu) = mu + alpha*mu^2   (alpha fixed per fit)
  gamma   : V(mu) = mu^2               (severity; no offset)
  tweedie : V(mu) = mu^p               (1 < p < 2, compound Poisson)

Validated against statsmodels GLM (deviance and coefficients agree to ~1e-4).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln


def _irls(X, y, offset, family, alpha=1.0, power=1.5, beta=None, max_iter=100, tol=1e-10, miniter=2):
    """Fisher scoring for log-link GLM. Returns (beta, mu, deviance, n_iter)."""
    n, k = X.shape
    if beta is None:
        beta = np.zeros(k)
        beta[0] = np.log(max(y.sum(), 0.5) / np.exp(offset).sum())
    eta = X @ beta + offset
    mu = np.exp(eta)
    dev_old = np.inf
    dev = np.nan
    it = 0
    for it in range(1, max_iter + 1):
        if family == "poisson":
            w = mu
        elif family == "nb2":
            w = mu / (1.0 + alpha * mu)
        elif family == "gamma":
            w = np.ones_like(mu)  # V=mu^2, (dmu/deta)^2/V = 1
        elif family == "tweedie":
            w = np.power(mu, 2.0 - power)
        else:
            raise ValueError(family)
        z = eta - offset + (y - mu) / mu
        XtW = X.T * w
        A = XtW @ X
        b = XtW @ z
        try:
            c, low = cho_factor(A, lower=True)
            beta = cho_solve((c, low), b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
        eta = X @ beta + offset
        mu = np.exp(eta)
        dev = _deviance(y, mu, family, alpha, power)
        if it > miniter and (np.isnan(dev) or abs(dev_old - dev) / (abs(dev) + 1e-12) < tol):
            break
        dev_old = dev
    return beta, mu, dev, it


def _deviance(y, mu, family, alpha=1.0, power=1.5):
    if family == "poisson":
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        return float(2.0 * (t - (y - mu)).sum())
    if family == "nb2":
        r = 1.0 / alpha
        t1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        t2 = (y + r) * np.log((y + r) / (mu + r))
        return float(2.0 * (t1 - t2).sum())
    if family == "gamma":
        return float(2.0 * (-np.log(y / mu) + (y - mu) / mu).sum())
    if family == "tweedie":
        return tweedie_deviance(y, mu, power)
    raise ValueError(family)


def tweedie_deviance(y, mu, power):
    """Sum of Tweedie unit deviances, 1 < p < 2 (y-term included; see tweedie.py)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    p = float(power)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y ** (2.0 - p) / ((1.0 - p) * (2.0 - p)), 0.0)
        t2 = y * mu ** (1.0 - p) / (1.0 - p)
        t3 = mu ** (2.0 - p) / (2.0 - p)
    d = 2.0 * (t1 - t2 + t3)
    return float(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).sum())


def nb2_loglike(y, mu, alpha):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    r = 1.0 / alpha
    return float(
        (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
         + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))).sum()
    )


def poisson_loglike(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return float((y * np.log(mu) - mu - gammaln(y + 1.0)).sum())


def profile_nb2_alpha(X, y, offset, alpha_lo=1e-4, alpha_hi=5.0, tol=1e-6, warm=None):
    """Maximize the NB2 profile log-likelihood over alpha (Brent, log scale).

    Returns (alpha_hat, beta_hat, mu_hat, loglike_hat, n_evals).
    """
    from scipy.optimize import minimize_scalar

    state = {"beta": warm, "n": 0}

    def negll(log_alpha):
        a = float(np.exp(log_alpha))
        beta, mu, dev, _ = _irls(X, y, offset, "nb2", alpha=a, beta=state["beta"])
        ll = nb2_loglike(y, mu, a)
        state["beta"] = beta
        state["n"] += 1
        return -ll

    res = minimize_scalar(negll, bounds=(np.log(alpha_lo), np.log(alpha_hi)), method="bounded",
                          options={"xatol": tol})
    alpha_hat = float(np.exp(res.x))
    beta, mu, dev, _ = _irls(X, y, offset, "nb2", alpha=alpha_hat, beta=state["beta"])
    ll = nb2_loglike(y, mu, alpha_hat)
    return alpha_hat, beta, mu, ll, state["n"]


def glm_cov(X, w):
    """Sandwich-free GLM covariance (X'WX)^-1 * scale (scale applied by caller)."""
    XtW = X.T * w
    A = XtW @ X
    try:
        c, low = cho_factor(A, lower=True)
        inv = cho_solve((c, low), np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(A)
    return inv
