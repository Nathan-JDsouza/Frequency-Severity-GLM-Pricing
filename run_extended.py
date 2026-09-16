"""Experiment engine: diagnostics, Tweedie grid search, repeated CV, holdout eval.

Run:  python run_extended.py [--repeats 3] [--folds 5] [--quick]

Outputs land in outputs_extended/. All fits use the fast IRLS (fastirls.py,
validated to ~1e-4 against statsmodels on this dataset).

Steps
-----
1. Refit repo models on the repo's 80/20 split (seed 42, stratified on
   ClaimNb>0): Poisson frequency + Gamma severity; run overdispersion
   diagnostics (deviance ratio, Pearson ratio, Person residuals,
   parametric bootstrap null); fit quasi-Poisson and NB2.
2. Tweedie grid over p in {1.1,...,1.9}: 5-fold CV on the training split,
   scored by mean validation Tweedie deviance plus p-agnostic metrics.
3. Repeated (3x) 5-fold CV comparing poisson+gamma, quasipoisson+gamma,
   negbin+gamma, tweedie(p*): Gini, loss ratio, MSE, decile calibration,
   family deviances, train/valid gap for overfitting.
4. Holdout evaluation: paired bootstrap Gini tests, decile calibration,
   tail analysis, rating relativity comparison across models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy.special import ndtr
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

from src.data import load_frequency, load_severity
from src.features import add_rating_factors, glm_formula
from src.premium import actual_loss_by_policy
from src.diagnostics import (
    dispersion_statistics,
    pearson_ratio_by_mu_bucket,
    pearson_residuals,
)
from src.fastirls import (
    _deviance,
    glm_cov,
    nb2_loglike,
    poisson_loglike,
    profile_nb2_alpha,
    tweedie_deviance,
)
from src.metrics import calibration_deciles, gini, gini_se, lift_table, rmse
from src.models import FrequencyModel, GammaSeverityModel, TweedieModel, build_design, drop_empty_levels

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_extended"
SEED = 42
POWERS = [round(0.1 * k, 1) for k in range(11, 20)]  # 1.1 ... 1.9


def log(*a):
    print(*a, flush=True)


def p_two_sided(z):
    return float(2.0 * (1.0 - ndtr(abs(z))))


# ------------------------------------------------------------------ data


def load_data():
    freq = add_rating_factors(load_frequency(ROOT / "data/raw/freMTPL2freq.csv"))
    sev = load_severity(ROOT / "data/raw/freMTPL2sev.csv")
    freq["actual_loss"] = actual_loss_by_policy(freq, sev).to_numpy()
    freq["PurePremium"] = freq["actual_loss"]
    return freq, sev


def split_train_test(freq):
    train_ids, test_ids = train_test_split(
        freq["IDpol"],
        test_size=0.2,
        random_state=SEED,
        stratify=(freq["ClaimNb"] > 0).astype(int),
    )
    tr_mask = freq["IDpol"].isin(train_ids).to_numpy()
    return tr_mask, ~tr_mask


def build_claim_frame(freq, sev):
    """Claim-level rows with rating factors, UNCAPPED amounts (cap applied per fit)."""
    drop = [c for c in ("ClaimNb", "Exposure") if c in freq.columns]
    claims = sev.merge(freq.drop(columns=drop), on="IDpol", how="inner")
    claims = claims.loc[claims["ClaimAmount"] > 0].copy()
    return claims


# ------------------------------------------------------------------ task 1


def task1(freq, tr_mask, n_boot):
    log("=== TASK 1: overdispersion diagnostics ===")
    y, X, names = build_design(glm_formula("ClaimNb"), freq)
    X, names = drop_empty_levels(X, names)
    off = np.log(freq["Exposure"].clip(lower=1e-6).to_numpy())
    Xtr, ytr, otr = X[tr_mask], y[tr_mask], off[tr_mask]

    t0 = time.time()
    pois = FrequencyModel("poisson").fit(Xtr, ytr, otr)
    log(f"poisson fit {time.time()-t0:.1f}s dev={pois.dev:,.0f} df={len(ytr)-Xtr.shape[1]:,}")

    stats = dispersion_statistics(ytr, pois.mu, len(ytr) - Xtr.shape[1])
    buckets = pearson_ratio_by_mu_bucket(ytr, pois.mu)
    log(f"deviance ratio {stats['deviance_ratio']:.4f}; pearson ratio {stats['pearson_ratio']:.4f}")

    rng = np.random.default_rng(SEED)
    dev_ratios = np.empty(n_boot)
    pearson_ratios = np.empty(n_boot)
    t0 = time.time()
    for b in range(n_boot):
        y_sim = rng.poisson(pois.mu)
        s = dispersion_statistics(y_sim, pois.mu, len(y_sim) - Xtr.shape[1])
        dev_ratios[b] = s["deviance_ratio"]
        pearson_ratios[b] = s["pearson_ratio"]
    log(f"bootstrap null ({n_boot} reps) {time.time()-t0:.1f}s: dev ratio {dev_ratios.mean():.4f}+-{dev_ratios.std():.4f}, pearson {pearson_ratios.mean():.4f}+-{pearson_ratios.std():.4f}")
    z_dev = (stats["deviance_ratio"] - dev_ratios.mean()) / dev_ratios.std()
    z_pearson = (stats["pearson_ratio"] - pearson_ratios.mean()) / pearson_ratios.std()

    phi = pois.phi
    log(f"quasi-Poisson phi = {phi:.4f}")

    t0 = time.time()
    nb = FrequencyModel("nb2").fit(Xtr, ytr, otr)
    log(f"nb2 fit {time.time()-t0:.1f}s alpha={nb.alpha:.5f} ll={nb.loglike:,.2f} (poisson ll={poisson_loglike(ytr, pois.mu):,.2f})")

    cov_p = pois.cov(Xtr)
    se_p = np.sqrt(np.diag(cov_p))
    se_q = se_p * np.sqrt(phi)
    cov_nb = nb.cov(Xtr)
    se_nb = np.sqrt(np.diag(cov_nb))

    rel = pd.DataFrame(
        {
            "term": names,
            "rel_poisson": np.exp(pois.beta),
            "rel_nb2": np.exp(nb.beta),
            "rel_ratio_nb_over_pois": np.exp(nb.beta - pois.beta),
            "se_poisson": se_p,
            "se_quasi": se_q,
            "se_nb2": se_nb,
            "z_poisson": pois.beta / se_p,
            "z_quasi": pois.beta / se_q,
            "z_nb2": nb.beta / se_nb,
        }
    )
    rel["p_poisson"] = [p_two_sided(z) for z in rel["z_poisson"]]
    rel["p_quasi"] = [p_two_sided(z) for z in rel["z_quasi"]]
    rel["p_nb2"] = [p_two_sided(z) for z in rel["z_nb2"]]
    rel["sig_poisson_5pct"] = rel["p_poisson"] < 0.05
    rel["sig_quasi_5pct"] = rel["p_quasi"] < 0.05
    rel["sig_nb2_5pct"] = rel["p_nb2"] < 0.05
    rel.to_csv(OUT / "task1_relativities.csv", index=False)

    out = {
        "stats": stats,
        "pearson_by_mu_band": buckets.to_dict(orient="records"),
        "bootstrap": {
            "B": n_boot,
            "deviance_ratio_mean": float(dev_ratios.mean()),
            "deviance_ratio_sd": float(dev_ratios.std()),
            "pearson_ratio_mean": float(pearson_ratios.mean()),
            "pearson_ratio_sd": float(pearson_ratios.std()),
        },
        "z_scores": {"deviance_ratio": float(z_dev), "pearson_ratio": float(z_pearson)},
        "phi_quasi": phi,
        "nb2_alpha": nb.alpha,
        "nb2_loglike": nb.loglike,
        "poisson_loglike": poisson_loglike(ytr, pois.mu),
        "n_terms": len(names),
        "n_sig_poisson_5pct": int(rel["sig_poisson_5pct"].sum()),
        "n_sig_quasi_5pct": int(rel["sig_quasi_5pct"].sum()),
        "n_sig_nb2_5pct": int(rel["sig_nb2_5pct"].sum()),
        "max_rel_ratio_nb_vs_pois": float(rel.loc[rel.term != "Intercept", "rel_ratio_nb_over_pois"].max()),
        "min_rel_ratio_nb_vs_pois": float(rel.loc[rel.term != "Intercept", "rel_ratio_nb_over_pois"].min()),
    }
    (OUT / "task1_diagnostics.json").write_text(json.dumps(out, indent=2))
    return out, pois, nb, X, y, off, names


# ------------------------------------------------------------------ task 2


def task2(freq, tr_mask, folds, tag):
    log("=== TASK 2: Tweedie power grid (5-fold CV within train) ===")
    y, X, names = build_design(glm_formula("PurePremium"), freq)
    X, names = drop_empty_levels(X, names)
    off = np.log(freq["Exposure"].clip(lower=1e-6).to_numpy())
    y_tr, X_tr, off_tr = y[tr_mask], X[tr_mask], off[tr_mask]
    strat = (freq["ClaimNb"] > 0).astype(int).to_numpy()[tr_mask]

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=1, random_state=SEED)
    splits = list(rskf.split(X_tr, strat))
    rows = []
    t0 = time.time()
    for p in POWERS:
        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            Xf, nf_ = drop_empty_levels(X_tr[tr_idx], names)
            m = TweedieModel(p).fit(Xf, y_tr[tr_idx], off_tr[tr_idx])
            # predict validation with same column subset
            mu_va = m.predict(X_tr[va_idx][:, [names.index(n) for n in nf_]], off_tr[va_idx])
            actual = y_tr[va_idx]
            dev = tweedie_deviance(actual, mu_va, p)
            # train predictions for overfitting gap
            mu_tr = m.predict(Xf, off_tr[tr_idx])
            dev_tr = tweedie_deviance(y_tr[tr_idx], mu_tr, p)
            rows.append(
                {
                    "power": p,
                    "fold": fold_i,
                    "mean_deviance": dev / len(va_idx),
                    "train_mean_deviance": dev_tr / len(tr_idx),
                    "gini": gini(actual, mu_va),
                    "train_gini": gini(y_tr[tr_idx], mu_tr),
                    "loss_ratio": float(actual.sum() / mu_va.sum()),
                    "mse": float(np.mean((actual - mu_va) ** 2)),
                    "n_iter": m.n_iter,
                    "dispersion": m.phi,
                }
            )
        done = [r for r in rows if r["power"] == p]
        log(f"p={p:.1f}: mean deviance {np.mean([r['mean_deviance'] for r in done]):,.1f} gini {np.mean([r['gini'] for r in done]):.4f} ({time.time()-t0:.0f}s)")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUT / f"task2_grid_folds{tag}.csv", index=False)
    summary = (
        fold_df.groupby("power")
        .agg(
            mean_deviance=("mean_deviance", "mean"),
            sd_deviance=("mean_deviance", "std"),
            train_mean_deviance=("train_mean_deviance", "mean"),
            mean_gini=("gini", "mean"),
            sd_gini=("gini", "std"),
            train_gini=("train_gini", "mean"),
            mean_loss_ratio=("loss_ratio", "mean"),
            mean_mse=("mse", "mean"),
            mean_iters=("n_iter", "mean"),
            mean_dispersion=("dispersion", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / f"task2_summary{tag}.csv", index=False)
    best = float(summary.loc[summary["mean_deviance"].idxmin(), "power"])
    log(f"best power by CV deviance: {best}")
    return summary, best, fold_df


# ------------------------------------------------------------------ task 3


def task3(freq, sev, tr_mask, best_power, folds, repeats, tag):
    log("=== TASK 3: repeated CV model comparison ===")
    y, X, names = build_design(glm_formula("ClaimNb"), freq)
    X, names = drop_empty_levels(X, names)
    off = np.log(freq["Exposure"].clip(lower=1e-6).to_numpy())

    # claim-level rows and a policy-frame design with the SAME formula/columns
    claims = build_claim_frame(freq, sev)
    sev_formula = glm_formula("ClaimAmount")
    # policy-frame design (dummy response) guarantees identical columns for
    # fitting claims and predicting policies
    _, X_pol_sev, sev_names = build_design(sev_formula, freq.assign(ClaimAmount=1.0))
    X_pol_sev = np.asarray(X_pol_sev)
    # map claims to positional index in freq frame
    pos = {pid: i for i, pid in enumerate(freq["IDpol"].to_numpy())}
    claim_pos = np.array([pos[pid] for pid in claims["IDpol"].to_numpy()])
    amounts = claims["ClaimAmount"].to_numpy()  # uncapped
    X_claim_sev = X_pol_sev[claim_pos]  # claim rows of the policy-frame design

    strat = (freq["ClaimNb"] > 0).astype(int).to_numpy()
    train_positions = np.where(tr_mask)[0]
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    rows = []
    t0 = time.time()
    n_fits = 0
    for rep_i, (tr_sub, va_sub) in enumerate(rskf.split(train_positions, strat[train_positions])):
        tr_idx = train_positions[tr_sub]  # positions in the full freq frame
        va_idx = train_positions[va_sub]
        # policy index sets
        tr_pos, va_pos = tr_idx, va_idx
        tr_mask_fold = np.zeros(len(freq), dtype=bool)
        tr_mask_fold[tr_pos] = True
        Xtr, ytr, otr = X[tr_pos], y[tr_pos], off[tr_pos]
        Xva, yva, ova = X[va_pos], y[va_pos], off[va_pos]

        # --- frequency models ---
        freq_models = {}
        pm = FrequencyModel("poisson").fit(Xtr, ytr, otr)
        freq_models["poisson"] = pm
        freq_models["quasipoisson"] = pm  # same mean fit; phi affects SEs only
        nbm = FrequencyModel("nb2").fit(Xtr, ytr, otr)
        freq_models["negbin"] = nbm

        # --- severity on fold-train claims (cap q=0.995 of fold-train amounts) ---
        fold_claim_mask = tr_mask_fold[claim_pos]
        cap = float(np.quantile(amounts[fold_claim_mask], 0.995))
        y_sev = np.minimum(amounts, cap)
        X_sev_fit, sev_fit_names = drop_empty_levels(X_claim_sev[fold_claim_mask], sev_names)
        sev_model = GammaSeverityModel().fit(X_sev_fit, y_sev[fold_claim_mask])
        keep_idx = [i for i, n in enumerate(sev_names) if n in set(sev_fit_names)]
        sev_hat_va = sev_model.predict(X_pol_sev[va_pos][:, keep_idx])

        # --- tweedie ---
        pp = freq["PurePremium"].to_numpy()
        Xf, nf_ = drop_empty_levels(Xtr, names)
        twm = TweedieModel(best_power).fit(Xf, pp[tr_pos], otr)
        mu_va_tw = twm.predict(Xva[:, [names.index(n) for n in nf_]], ova)
        n_fits += 1

        actual_loss = pp[va_pos]
        exp_va = freq["Exposure"].to_numpy()[va_pos]
        ids_va = freq["IDpol"].to_numpy()[va_pos]

        # metrics per model
        va_frame = pd.DataFrame({"IDpol": ids_va, "Exposure": exp_va, "actual_loss": actual_loss})
        tr_frame = pd.DataFrame(
            {
                "IDpol": freq["IDpol"].to_numpy()[tr_pos],
                "Exposure": freq["Exposure"].to_numpy()[tr_pos],
                "actual_loss": pp[tr_pos],
            }
        )
        # train-side severity + tweedie predictions for the overfitting gap
        sev_hat_tr = sev_model.predict(X_pol_sev[tr_pos][:, keep_idx])
        mu_tr_tw = twm.predict(X[tr_pos][:, [names.index(n) for n in nf_]], off[tr_pos])
        for key, fm in freq_models.items():
            freq_hat_va = fm.predict(Xva, ova)
            freq_hat_tr = fm.predict(X[tr_pos], off[tr_pos])
            prem = freq_hat_va * sev_hat_va
            prem_tr = freq_hat_tr * sev_hat_tr
            f = va_frame.copy()
            f["pure_premium"] = prem
            row = {
                "repeat": rep_i // folds,
                "fold": rep_i % folds,
                "model": f"{key}+gamma",
                "gini": gini(actual_loss, prem),
                "train_gini": gini(pp[tr_pos], prem_tr),
                "loss_ratio": float(actual_loss.sum() / prem.sum()),
                "train_loss_ratio": float(pp[tr_pos].sum() / prem_tr.sum()),
                "mse": float(np.mean((actual_loss - prem) ** 2)),
                "freq_poisson_deviance": float(_deviance(yva, freq_hat_va, "poisson")),
                "freq_nb2_deviance": float(_deviance(yva, freq_hat_va, "nb2", nbm.alpha if nbm.alpha is not None else 1.0)),
                "freq_nb2_loglike": nb2_loglike(yva, freq_hat_va, nbm.alpha),
                "alpha_nb2": nbm.alpha if key == "negbin" else np.nan,
                "phi_quasi": pm.phi,
                "mean_deviance": float(_deviance(yva, freq_hat_va, "poisson")) / len(va_idx),
            }
            rows.append(row)

        f = va_frame.copy()
        f["pure_premium"] = mu_va_tw
        cal = calibration_deciles(f, "pure_premium")
        rows.append(
            {
                "repeat": rep_i // folds,
                "fold": rep_i % folds,
                "model": f"tweedie{best_power:.1f}",
                "gini": gini(actual_loss, mu_va_tw),
                "train_gini": gini(pp[tr_pos], mu_tr_tw),
                "loss_ratio": float(actual_loss.sum() / mu_va_tw.sum()),
                "train_loss_ratio": float(pp[tr_pos].sum() / mu_tr_tw.sum()),
                "mse": float(np.mean((actual_loss - mu_va_tw) ** 2)),
                "freq_poisson_deviance": np.nan,
                "freq_nb2_deviance": np.nan,
                "freq_nb2_loglike": np.nan,
                "alpha_nb2": np.nan,
                "phi_quasi": np.nan,
                "mean_deviance": tweedie_deviance(actual_loss, mu_va_tw, best_power) / len(va_idx),
            }
        )
        if n_fits % 5 == 0:
            log(f"  fold {n_fits}/{folds*repeats} done ({time.time()-t0:.0f}s)")

    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / f"task3_cv_folds{tag}.csv", index=False)
    summary = (
        cv.groupby("model")
        .agg(
            mean_gini=("gini", "mean"),
            sd_gini=("gini", "std"),
            mean_train_gini=("train_gini", "mean"),
            mean_loss_ratio=("loss_ratio", "mean"),
            sd_loss_ratio=("loss_ratio", "std"),
            mean_train_loss_ratio=("train_loss_ratio", "mean"),
            mean_mse=("mse", "mean"),
            mean_deviance=("mean_deviance", "mean"),
            sd_deviance=("mean_deviance", "std"),
            mean_freq_poisson_deviance=("freq_poisson_deviance", "mean"),
            mean_freq_nb2_deviance=("freq_nb2_deviance", "mean"),
            mean_freq_nb2_loglike=("freq_nb2_loglike", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / f"task3_summary{tag}.csv", index=False)
    log(summary.to_string(index=False))
    return cv, summary


# ------------------------------------------------------------------ holdout (task 4 support)


def holdout_eval(freq, sev, tr_mask, best_power, n_boot):
    log("=== TASK 4: holdout evaluation ===")
    y, X, names = build_design(glm_formula("ClaimNb"), freq)
    X, names = drop_empty_levels(X, names)
    off = np.log(freq["Exposure"].clip(lower=1e-6).to_numpy())
    Xtr, ytr, otr = X[tr_mask], y[tr_mask], off[tr_mask]
    Xva, yva, ova = X[~tr_mask], y[~tr_mask], off[~tr_mask]

    pm = FrequencyModel("poisson").fit(Xtr, ytr, otr)
    nbm = FrequencyModel("nb2").fit(Xtr, ytr, otr)

    claims = build_claim_frame(freq, sev)
    _, X_pol_sev, sev_names = build_design(glm_formula("ClaimAmount"), freq.assign(ClaimAmount=1.0))
    X_pol_sev = np.asarray(X_pol_sev)
    pos = {pid: i for i, pid in enumerate(freq["IDpol"].to_numpy())}
    claim_pos = np.array([pos[pid] for pid in claims["IDpol"].to_numpy()])
    X_claim_sev = X_pol_sev[claim_pos]
    cap = float(np.quantile(claims["ClaimAmount"].to_numpy()[tr_mask[claim_pos]], 0.995))
    y_sev = np.minimum(claims["ClaimAmount"].to_numpy(), cap)
    X_sev_fit, sev_fit_names = drop_empty_levels(X_claim_sev[tr_mask[claim_pos]], sev_names)
    sev_model = GammaSeverityModel().fit(X_sev_fit, y_sev[tr_mask[claim_pos]])
    keep_idx = [i for i, n in enumerate(sev_names) if n in set(sev_fit_names)]
    sev_hat_va = sev_model.predict(X_pol_sev[~tr_mask][:, keep_idx])

    pp = freq["PurePremium"].to_numpy()
    twm = TweedieModel(best_power).fit(Xtr, pp[tr_mask], otr)

    va = freq[~tr_mask].copy()
    va["prem_poisson_gamma"] = pm.predict(Xva, ova) * sev_hat_va
    va["prem_negbin_gamma"] = nbm.predict(Xva, ova) * sev_hat_va
    va["prem_tweedie"] = twm.predict(Xva, ova)

    actual = va["actual_loss"].to_numpy()

    # Calibrated variants: multiplicative intercept recalibration on TRAIN
    # (balanced-book principle: total train actual loss / total train premium)
    # so holdout loss ratios measure discrimination + generalization, not the
    # family's level bias. Gini is scale-invariant and unaffected.
    train_actual_total = float(pp[tr_mask].sum())
    prem_pois_tr = pm.predict(Xtr, otr) * sev_model.predict(X_pol_sev[tr_mask][:, keep_idx])
    prem_nb_tr = nbm.predict(Xtr, otr) * sev_model.predict(X_pol_sev[tr_mask][:, keep_idx])
    prem_tw_tr = twm.predict(Xtr, otr)
    calib = {
        "poisson+gamma": train_actual_total / float(prem_pois_tr.sum()),
        "negbin+gamma": train_actual_total / float(prem_nb_tr.sum()),
        f"tweedie{best_power:.1f}": train_actual_total / float(prem_tw_tr.sum()),
    }
    va["prem_poisson_gamma_cal"] = va["prem_poisson_gamma"] * calib["poisson+gamma"]
    va["prem_negbin_gamma_cal"] = va["prem_negbin_gamma"] * calib["negbin+gamma"]
    va["prem_tweedie_cal"] = va["prem_tweedie"] * calib[f"tweedie{best_power:.1f}"]
    log(f"train-balanced calibration factors: {json.dumps({k: round(v, 3) for k, v in calib.items()})}")

    results = {}
    for col, label in [
        ("prem_poisson_gamma", "poisson+gamma"),
        ("prem_negbin_gamma", "negbin+gamma"),
        ("prem_tweedie", f"tweedie{best_power:.1f}"),
        ("prem_poisson_gamma_cal", "poisson+gamma_cal"),
        ("prem_negbin_gamma_cal", "negbin+gamma_cal"),
        ("prem_tweedie_cal", f"tweedie{best_power:.1f}_cal"),
        ("Exposure", "exposure_only"),
    ]:
        score = va[col].to_numpy()
        results[label] = {
            "gini": gini(actual, score),
            "loss_ratio": float(actual.sum() / score.sum()),
            "mse": float(np.mean((actual - score) ** 2)),
        }
    log(json.dumps(results, indent=2))

    # paired bootstrap: gini differences
    prem_a = va["prem_poisson_gamma"].to_numpy()
    prem_b = va["prem_negbin_gamma"].to_numpy()
    prem_t = va["prem_tweedie"].to_numpy()
    rng = np.random.default_rng(SEED)
    n = len(va)
    gini_pairs = {"nb_minus_pois": [], "tweedie_minus_pois": [], "tweedie_minus_nb": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gini_pairs["nb_minus_pois"].append(gini(actual[idx], prem_b[idx]) - gini(actual[idx], prem_a[idx]))
        gini_pairs["tweedie_minus_pois"].append(gini(actual[idx], prem_t[idx]) - gini(actual[idx], prem_a[idx]))
        gini_pairs["tweedie_minus_nb"].append(gini(actual[idx], prem_t[idx]) - gini(actual[idx], prem_b[idx]))
    boot_summary = {}
    for k, v in gini_pairs.items():
        v = np.array(v)
        boot_summary[k] = {
            "mean": float(v.mean()),
            "sd": float(v.std()),
            "ci_lo": float(np.quantile(v, 0.025)),
            "ci_hi": float(np.quantile(v, 0.975)),
            "p_two_sided": p_two_sided(v.mean() / v.std()) if v.std() > 0 else 0.0,
        }
    log(json.dumps(boot_summary, indent=2))

    # decile calibration per model
    cal_tables = {}
    for col, label in [
        ("prem_poisson_gamma", "poisson+gamma"),
        ("prem_negbin_gamma", "negbin+gamma"),
        ("prem_tweedie", "tweedie"),
    ]:
        f = va[["IDpol", "Exposure", "actual_loss"]].copy()
        f["pure_premium"] = va[col].to_numpy()
        cal_tables[label] = calibration_deciles(f, "pure_premium")
        cal_tables[label].to_csv(OUT / f"holdout_calibration_{label.replace('+','_')}.csv", index=False)

    # tail analysis: top slices by each score
    tail_rows = []
    for col, label in [
        ("prem_poisson_gamma", "poisson+gamma"),
        ("prem_negbin_gamma", "negbin+gamma"),
        ("prem_tweedie", "tweedie"),
    ]:
        score = va[col].to_numpy()
        order = np.argsort(-score)
        for frac in (0.10, 0.01, 0.001):
            k = max(1, int(len(order) * frac))
            top = order[:k]
            tail_rows.append(
                {
                    "model": label,
                    "slice": f"top {frac:.1%}",
                    "n_policies": k,
                    "exposure_share": float(va["Exposure"].to_numpy()[top].sum() / va["Exposure"].sum()),
                    "actual_loss_share": float(actual[top].sum() / actual.sum()),
                    "actual_over_pred": float(actual[top].sum() / score[top].sum()),
                    "lift": float((actual[top].sum() / va["Exposure"].to_numpy()[top].sum()) / (actual.sum() / va["Exposure"].sum())),
                }
            )
    # bottom tail
    for col, label in [
        ("prem_poisson_gamma", "poisson+gamma"),
        ("prem_tweedie", "tweedie"),
    ]:
        score = va[col].to_numpy()
        order = np.argsort(score)
        k = max(1, int(len(order) * 0.10))
        bot = order[:k]
        tail_rows.append(
            {
                "model": label,
                "slice": "bottom 10%",
                "n_policies": k,
                "exposure_share": float(va["Exposure"].to_numpy()[bot].sum() / va["Exposure"].sum()),
                "actual_loss_share": float(actual[bot].sum() / actual.sum()),
                "actual_over_pred": float(actual[bot].sum() / score[bot].sum()),
                "lift": float((actual[bot].sum() / va["Exposure"].to_numpy()[bot].sum()) / (actual.sum() / va["Exposure"].sum())),
            }
        )
    tails = pd.DataFrame(tail_rows)
    tails.to_csv(OUT / "holdout_tails.csv", index=False)
    log(tails.to_string(index=False))

    # tweedie vs freq*sev premium by decile of freq*sev premium (do they differ in the tail?)
    f = va[["IDpol", "Exposure", "actual_loss"]].copy()
    f["prem_fs"] = prem_a
    f["prem_tw"] = prem_t
    f["decile_fs"] = pd.qcut(f["prem_fs"].rank(method="first", ascending=False), 10, labels=False) + 1
    cmp_dec = f.groupby("decile_fs", as_index=False).agg(
        prem_fs=("prem_fs", "sum"), prem_tw=("prem_tw", "sum"), actual_loss=("actual_loss", "sum")
    )
    cmp_dec["ratio_tw_over_fs"] = cmp_dec["prem_tw"] / cmp_dec["prem_fs"]
    cmp_dec["actual_over_fs"] = cmp_dec["actual_loss"] / cmp_dec["prem_fs"]
    cmp_dec.to_csv(OUT / "holdout_tweedie_vs_fs_deciles.csv", index=False)

    # relativities on the holdout-fit models (poisson vs nb2 vs tweedie)
    rel_rows = []
    for i, nm in enumerate(names):
        rel_rows.append(
            {
                "term": nm,
                "rel_poisson": float(np.exp(pm.beta[i])),
                "rel_nb2": float(np.exp(nbm.beta[i])),
                "rel_tweedie": float(np.exp(twm.beta[i])),
            }
        )
    rel_hold = pd.DataFrame(rel_rows)
    rel_hold.to_csv(OUT / "holdout_relativities.csv", index=False)

    payload = {
        "results": results,
        "gini_bootstrap": boot_summary,
        "tails": tails.to_dict(orient="records"),
        "tw_vs_fs_deciles": cmp_dec.to_dict(orient="records"),
        "calibration": {k: v.to_dict(orient="records") for k, v in cal_tables.items()},
        "best_power": best_power,
        "severity_cap": cap,
        "nb2_alpha": nbm.alpha,
        "phi_quasi": pm.phi,
    }
    (OUT / "holdout_summary.json").write_text(json.dumps(payload, indent=2))
    return payload, va


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 folds, 1 repeat, 50 bootstrap")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tag = "_quick" if args.quick else ""
    folds = 2 if args.quick else args.folds
    repeats = 1 if args.quick else args.repeats
    n_boot = 50 if args.quick else args.n_boot

    freq, sev = load_data()
    tr_mask, te_mask = split_train_test(freq)
    log(f"policies={len(freq):,} train={tr_mask.sum():,} test={(~tr_mask).sum():,}")

    t1, pois, nb, X, y, off, names = task1(freq, tr_mask, n_boot)
    summary, best_power, fold_df = task2(freq, tr_mask, folds, tag)
    cv, cv_summary = task3(freq, sev, tr_mask, best_power, folds, repeats, tag)
    payload, va = holdout_eval(freq, sev, tr_mask, best_power, n_boot)
    log("done. outputs in outputs_extended/")


if __name__ == "__main__":
    main()
