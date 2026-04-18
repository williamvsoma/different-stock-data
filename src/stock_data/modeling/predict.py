"""Portfolio optimization and prediction helpers."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf


def winsorize(a, pct=0.05):
    """Clip extreme values at pct and 1-pct quantiles."""
    return np.clip(a, *np.nanquantile(a, [pct, 1 - pct]))


def shrink_to_mean(a, alpha=0.5):
    """Shrink predictions toward cross-sectional mean."""
    return alpha * a + (1 - alpha) * np.mean(a)


def ledoit_wolf_cov(symbols, buy_date, close_prices, lookback=252):
    """Compute Ledoit-Wolf quarterly covariance from daily returns.

    Returns ``(cov_matrix, symbol_list)`` or ``(None, symbol_list)``
    on failure.
    """
    import pandas as pd

    t0 = buy_date - pd.Timedelta(days=int(lookback * 1.5))
    mask = (
        close_prices["symbol"].isin(symbols)
        & (close_prices["date"] >= t0)
        & (close_prices["date"] < buy_date)
    )
    px = close_prices[mask].pivot(index="date", columns="symbol", values="close")
    dr = px.pct_change().dropna(how="all")

    min_obs = max(60, lookback // 2)
    ok_cols = dr.columns[dr.notna().sum() >= min_obs]
    syms = [s for s in symbols if s in ok_cols]
    if len(syms) < 30:
        return None, syms

    mat = dr[syms].dropna()
    if len(mat) < min_obs:
        return None, syms

    try:
        lw = LedoitWolf().fit(mat.values)
        return lw.covariance_ * 63, syms  # quarterly scale
    except Exception:
        return None, syms


def mv_optimize(mu, cov, max_w, lam):
    """Mean-variance optimization with full covariance matrix."""
    n = len(mu)

    def obj(w):
        return -(w @ mu - lam * (w @ cov @ w))

    def jac(w):
        return -(mu - 2 * lam * (cov @ w))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bds = [(0, max_w)] * n
    w0 = np.ones(n) / n
    r = minimize(obj, w0, jac=jac, method="SLSQP", bounds=bds,
                 constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.maximum(r.x if r.success else w0, 0)
    return w / w.sum()


def mv_optimize_diag(mu, vol, max_w, lam):
    """Mean-variance optimization with diagonal covariance (fallback)."""
    n = len(mu)
    v = vol ** 2

    def obj(w):
        return -(w @ mu - lam * np.dot(w ** 2, v))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bds = [(0, max_w)] * n
    w0 = np.ones(n) / n
    r = minimize(obj, w0, method="SLSQP", bounds=bds,
                 constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.maximum(r.x if r.success else w0, 0)
    return w / w.sum()


def portfolio_turnover(prev_w, prev_syms, cur_w, cur_syms):
    """One-way portfolio turnover."""
    old = dict(zip(prev_syms, prev_w))
    new = dict(zip(cur_syms, cur_w))
    return sum(abs(new.get(s, 0) - old.get(s, 0))
               for s in set(prev_syms) | set(cur_syms)) / 2


def safe_spearmanr(a, b):
    if len(a) < 5:
        return np.nan
    r, _ = spearmanr(a, b)
    return r if np.isfinite(r) else np.nan


def bootstrap_ci(vals, n_boot=10_000, seed=42):
    """Bootstrap confidence interval for the mean.

    Returns ``(lo, hi, p_neg, boot_means)``.
    """
    rng = np.random.RandomState(seed)
    n = len(vals)
    means = np.array([
        np.mean(rng.choice(vals, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_neg = np.mean(means <= 0)
    return lo, hi, p_neg, means


def power_analysis_quarters(excess_mean, excess_std, alpha=0.05, power=0.80):
    """Compute quarters needed for 80% power to detect observed excess return.

    Uses the formula: N = (z_alpha + z_power)^2 * sigma^2 / mu^2
    where z_alpha and z_power are standard normal quantiles.
    Returns the required number of quarterly observations.
    """
    from scipy.stats import norm

    if excess_std <= 0 or excess_mean == 0:
        return float("inf")

    z_a = norm.ppf(1 - alpha / 2)  # two-sided
    z_b = norm.ppf(power)
    n = ((z_a + z_b) ** 2 * excess_std ** 2) / excess_mean ** 2
    return int(np.ceil(n))


def block_bootstrap_ci(vals, block_size=4, n_boot=10_000, seed=42):
    """Block bootstrap CI preserving autocorrelation in quarterly returns.

    Uses non-overlapping blocks of `block_size` quarters.
    Returns ``(lo, hi, p_neg, boot_means)``.
    """
    rng = np.random.RandomState(seed)
    n = len(vals)
    if n < block_size:
        # Fall back to i.i.d. bootstrap
        return bootstrap_ci(vals, n_boot, seed)

    n_blocks = n // block_size
    blocks = [vals[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]

    means = np.array([
        np.mean(np.concatenate([blocks[i] for i in rng.randint(0, len(blocks), size=n_blocks)]))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_neg = np.mean(means <= 0)
    return lo, hi, p_neg, means
