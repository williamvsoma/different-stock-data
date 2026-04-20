"""Portfolio optimization and prediction helpers."""

import warnings

import numpy as np
import pandas as pd
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
        daily_cov = lw.covariance_
        # Correlation decomposition: Σ_q = D_q C D_q where D_q = D_d * sqrt(63)
        # Mathematically equivalent to * 63 with LW's own vols; enables
        # future drop-in replacement of D_q with EWMA/GARCH/predicted vols.
        d = np.sqrt(np.diag(daily_cov))          # daily std
        d[d == 0] = 1e-10                          # guard division
        corr = daily_cov / np.outer(d, d)          # correlation matrix
        np.fill_diagonal(corr, 1.0)                # numerical cleanup
        dq = d * np.sqrt(63)                        # quarterly std
        return np.outer(dq, dq) * corr, syms       # quarterly covariance
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
    if not r.success:
        warnings.warn(
            f"MV optimizer failed (N={n}): {r.message} "
            f"mu_range=[{mu.min():.4f}, {mu.max():.4f}]"
        )
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
    if not r.success:
        warnings.warn(
            f"MV diag optimizer failed (N={n}): {r.message} "
            f"mu_range=[{mu.min():.4f}, {mu.max():.4f}]"
        )
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



def multi_source_fi(xgb_m, ridge_m, rf_m, feature_cols, ridge_cols=None):
    """Aggregate feature importance from all ensemble models.

    Returns a DataFrame with columns: xgb_gain, ridge_coef, rf_impurity, combined.
    ``ridge_cols`` are the feature names used by Ridge (may differ from tree cols).
    """
    import pandas as pd

    fi = pd.DataFrame(index=feature_cols)

    # XGBoost gain-based importance
    if hasattr(xgb_m, "feature_importances_"):
        xgb_fi = xgb_m.feature_importances_
        xgb_idx = getattr(xgb_m, "feature_names_in_", feature_cols[:len(xgb_fi)])
        fi["xgb_gain"] = pd.Series(xgb_fi, index=xgb_idx).reindex(feature_cols, fill_value=0.0)
    else:
        fi["xgb_gain"] = 0.0

    # Ridge absolute coefficients (normalised)
    if hasattr(ridge_m, "coef_"):
        r_cols = ridge_cols if ridge_cols is not None else feature_cols[:len(ridge_m.coef_)]
        abs_coef = np.abs(ridge_m.coef_)
        total = abs_coef.sum()
        norm_coef = abs_coef / total if total > 0 else abs_coef
        fi["ridge_coef"] = pd.Series(norm_coef, index=r_cols).reindex(feature_cols, fill_value=0.0)
    else:
        fi["ridge_coef"] = 0.0

    # RF impurity-based importance
    if hasattr(rf_m, "feature_importances_"):
        rf_fi = rf_m.feature_importances_
        rf_idx = getattr(rf_m, "feature_names_in_", feature_cols[:len(rf_fi)])
        fi["rf_impurity"] = pd.Series(rf_fi, index=rf_idx).reindex(feature_cols, fill_value=0.0)
    else:
        fi["rf_impurity"] = 0.0

    # Combined: equal-weight average of normalised importances
    for col in ["xgb_gain", "ridge_coef", "rf_impurity"]:
        s = fi[col].sum()
        if s > 0:
            fi[col] = fi[col] / s
    fi["combined"] = fi[["xgb_gain", "ridge_coef", "rf_impurity"]].mean(axis=1)

    return fi


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


def compute_spx_return(buy_date, sell_date, close_prices):
    """Compute S&P 500 (^GSPC) return between buy_date and sell_date.

    Uses the same price-selection convention as strategy returns:
    first available close on or after each target date.
    Returns the total return or np.nan if data is insufficient.
    """
    spx = close_prices[
        (close_prices["symbol"] == "^GSPC")
    ].sort_values("date")
    buy_w = spx[spx["date"] >= buy_date]["close"]
    sell_w = spx[spx["date"] >= sell_date]["close"]
    if len(buy_w) == 0 or len(sell_w) == 0:
        return np.nan
    return sell_w.iloc[0] / buy_w.iloc[0] - 1


def mv_optimize_turnover(mu, cov, max_w, lam, prev_w, max_turnover):
    """Mean-variance optimization with turnover constraint.

    Adds constraint: sum(|w_new - w_old|) / 2 <= max_turnover.
    Uses linear reformulation with auxiliary vars to keep SLSQP-compatible.
    """
    n = len(mu)
    if prev_w is None:
        prev_w = np.ones(n) / n

    # Augmented variable: x = [w, t+, t-] where w_new - w_old = t+ - t-
    # turnover = sum(t+ + t-) / 2
    n_aug = 3 * n

    def obj(x):
        w = x[:n]
        return -(w @ mu - lam * (w @ cov @ w))

    def jac_obj(x):
        w = x[:n]
        g = np.zeros(n_aug)
        g[:n] = -(mu - 2 * lam * (cov @ w))
        return g

    cons = [
        # sum(w) = 1
        {"type": "eq", "fun": lambda x: x[:n].sum() - 1.0},
        # w - prev_w = t+ - t-
        {"type": "eq", "fun": lambda x: x[:n] - prev_w - x[n:2*n] + x[2*n:]},
        # turnover: sum(t+ + t-) / 2 <= max_turnover
        {"type": "ineq", "fun": lambda x: max_turnover - (x[n:2*n].sum() + x[2*n:].sum()) / 2},
    ]
    bds = [(0, max_w)] * n + [(0, 1.0)] * (2 * n)
    x0 = np.concatenate([np.ones(n) / n, np.zeros(2 * n)])

    r = minimize(obj, x0, jac=jac_obj, method="SLSQP", bounds=bds,
                 constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    if not r.success:
        print(f"    ⚠ Turnover-constrained optimizer failed ({r.message}), using equal-weight fallback")
        return np.ones(n) / n
    w = np.maximum(r.x[:n], 0)
    w = np.minimum(w, max_w)
    return w / w.sum()


def select_vol_estimate(p_vol_ml, p_vol_naive, vol_rc_train, vol_rc_gate, vol_floor):
    """Select between ML vol predictions and naive hist_vol based on quality gate.

    Falls back to hist_vol_3m when ML model rank correlation is below gate threshold.
    """
    if (p_vol_naive is not None
            and np.isfinite(vol_rc_train)
            and vol_rc_train < vol_rc_gate):
        return np.maximum(np.nan_to_num(p_vol_naive, nan=p_vol_ml.mean()), vol_floor)
    return p_vol_ml


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
