"""Portfolio optimization and prediction helpers."""

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

    Returns the total return or np.nan if data is insufficient.
    """
    spx = close_prices[
        (close_prices["symbol"] == "^GSPC")
        & (close_prices["date"] >= buy_date)
        & (close_prices["date"] <= sell_date)
    ].sort_values("date")
    if len(spx) < 5:
        return np.nan
    return spx["close"].iloc[-1] / spx["close"].iloc[0] - 1
