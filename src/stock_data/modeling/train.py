"""Walk-forward portfolio training engine."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stock_data.config import (
    ENS_W,
    EARNINGS_LAG_DAYS,
    PROD_CFG,
    RF_PARAMS,
    RIDGE_PARAMS,
    XGB_PARAMS,
)
from stock_data.modeling.predict import (
    ledoit_wolf_cov,
    mv_optimize,
    mv_optimize_diag,
    portfolio_turnover,
    safe_spearmanr,
    shrink_to_mean,
    winsorize,
)


def _select_features(Xtr, ytr, threshold=0.3):
    """Select features when dimensionality ratio is too high.

    When ``n_features / n_samples <= threshold`` all columns are returned.
    Otherwise keeps the top-N features ranked by univariate F-statistic
    against the target (``f_regression``), where
    ``N = floor(n_samples * threshold)``.

    Uses F-statistic rather than variance so that rank features with bounded
    spread are not penalised relative to raw magnitude features.
    """
    n_samples, n_features = Xtr.shape
    if n_features <= 1 or n_features / n_samples <= threshold:
        return Xtr.columns.tolist()

    n_keep = max(1, int(n_samples * threshold))

    # Impute NaNs with column medians before computing F-statistics.
    # keep_empty_features=True ensures all-NaN columns stay (imputed to 0)
    # so F-scores can be computed for all original columns.
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imp = imp.fit_transform(Xtr)

    f_scores, _ = f_regression(X_imp, ytr)
    f_scores = pd.Series(f_scores, index=Xtr.columns).fillna(0)
    return f_scores.nlargest(n_keep).index.tolist()


def walk_forward(risk_model_df, feature_cols_all, close_prices):
    """Run the production walk-forward engine.

    Returns ``(prod_df, prod_fi, prod_weights_history)``.
    """
    X_p = risk_model_df[feature_cols_all].copy()
    y_ret = risk_model_df["next_q_return"].copy()
    y_vol = risk_model_df["realized_vol"].copy()

    ok = y_ret.notna() & y_vol.notna()
    X_p, y_ret, y_vol = X_p[ok], y_ret[ok], y_vol[ok]

    dp = X_p.index.get_level_values("date")
    unique_dates = sorted(dp.unique())

    prev_w, prev_s = None, None
    results = []
    fi_list = []
    weights_history = {}

    cfg = PROD_CFG

    print("=" * 80)
    print("WALK-FORWARD ENGINE")
    print("=" * 80)

    for td in unique_dates:
        tr_dates = [d for d in unique_dates if d < td]

        max_q = cfg.get("max_train_q")
        if max_q and len(tr_dates) > max_q:
            tr_dates = tr_dates[-max_q:]

        if len(tr_dates) < cfg["min_train_q"]:
            continue

        tr_mask = dp.isin(tr_dates)
        te_mask = dp == td
        Xtr, Xte = X_p[tr_mask], X_p[te_mask]
        ytr_r, ytr_v = y_ret[tr_mask], y_vol[tr_mask]

        if len(Xtr) < cfg["min_train_rows"] or len(Xte) < cfg["min_test_stocks"]:
            continue

        # ── Feature selection for high-dimensionality regimes ──
        sel_cols = _select_features(Xtr, ytr_r, cfg["feat_ratio_threshold"])
        Xtr_sel = Xtr[sel_cols]
        Xte_sel = Xte[sel_cols]

        # ── Ensemble return predictions ──
        xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_m.fit(Xtr_sel, ytr_r, verbose=0)
        p_xgb = xgb_m.predict(Xte_sel)

        imp = SimpleImputer(strategy="median")
        Xtr_imp = imp.fit_transform(Xtr_sel)
        Xte_imp = imp.transform(Xte_sel)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_imp)
        Xte_s = sc.transform(Xte_imp)
        ridge_m = Ridge(alpha=RIDGE_PARAMS["alpha"])
        ridge_m.fit(Xtr_s, ytr_r)
        p_rdg = ridge_m.predict(Xte_s)

        rf_m = RandomForestRegressor(**RF_PARAMS)
        rf_m.fit(Xtr_imp, ytr_r)
        p_rf = rf_m.predict(Xte_imp)

        p_ens = ENS_W["xgb"] * p_xgb + ENS_W["ridge"] * p_rdg + ENS_W["rf"] * p_rf
        p_ret = shrink_to_mean(
            winsorize(p_ens, cfg["winsor_pct"]),
            cfg["shrinkage_alpha"],
        )

        fi_full = pd.Series(0.0, index=feature_cols_all)
        fi_full[sel_cols] = xgb_m.feature_importances_
        fi_list.append({
            "date": td,
            "fi": fi_full,
        })

        # ── Volatility model ──
        vol_m = xgb.XGBRegressor(**XGB_PARAMS)
        vol_m.fit(Xtr_sel, ytr_v, verbose=0)
        p_vol = np.maximum(vol_m.predict(Xte_sel), 0.05)

        # ── Covariance ──
        test_syms = Xte.index.get_level_values("symbol").tolist()
        buy_dt = td + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        cov_mat, cov_syms = ledoit_wolf_cov(
            test_syms, buy_dt, close_prices, cfg["cov_lookback_days"],
        )

        act_ret = risk_model_df.loc[Xte.index, "next_q_return"].values
        act_vol = risk_model_df.loc[Xte.index, "realized_vol"].values

        used_lw = False
        if cov_mat is not None:
            both = [s for s in test_syms if s in cov_syms]
            if len(both) >= cfg["min_test_stocks"]:
                ti = {s: i for i, s in enumerate(test_syms)}
                ci = {s: i for i, s in enumerate(cov_syms)}
                kt = [ti[s] for s in both]
                kc = [ci[s] for s in both]
                w = mv_optimize(
                    p_ret[kt], cov_mat[np.ix_(kc, kc)],
                    cfg["max_weight"], cfg["risk_aversion"],
                )
                port_ret = w @ act_ret[kt]
                opt_syms = both
                used_lw = True

        if not used_lw:
            w = mv_optimize_diag(
                p_ret, p_vol, cfg["max_weight"], cfg["risk_aversion"],
            )
            port_ret = w @ act_ret
            opt_syms = test_syms

        mkt_ret = act_ret.mean()

        # ── Turnover & costs ──
        to = (portfolio_turnover(prev_w, prev_s, w, opt_syms)
              if prev_w is not None else 1.0)
        txc = to * cfg["cost_bps"] / 10000
        net_ret = port_ret - txc

        # ── Model quality ──
        vrc = safe_spearmanr(p_vol, act_vol)
        rrc = safe_spearmanr(p_ret, act_ret)
        rrc_x = safe_spearmanr(p_xgb, act_ret)
        rrc_r = safe_spearmanr(p_rdg, act_ret)
        rrc_f = safe_spearmanr(p_rf, act_ret)
        n_held = int((w > 0.001).sum())

        results.append({
            "test_date": td, "n_train_q": len(tr_dates),
            "n_stocks": len(Xte), "n_eligible": len(opt_syms),
            "n_held": n_held, "max_wt": w.max(),
            "mkt_ret": mkt_ret, "gross_ret": port_ret,
            "net_ret": net_ret, "turnover": to, "tx_cost": txc,
            "vol_rc": vrc, "ret_rc": rrc,
            "ret_rc_xgb": rrc_x, "ret_rc_ridge": rrc_r, "ret_rc_rf": rrc_f,
            "used_lw": used_lw,
        })
        weights_history[td] = {"weights": w, "symbols": opt_syms}
        prev_w, prev_s = w, opt_syms

        ex = net_ret - mkt_ret
        print(f"  {td.date()} | {len(tr_dates)}Q | {n_held:3d}h | "
              f"G={port_ret:+.2%} N={net_ret:+.2%} M={mkt_ret:+.2%} Ex={ex:+.2%} | "
              f"TO={to:.0%} | {'LW' if used_lw else 'DG'} | rc={rrc:.2f}")

    prod_df = pd.DataFrame(results)
    return prod_df, fi_list, weights_history


def factor_benchmarks(risk_model_df, feature_cols_all, prod_df):
    """Compare production MV against simple single-factor strategies."""
    factor_defs = {
        "Low Vol":       {"col": "hist_vol_6m",   "ascending": True},
        "Momentum (3m)": {"col": "momentum_3m",   "ascending": False},
        "Quality (ROE)": {"col": "roe",           "ascending": False},
        "Value (FCF/A)": {"col": "fcf_to_assets", "ascending": False},
    }

    X_p = risk_model_df[feature_cols_all].copy()
    dp = X_p.index.get_level_values("date")
    unique_dates = sorted(dp.unique())
    cfg = PROD_CFG

    factor_results = {name: [] for name in factor_defs}

    for td in unique_dates:
        tr_dates = [d for d in unique_dates if d < td]
        if len(tr_dates) < cfg["min_train_q"]:
            continue
        te_mask = dp == td
        if te_mask.sum() < cfg["min_test_stocks"]:
            continue

        te_idx = X_p[te_mask].index
        act_rets = risk_model_df.loc[te_idx, "next_q_return"].values
        mkt = act_rets.mean()

        for fname, fdef in factor_defs.items():
            col = fdef["col"]
            if col not in risk_model_df.columns:
                continue
            vals = risk_model_df.loc[te_idx, col].values
            valid = ~np.isnan(vals) & ~np.isnan(act_rets)
            if valid.sum() < 20:
                continue
            n_pick = max(1, int(valid.sum() * 0.2))
            valid_idx = np.where(valid)[0]
            sorted_idx = valid_idx[np.argsort(vals[valid_idx])]
            pick_idx = (sorted_idx[:n_pick] if fdef["ascending"]
                        else sorted_idx[-n_pick:])
            port_ret = act_rets[pick_idx].mean()
            factor_results[fname].append({
                "test_date": td, "port_ret": port_ret,
                "mkt_ret": mkt, "excess": port_ret - mkt, "n_held": n_pick,
            })

    return factor_results
