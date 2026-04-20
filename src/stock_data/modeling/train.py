"""Walk-forward portfolio training engine."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stock_data.config import (
    ENS_W,
    EARNINGS_LAG_DAYS,
    PROD_CFG,
    RF_PARAMS,
    RIDGE_PARAMS,
    VOL_FLOOR,
    VOL_RC_GATE,
    XGB_PARAMS,
    XGB_VOL_PARAMS,
)
from stock_data.modeling.predict import (
    compute_spx_return,
    ledoit_wolf_cov,
    multi_source_fi,
    mv_optimize,
    mv_optimize_diag,
    portfolio_turnover,
    rank_transform_mu,
    safe_spearmanr,
    select_vol_estimate,
    shrink_to_mean,
    winsorize,
)


def _select_features(Xtr, ytr, threshold=0.3):
    """Select features when dimensionality ratio is too high.

    When ``n_features / n_samples <= threshold`` all columns are returned.
    Otherwise keeps the top-N features using a blended score of
    F-statistic (linear) and mutual information (nonlinear), capturing
    both linear and nonlinear dependencies.
    """
    n_samples, n_features = Xtr.shape
    if n_features <= 1 or n_features / n_samples <= threshold:
        return Xtr.columns.tolist()

    n_keep = max(1, int(n_samples * threshold))

    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imp = imp.fit_transform(Xtr)

    # Linear: F-statistic
    f_scores, _ = f_regression(X_imp, ytr)
    f_scores = pd.Series(f_scores, index=Xtr.columns).fillna(0)
    f_ranks = f_scores.rank(pct=True)

    # Nonlinear: mutual information (needs ≥3 samples for k-NN)
    if n_samples >= 3:
        mi_scores = mutual_info_regression(X_imp, ytr, random_state=42)
        mi_scores = pd.Series(mi_scores, index=Xtr.columns).fillna(0)
        mi_ranks = mi_scores.rank(pct=True)
        blended = 0.5 * f_ranks + 0.5 * mi_ranks
    else:
        blended = f_ranks

    return blended.nlargest(n_keep).index.tolist()


# ── Shared helpers (used by walk_forward and evaluation.py) ────────────────────


def fit_ensemble(Xtr_sel, Xte_sel, ytr_r, ens_weights=None):
    """Fit XGB/Ridge/RF ensemble and return predictions + models.

    Parameters
    ----------
    Xtr_sel, Xte_sel : DataFrames with selected features
    ytr_r : training return targets
    ens_weights : dict like {"xgb": 0.5, "ridge": 0.25, "rf": 0.25}

    Returns
    -------
    (p_ens, p_xgb, p_rdg, p_rf, models) where models = (xgb_m, ridge_m, rf_m)
    """
    if ens_weights is None:
        ens_weights = ENS_W

    xgb_m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=20)
    # Use last 20% of training data as validation for early stopping
    n_val = max(1, int(len(Xtr_sel) * 0.2))
    Xtr_fit, Xval = Xtr_sel.iloc[:-n_val], Xtr_sel.iloc[-n_val:]
    ytr_fit, yval = ytr_r.iloc[:-n_val], ytr_r.iloc[-n_val:]
    xgb_m.fit(
        Xtr_fit, ytr_fit, eval_set=[(Xval, yval)], verbose=0,
    )
    p_xgb = xgb_m.predict(Xte_sel)

    imp = SimpleImputer(strategy="median")
    Xtr_imp = imp.fit_transform(Xtr_sel)
    Xte_imp = imp.transform(Xte_sel)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_imp)
    Xte_s = sc.transform(Xte_imp)

    ridge_m = Ridge(alpha=RIDGE_PARAMS["alpha"])
    if ens_weights.get("ridge", 0) > 0:
        ridge_m.fit(Xtr_s, ytr_r)
        p_rdg = ridge_m.predict(Xte_s)
    else:
        p_rdg = np.zeros(len(Xte_sel))

    rf_m = RandomForestRegressor(**RF_PARAMS)
    rf_m.fit(Xtr_imp, ytr_r)
    p_rf = rf_m.predict(Xte_imp)

    p_ens = (ens_weights.get("xgb", 0) * p_xgb
             + ens_weights.get("ridge", 0) * p_rdg
             + ens_weights.get("rf", 0) * p_rf)

    return p_ens, p_xgb, p_rdg, p_rf, (xgb_m, ridge_m, rf_m)


def fit_vol_model(Xtr_sel, Xte_sel, ytr_v):
    """Fit XGBoost vol model with quality gate.

    Returns (p_vol, vol_rc_train, hist_vol_for_diag).
    """
    vol_m = xgb.XGBRegressor(**XGB_VOL_PARAMS)
    vol_m.fit(Xtr_sel, ytr_v, verbose=0)
    p_vol_ml = np.maximum(vol_m.predict(Xte_sel), VOL_FLOOR)

    p_vol_naive = Xte_sel["hist_vol_3m"].values if "hist_vol_3m" in Xte_sel.columns else None
    vol_rc_train = safe_spearmanr(vol_m.predict(Xtr_sel), ytr_v) if len(ytr_v) >= 10 else np.nan
    p_vol = select_vol_estimate(p_vol_ml, p_vol_naive, vol_rc_train, VOL_RC_GATE, VOL_FLOOR)

    hist_vol_for_diag = Xte_sel["hist_vol_3m"].values if "hist_vol_3m" in Xte_sel.columns else p_vol
    return p_vol, vol_rc_train, hist_vol_for_diag


def build_covariance(test_syms, buy_dt, close_prices, cfg):
    """Compute Ledoit-Wolf covariance or return None for diagonal fallback."""
    return ledoit_wolf_cov(test_syms, buy_dt, close_prices, cfg["cov_lookback_days"])


def optimize_portfolio(p_ret, p_vol, hist_vol_for_diag, cov_mat, cov_syms, test_syms, cfg):
    """MV optimize with full cov or diagonal fallback.

    When LW covariance is available, rescales the correlation matrix using
    ML vol predictions for more accurate forward-looking risk estimates.

    Returns (w, opt_syms, used_lw).
    """
    used_lw = False
    if cov_mat is not None:
        both = [s for s in test_syms if s in cov_syms]
        if len(both) >= cfg["min_test_stocks"]:
            ti = {s: i for i, s in enumerate(test_syms)}
            ci = {s: i for i, s in enumerate(cov_syms)}
            kt = [ti[s] for s in both]
            kc = [ci[s] for s in both]

            # Rescale LW cov with ML vol predictions (#73):
            # cov_rescaled = D_ml @ corr(LW) @ D_ml
            lw_sub = cov_mat[np.ix_(kc, kc)]
            lw_diag = np.sqrt(np.diag(lw_sub))
            # Convert to correlation, guarding against zero-vol assets
            safe_diag = np.where(lw_diag > 0, lw_diag, 1.0)
            corr = lw_sub / np.outer(safe_diag, safe_diag)
            np.fill_diagonal(corr, 1.0)

            ml_vol = np.maximum(p_vol[kt], VOL_FLOOR) / 2.0  # annualized → quarterly
            rescaled_cov = corr * np.outer(ml_vol, ml_vol)

            w = mv_optimize(
                p_ret[kt], rescaled_cov,
                cfg["max_weight"], cfg["risk_aversion"],
            )
            return w, both, True

    diag_vol = np.maximum(np.nan_to_num(hist_vol_for_diag, nan=0.20), VOL_FLOOR)
    w = mv_optimize_diag(p_ret, diag_vol, cfg["max_weight"], cfg["risk_aversion"])
    return w, test_syms, False


def predict_all_quarters(risk_model_df, feature_cols_all, close_prices,
                         cfg=None, ens_weights=None, checkpoint_dir=None):
    """Run walk-forward prediction only (no optimization).

    Returns a list of per-quarter prediction dicts that can be passed to
    ``optimize_from_predictions()`` with different optimizer settings.

    If ``checkpoint_dir`` is provided, saves predictions after each quarter
    and resumes from the last checkpoint on restart.
    """
    import pickle as _pkl
    from pathlib import Path

    if cfg is None:
        cfg = PROD_CFG
    if ens_weights is None:
        ens_weights = ENS_W

    # Checkpoint loading
    predictions = []
    fi_list = []
    completed_dates = set()
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = checkpoint_dir / "predictions_checkpoint.pkl"
        if ckpt_file.exists():
            with open(ckpt_file, "rb") as f:
                saved = _pkl.load(f)
            predictions = saved["predictions"]
            fi_list = saved["fi_list"]
            completed_dates = {p["test_date"] for p in predictions}
            print(f"  Resumed from checkpoint: {len(predictions)} quarters done")

    X_p = risk_model_df[feature_cols_all].copy()
    y_ret = risk_model_df["next_q_return"].copy()
    y_vol = risk_model_df["realized_vol"].copy()

    ok = y_ret.notna() & y_vol.notna()
    X_p, y_ret, y_vol = X_p[ok], y_ret[ok], y_vol[ok]

    dp = X_p.index.get_level_values("date")
    unique_dates = sorted(dp.unique())

    for td in unique_dates:
        if td in completed_dates:
            continue

        tr_dates = [d for d in unique_dates if d < td]

        embargo_q = cfg.get("embargo_q", 0)
        if embargo_q > 0 and len(tr_dates) > embargo_q:
            tr_dates = tr_dates[:-embargo_q]

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

        sel_cols = _select_features(Xtr, ytr_r, cfg["feat_ratio_threshold"])
        Xtr_sel, Xte_sel = Xtr[sel_cols], Xte[sel_cols]

        p_ens, p_xgb, p_rdg, p_rf, (xgb_m, ridge_m, rf_m) = fit_ensemble(
            Xtr_sel, Xte_sel, ytr_r, ens_weights,
        )

        fi_df = multi_source_fi(xgb_m, ridge_m, rf_m, feature_cols_all, sel_cols)
        fi_list.append({"date": td, "fi": fi_df["combined"], "fi_detail": fi_df})

        p_vol, vol_rc_train, hist_vol_for_diag = fit_vol_model(Xtr_sel, Xte_sel, ytr_v)

        test_syms = Xte.index.get_level_values("symbol").tolist()
        act_ret = risk_model_df.loc[Xte.index, "next_q_return"].values
        act_vol = risk_model_df.loc[Xte.index, "realized_vol"].values

        predictions.append({
            "test_date": td, "n_train_q": len(tr_dates),
            "test_index": Xte.index,
            "test_syms": test_syms,
            "p_ens": p_ens, "p_xgb": p_xgb, "p_rdg": p_rdg, "p_rf": p_rf,
            "p_vol": p_vol, "vol_rc_train": vol_rc_train,
            "hist_vol_for_diag": hist_vol_for_diag,
            "act_ret": act_ret, "act_vol": act_vol,
        })

        # Checkpoint save after each quarter
        if checkpoint_dir is not None:
            ckpt_file = checkpoint_dir / "predictions_checkpoint.pkl"
            with open(ckpt_file, "wb") as f:
                _pkl.dump({"predictions": predictions, "fi_list": fi_list}, f)

    return predictions, fi_list


def optimize_from_predictions(predictions, close_prices, cfg=None):
    """Run MV optimization on cached predictions.

    Accepts the output of ``predict_all_quarters()`` and returns the same
    ``(prod_df, weights_history)`` as the optimization half of ``walk_forward``.
    This allows re-running optimizer sweeps without re-training.
    """
    if cfg is None:
        cfg = PROD_CFG

    prev_w, prev_s = None, None
    results = []
    weights_history = {}

    for pred in predictions:
        td = pred["test_date"]
        p_ens = pred["p_ens"]
        p_ret = shrink_to_mean(winsorize(p_ens, cfg["winsor_pct"]), cfg["shrinkage_alpha"])
        if cfg.get("use_rank_mu", False):
            p_ret = rank_transform_mu(p_ret)
        p_vol = pred["p_vol"]
        hist_vol_for_diag = pred["hist_vol_for_diag"]
        test_syms = pred["test_syms"]
        act_ret = pred["act_ret"]
        act_vol = pred["act_vol"]

        buy_dt = td + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        cov_mat, cov_syms = build_covariance(test_syms, buy_dt, close_prices, cfg)
        w, opt_syms, used_lw = optimize_portfolio(
            p_ret, p_vol, hist_vol_for_diag, cov_mat, cov_syms, test_syms, cfg,
        )

        if used_lw:
            ti = {s: i for i, s in enumerate(test_syms)}
            kt = [ti[s] for s in opt_syms]
            port_ret = w @ act_ret[kt]
        else:
            port_ret = w @ act_ret

        mkt_ret = act_ret.mean()
        sell_dt = td + pd.DateOffset(months=3) + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        spx_ret = compute_spx_return(buy_dt, sell_dt, close_prices)

        to = (portfolio_turnover(prev_w, prev_s, w, opt_syms)
              if prev_w is not None else 1.0)
        txc = to * cfg["cost_bps"] / 10000
        net_ret = port_ret - txc

        vrc = safe_spearmanr(p_vol, act_vol)
        rrc = safe_spearmanr(p_ret, act_ret)
        rrc_x = safe_spearmanr(pred["p_xgb"], act_ret)
        rrc_r = safe_spearmanr(pred["p_rdg"], act_ret)
        rrc_f = safe_spearmanr(pred["p_rf"], act_ret)
        n_held = int((w > 0.001).sum())

        results.append({
            "test_date": td, "n_train_q": pred["n_train_q"],
            "n_stocks": len(test_syms), "n_eligible": len(opt_syms),
            "n_held": n_held, "max_wt": w.max(),
            "mkt_ret": mkt_ret, "spx_ret": spx_ret, "gross_ret": port_ret,
            "net_ret": net_ret, "turnover": to, "tx_cost": txc,
            "vol_rc": vrc, "ret_rc": rrc,
            "ret_rc_xgb": rrc_x, "ret_rc_ridge": rrc_r, "ret_rc_rf": rrc_f,
            "used_lw": used_lw,
        })
        weights_history[td] = {"weights": w, "symbols": opt_syms}
        prev_w, prev_s = w, opt_syms

    return pd.DataFrame(results), weights_history


def walk_forward(risk_model_df, feature_cols_all, close_prices, cfg=None,
                 ens_weights=None):
    """Run the production walk-forward engine.

    Returns ``(prod_df, prod_fi, prod_weights_history)``.
    """
    if cfg is None:
        cfg = PROD_CFG
    if ens_weights is None:
        ens_weights = ENS_W

    predictions, fi_list = predict_all_quarters(
        risk_model_df, feature_cols_all, close_prices, cfg, ens_weights,
    )
    prod_df, weights_history = optimize_from_predictions(
        predictions, close_prices, cfg,
    )

    # Print summary for backward compatibility
    print("=" * 80)
    print("WALK-FORWARD ENGINE")
    print("=" * 80)
    for _, row in prod_df.iterrows():
        ex = row["net_ret"] - row["mkt_ret"]
        lw = "LW" if row["used_lw"] else "DG"
        print(f"  {row['test_date'].date()} | {row['n_train_q']}Q | "
              f"{row['n_held']:3d}h | "
              f"G={row['gross_ret']:+.2%} N={row['net_ret']:+.2%} "
              f"M={row['mkt_ret']:+.2%} Ex={ex:+.2%} | "
              f"TO={row['turnover']:.0%} | {lw} | rc={row['ret_rc']:.2f}")

    return prod_df, fi_list, weights_history


def factor_benchmarks(risk_model_df, feature_cols_all, prod_df, cfg=None):
    """Compare production MV against simple single-factor strategies.

    Factor portfolios now track turnover and apply transaction costs
    for fair comparison with the MV strategy.
    """
    if cfg is None:
        cfg = PROD_CFG

    factor_defs = {
        "Low Vol":       {"col": "hist_vol_6m",   "ascending": True},
        "Momentum (3m)": {"col": "momentum_3m",   "ascending": False},
        "Quality (ROE)": {"col": "roe",           "ascending": False},
        "Value (FCF/A)": {"col": "fcf_to_assets", "ascending": False},
    }

    X_p = risk_model_df[feature_cols_all].copy()
    dp = X_p.index.get_level_values("date")
    unique_dates = sorted(dp.unique())

    factor_results = {name: [] for name in factor_defs}
    prev_held = {name: set() for name in factor_defs}

    for td in unique_dates:
        tr_dates = [d for d in unique_dates if d < td]
        if len(tr_dates) < cfg["min_train_q"]:
            continue
        te_mask = dp == td
        if te_mask.sum() < cfg["min_test_stocks"]:
            continue

        te_idx = X_p[te_mask].index
        act_rets = risk_model_df.loc[te_idx, "next_q_return"].values
        syms = te_idx.get_level_values("symbol").values
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

            # Estimate turnover from held-set changes (equal-weight)
            cur_held = set(syms[pick_idx])
            if prev_held[fname]:
                n_old = len(prev_held[fname])
                n_new = len(cur_held)
                n_common = len(cur_held & prev_held[fname])
                turnover = 1.0 - n_common / max(n_old, n_new)
            else:
                turnover = 1.0
            prev_held[fname] = cur_held

            txc = turnover * cfg["cost_bps"] / 10000
            factor_results[fname].append({
                "test_date": td, "port_ret": port_ret,
                "net_ret": port_ret - txc,
                "mkt_ret": mkt, "excess": port_ret - mkt,
                "excess_net": port_ret - txc - mkt,
                "turnover": turnover, "tx_cost": txc,
                "n_held": n_pick,
            })

    return factor_results
