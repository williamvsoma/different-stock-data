"""Controlled experiments comparing baseline vs proposed changes.

Each experiment runs the walk-forward engine with ONE change vs baseline,
captures key metrics, and prints a comparison table.

Usage:
    uv run python experiments/run_experiments.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stock_data.config import (
    ENS_W, EARNINGS_LAG_DAYS, PROD_CFG, XGB_PARAMS, RIDGE_PARAMS, RF_PARAMS,
)
from stock_data.modeling.predict import (
    ledoit_wolf_cov, mv_optimize, mv_optimize_diag,
    portfolio_turnover, safe_spearmanr, shrink_to_mean, winsorize,
)
from stock_data.modeling.train import _select_features
from stock_data.modeling.train import _select_features


# ── Data loading ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED = DATA_DIR / "data" / "processed"
INTERIM = DATA_DIR / "data" / "interim"

risk_model_df = pd.read_parquet(PROCESSED / "risk_model_df.parquet")
with open(PROCESSED / "feature_cols.pkl", "rb") as f:
    feature_cols_all = pickle.load(f)
close_prices = pd.read_parquet(INTERIM / "close_prices.parquet")

rank_cols = [c for c in feature_cols_all if c.endswith("_rank")]
raw_cols = [c for c in feature_cols_all if not c.endswith("_rank")]


# ── Minimal walk-forward engine (parameterised) ───────────────────────────────

def run_walk_forward(
    risk_model_df, feature_cols, close_prices, cfg,
    *,
    ens_weights=None,
    adaptive_weights=False,
    adaptive_floor=0.10,
    adaptive_lookback=4,
    vol_params=None,
    vol_floor=0.05,
    vol_rc_gate=None,
    tree_cols=None,
    ridge_feature_cols=None,
    shrinkage_alpha=None,
    permute_seed=None,
    label="baseline",
    return_detail=False,
):
    """Run walk-forward and return per-quarter results DataFrame.

    Parameters control which experimental variant is active.
    """
    if ens_weights is None:
        ens_weights = ENS_W.copy()
    if vol_params is None:
        vol_params = XGB_PARAMS.copy()
    if tree_cols is None:
        tree_cols = feature_cols
    if ridge_feature_cols is None:
        ridge_feature_cols = feature_cols

    X_p = risk_model_df[feature_cols].copy()
    y_ret = risk_model_df["next_q_return"].copy()
    y_vol = risk_model_df["realized_vol"].copy()

    ok = y_ret.notna() & y_vol.notna()
    X_p, y_ret, y_vol = X_p[ok], y_ret[ok], y_vol[ok]

    dp = X_p.index.get_level_values("date")
    unique_dates = sorted(dp.unique())

    prev_w, prev_s = None, None
    results = []
    rc_history = []
    stock_detail = []

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

        # ── Feature selection ──
        avail_tree = [c for c in tree_cols if c in Xtr.columns]
        avail_ridge = [c for c in ridge_feature_cols if c in Xtr.columns]
        sel_tree = _select_features(Xtr[avail_tree], ytr_r, cfg["feat_ratio_threshold"])
        sel_ridge = _select_features(Xtr[avail_ridge], ytr_r, cfg["feat_ratio_threshold"])
        sel_all = sorted(set(sel_tree) | set(sel_ridge))
        Xtr_sel = Xtr[sel_all]
        Xte_sel = Xte[sel_all]

        # ── Models ──
        xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_m.fit(Xtr[sel_tree], ytr_r, verbose=0)
        p_xgb = xgb_m.predict(Xte[sel_tree])

        imp_r = SimpleImputer(strategy="median")
        Xtr_rdg = imp_r.fit_transform(Xtr[sel_ridge])
        Xte_rdg = imp_r.transform(Xte[sel_ridge])
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_rdg)
        Xte_s = sc.transform(Xte_rdg)
        ridge_m = Ridge(alpha=RIDGE_PARAMS["alpha"])
        ridge_m.fit(Xtr_s, ytr_r)
        p_rdg = ridge_m.predict(Xte_s)

        imp_t = SimpleImputer(strategy="median")
        rf_m = RandomForestRegressor(**RF_PARAMS)
        rf_m.fit(imp_t.fit_transform(Xtr[sel_tree]), ytr_r)
        p_rf = rf_m.predict(imp_t.transform(Xte[sel_tree]))

        # ── Ensemble weights ──
        if adaptive_weights and rc_history:
            recent = rc_history[-adaptive_lookback:]
            ew = _compute_adaptive_weights(recent, floor=adaptive_floor)
        else:
            ew = ens_weights

        p_ens = ew["xgb"] * p_xgb + ew["ridge"] * p_rdg + ew["rf"] * p_rf
        s_a = shrinkage_alpha if shrinkage_alpha is not None else cfg["shrinkage_alpha"]
        p_ret = shrink_to_mean(winsorize(p_ens, cfg["winsor_pct"]), s_a)

        # Permutation test: shuffle cross-sectional ranking (null = random stock→prediction mapping)
        if permute_seed is not None:
            perm_rng = np.random.RandomState(permute_seed + int(td.timestamp()) % 10000)
            p_ret = perm_rng.permutation(p_ret)

        # ── Vol model ──
        vol_m = xgb.XGBRegressor(**vol_params)
        vol_m.fit(Xtr_sel, ytr_v, verbose=0)
        p_vol_ml = np.maximum(vol_m.predict(Xte_sel), vol_floor)

        # Vol quality gate
        if vol_rc_gate is not None and "hist_vol_3m" in Xte_sel.columns:
            p_vol_naive = Xte_sel["hist_vol_3m"].values
            vrc_prelim = safe_spearmanr(p_vol_ml, ytr_v[-len(p_vol_ml):]) if len(p_vol_ml) >= 5 else np.nan
            if np.isfinite(vrc_prelim) and vrc_prelim < vol_rc_gate:
                p_vol = np.maximum(np.nan_to_num(p_vol_naive, nan=p_vol_ml.mean()), vol_floor)
            else:
                p_vol = p_vol_ml
        else:
            p_vol = p_vol_ml

        # ── Covariance + optimisation ──
        test_syms = Xte.index.get_level_values("symbol").tolist()
        buy_dt = td + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        cov_mat, cov_syms = ledoit_wolf_cov(test_syms, buy_dt, close_prices, cfg["cov_lookback_days"])

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
                w = mv_optimize(p_ret[kt], cov_mat[np.ix_(kc, kc)], cfg["max_weight"], cfg["risk_aversion"])
                port_ret = w @ act_ret[kt]
                opt_syms = both
                used_lw = True

        if not used_lw:
            w = mv_optimize_diag(p_ret, p_vol, cfg["max_weight"], cfg["risk_aversion"])
            port_ret = w @ act_ret
            opt_syms = test_syms

        mkt_ret = act_ret.mean()
        to = portfolio_turnover(prev_w, prev_s, w, opt_syms) if prev_w is not None else 1.0
        txc = to * cfg["cost_bps"] / 10000
        net_ret = port_ret - txc

        vrc = safe_spearmanr(p_vol, act_vol)
        rrc = safe_spearmanr(p_ret, act_ret)
        rrc_x = safe_spearmanr(p_xgb, act_ret)
        rrc_r = safe_spearmanr(p_rdg, act_ret)
        rrc_f = safe_spearmanr(p_rf, act_ret)

        rc_history.append({"xgb": rrc_x, "ridge": rrc_r, "rf": rrc_f})

        results.append({
            "test_date": td, "mkt_ret": mkt_ret, "gross_ret": port_ret,
            "net_ret": net_ret, "turnover": to, "tx_cost": txc,
            "vol_rc": vrc, "ret_rc": rrc,
            "ret_rc_xgb": rrc_x, "ret_rc_ridge": rrc_r, "ret_rc_rf": rrc_f,
            "used_lw": used_lw, "n_held": int((w > 0.001).sum()),
            "ens_w_xgb": ew.get("xgb", 0), "ens_w_ridge": ew.get("ridge", 0),
            "ens_w_rf": ew.get("rf", 0),
        })
        if return_detail:
            for i, sym in enumerate(test_syms):
                stock_detail.append({
                    "test_date": td, "symbol": sym,
                    "pred_ret": float(p_ret[i]), "act_ret": float(act_ret[i]),
                    "pred_vol": float(p_vol[i]) if i < len(p_vol) else np.nan,
                    "act_vol": float(act_vol[i]),
                })
        prev_w, prev_s = w, opt_syms

    df = pd.DataFrame(results)
    df["label"] = label
    if return_detail:
        return df, pd.DataFrame(stock_detail)
    return df


def _compute_adaptive_weights(rc_history, floor=0.10):
    """Same logic as predict.py compute_adaptive_weights."""
    if not rc_history:
        return {"xgb": 1/3, "ridge": 1/3, "rf": 1/3}
    models = ["xgb", "ridge", "rf"]
    scores = {}
    for m in models:
        vals = [h[m] for h in rc_history if np.isfinite(h.get(m, np.nan))]
        mean_rc = np.mean([max(v, 0) for v in vals]) if vals else 0.0
        scores[m] = mean_rc ** 2 + 1e-8
    total = sum(scores.values())
    raw = {m: scores[m] / total for m in models}
    floored = {m: max(raw[m], floor) for m in models}
    s = sum(floored.values())
    return {m: floored[m] / s for m in models}


# ── Metrics summary ────────────────────────────────────────────────────────────

def summarise(df, label):
    """Compute summary metrics from walk-forward results."""
    if len(df) == 0:
        return {"label": label, "n_quarters": 0}
    ex_n = df["net_ret"] - df["mkt_ret"]
    ex_g = df["gross_ret"] - df["mkt_ret"]
    sharpe = (df["net_ret"].mean() / df["net_ret"].std() * np.sqrt(4)) if df["net_ret"].std() > 0 else 0
    ir = (ex_n.mean() / ex_n.std() * np.sqrt(4)) if ex_n.std() > 0 else 0
    return {
        "label": label,
        "n_quarters": len(df),
        "avg_net_ret": df["net_ret"].mean(),
        "avg_mkt_ret": df["mkt_ret"].mean(),
        "avg_excess_net": ex_n.mean(),
        "avg_excess_gross": ex_g.mean(),
        "win_rate": (ex_n > 0).mean(),
        "sharpe_ann": sharpe,
        "ir_ann": ir,
        "avg_ret_rc": df["ret_rc"].mean(),
        "avg_ret_rc_xgb": df["ret_rc_xgb"].mean(),
        "avg_ret_rc_ridge": df["ret_rc_ridge"].mean(),
        "avg_ret_rc_rf": df["ret_rc_rf"].mean(),
        "avg_vol_rc": df["vol_rc"].mean(),
        "avg_turnover": df["turnover"].mean(),
        "avg_holdings": df["n_held"].mean(),
    }


def print_comparison(results_list):
    """Print a comparison table."""
    summaries = [summarise(r, r["label"].iloc[0]) for r in results_list]
    sdf = pd.DataFrame(summaries).set_index("label")

    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 100)

    fmt = {
        "n_quarters": "{:.0f}",
        "avg_net_ret": "{:+.3%}",
        "avg_mkt_ret": "{:+.3%}",
        "avg_excess_net": "{:+.3%}",
        "avg_excess_gross": "{:+.3%}",
        "win_rate": "{:.0%}",
        "sharpe_ann": "{:.3f}",
        "ir_ann": "{:.3f}",
        "avg_ret_rc": "{:.4f}",
        "avg_ret_rc_xgb": "{:.4f}",
        "avg_ret_rc_ridge": "{:.4f}",
        "avg_ret_rc_rf": "{:.4f}",
        "avg_vol_rc": "{:.4f}",
        "avg_turnover": "{:.0%}",
        "avg_holdings": "{:.0f}",
    }

    for col in sdf.columns:
        vals = []
        for v in sdf[col]:
            try:
                vals.append(fmt.get(col, "{:.4f}").format(v))
            except (ValueError, TypeError):
                vals.append(str(v))
        row = "  ".join(f"{v:>12s}" for v in vals)
        labels = "  ".join(f"{l:>12s}" for l in sdf.index)
        if col == sdf.columns[0]:
            print(f"\n  {'':30s} {labels}")
            print(f"  {'':30s} {'---':>12s}" * len(sdf))
        print(f"  {col:30s} {row}")

    return sdf


# ── Experiments ────────────────────────────────────────────────────────────────

def experiment_baseline():
    """Baseline: master code, all features for all models, fixed 50/25/25."""
    print("\n>>> BASELINE: Fixed weights (50/25/25), same XGB params for vol, all features for all models")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        ens_weights=ENS_W,
        vol_params=XGB_PARAMS,
        label="baseline",
    )


def experiment_21_equal_weights():
    """Issue #21 variant: equal weights 1/3 each."""
    print("\n>>> EXP #21a: Equal weights (1/3 each)")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        ens_weights={"xgb": 1/3, "ridge": 1/3, "rf": 1/3},
        label="equal_1/3",
    )


def experiment_21_adaptive_weights():
    """Issue #21 variant: adaptive weights from OOS RC."""
    print("\n>>> EXP #21b: Adaptive weights (OOS RC, lookback=4, floor=0.10)")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        adaptive_weights=True,
        adaptive_floor=0.10,
        adaptive_lookback=4,
        label="adaptive",
    )


def experiment_21_ridge_heavy():
    """Issue #21 variant: Ridge gets 50% (hypothesis: Ridge generalises better)."""
    print("\n>>> EXP #21c: Ridge-heavy (25/50/25)")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        ens_weights={"xgb": 0.25, "ridge": 0.50, "rf": 0.25},
        label="ridge_heavy",
    )


def experiment_32_vol_separate_params():
    """Issue #32: Deeper vol model (depth=5, less reg)."""
    print("\n>>> EXP #32a: Vol model — deeper trees (depth=5, min_child=5, less reg)")
    vol_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "tree_method": "hist", "random_state": 42,
    }
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=vol_params,
        label="vol_deep",
    )


def experiment_32_vol_with_gate():
    """Issue #32: Deeper vol + quality gate (fallback to hist_vol_3m)."""
    print("\n>>> EXP #32b: Vol model — deeper + quality gate (RC<0.10 → hist_vol_3m)")
    vol_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "tree_method": "hist", "random_state": 42,
    }
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=vol_params,
        vol_rc_gate=0.10,
        label="vol_deep+gate",
    )


def experiment_34_model_specific_features():
    """Issue #34: Raw features for trees, rank features for Ridge."""
    print("\n>>> EXP #34: Model-specific features (raw→trees, rank→Ridge)")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        tree_cols=raw_cols,
        ridge_feature_cols=rank_cols,
        label="model_specific",
    )


def experiment_34_raw_only():
    """Issue #34 ablation: raw features only for ALL models."""
    print("\n>>> EXP #34b: Raw features only (all models)")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        tree_cols=raw_cols,
        ridge_feature_cols=raw_cols,
        label="raw_only",
    )


def experiment_34_rank_only():
    """Issue #34 ablation: rank features only for ALL models."""
    print("\n>>> EXP #34c: Rank features only (all models)")
    return run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        tree_cols=rank_cols,
        ridge_feature_cols=rank_cols,
        label="rank_only",
    )



# ── Issue #31: Comprehensive robustness experiments ────────────────────────────

def robustness_31_subperiod_stability(baseline_df):
    """Split backtest into halves and compare metrics."""
    print("\n>>> ROBUSTNESS #31.1: Subperiod Stability")
    n = len(baseline_df)
    if n < 2:
        print("  ⚠ Only 1 quarter — cannot split into subperiods.")
        return
    mid = n // 2
    first_half = baseline_df.iloc[:mid].copy()
    second_half = baseline_df.iloc[mid:].copy()
    first_half["label"] = "first_half"
    second_half["label"] = "second_half"

    s1 = summarise(first_half, "first_half")
    s2 = summarise(second_half, "second_half")
    sf = summarise(baseline_df, "full_sample")

    print(f"  Full sample:  N={sf['n_quarters']}, excess={sf['avg_excess_net']:+.2%}, "
          f"Sharpe={sf['sharpe_ann']:.2f}, IC={sf['avg_ret_rc']:.4f}")
    print(f"  First half:   N={s1['n_quarters']}, excess={s1['avg_excess_net']:+.2%}, "
          f"Sharpe={s1['sharpe_ann']:.2f}, IC={s1['avg_ret_rc']:.4f}")
    print(f"  Second half:  N={s2['n_quarters']}, excess={s2['avg_excess_net']:+.2%}, "
          f"Sharpe={s2['sharpe_ann']:.2f}, IC={s2['avg_ret_rc']:.4f}")

    if s1["avg_excess_net"] * s2["avg_excess_net"] < 0:
        print("  ⚠ SIGN REVERSAL between halves — alpha is likely spurious.")
    else:
        print("  ✓ Consistent sign across halves.")
    return s1, s2


def robustness_31_decile_analysis(stock_detail_df):
    """Rank stocks by predicted return, report realized return by decile."""
    print("\n>>> ROBUSTNESS #31.3: Prediction Decile Analysis")
    if stock_detail_df.empty:
        print("  No stock-level data.")
        return

    all_deciles = []
    for td, grp in stock_detail_df.groupby("test_date"):
        if len(grp) < 10:
            continue
        grp = grp.copy()
        grp["decile"] = pd.qcut(grp["pred_ret"], 10, labels=False, duplicates="drop") + 1
        for d, dg in grp.groupby("decile"):
            all_deciles.append({"test_date": td, "decile": d, "avg_ret": dg["act_ret"].mean(), "n": len(dg)})

    if not all_deciles:
        print("  Insufficient data for decile analysis.")
        return

    dec_df = pd.DataFrame(all_deciles)
    agg = dec_df.groupby("decile")["avg_ret"].agg(["mean", "std", "count"]).reset_index()
    print(f"  {'Decile':>7s}  {'Avg Ret':>10s}  {'Std':>10s}  {'N quarters':>10s}")
    print(f"  {'-------':>7s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}")
    for _, row in agg.iterrows():
        print(f"  {int(row['decile']):7d}  {row['mean']:+10.2%}  {row['std']:10.2%}  {int(row['count']):10d}")

    top = agg[agg["decile"] == agg["decile"].max()]["mean"].values[0]
    bot = agg[agg["decile"] == agg["decile"].min()]["mean"].values[0]
    spread = top - bot
    print(f"\n  Long-short spread (D10 - D1): {spread:+.2%}")
    if spread > 0:
        print("  ✓ Monotonic spread — signal has content.")
    else:
        print("  ⚠ INVERTED spread — predicted top decile underperforms bottom.")
    return agg


def robustness_31_rolling_ic(baseline_df, window=4):
    """Rolling average IC over a window of quarters."""
    print(f"\n>>> ROBUSTNESS #31.4: Rolling IC (window={window} quarters)")
    if len(baseline_df) < window:
        print(f"  ⚠ Only {len(baseline_df)} quarters — need at least {window} for rolling IC.")
        return

    ic_series = baseline_df["ret_rc"].values
    dates = baseline_df["test_date"].values
    rolling = []
    for i in range(window - 1, len(ic_series)):
        avg_ic = np.mean(ic_series[i - window + 1:i + 1])
        rolling.append({"end_date": dates[i], "rolling_ic": avg_ic})

    rdf = pd.DataFrame(rolling)
    print(f"  Rolling IC (last {min(len(rdf), 8)} points):")
    for _, row in rdf.tail(8).iterrows():
        d = pd.Timestamp(row["end_date"]).date()
        print(f"    {d}: IC={row['rolling_ic']:.4f}")

    if len(rdf) >= 2:
        trend = np.polyfit(range(len(rdf)), rdf["rolling_ic"].values, 1)[0]
        if trend > 0.01:
            print("  Signal quality IMPROVING over time.")
        elif trend < -0.01:
            print("  ⚠ Signal quality DECAYING — possible overfitting to historical patterns.")
        else:
            print("  Signal quality STABLE.")
    return rdf


def robustness_31_param_sensitivity_grid():
    """Run walk-forward over a parameter grid."""
    print("\n>>> ROBUSTNESS #31.6: Parameter Sensitivity Grid")
    grid = [
        {"risk_aversion": ra, "max_weight": mw}
        for ra in [0.5, 1.0, 2.0, 5.0]
        for mw in [0.01, 0.02, 0.05]
    ]
    results = []
    for params in grid:
        cfg = PROD_CFG.copy()
        cfg.update(params)
        label = f"ra={params['risk_aversion']}_mw={params['max_weight']}"
        print(f"  Running: {label}")
        df = run_walk_forward(
            risk_model_df, feature_cols_all, close_prices, cfg,
            label=label,
        )
        s = summarise(df, label)
        s["risk_aversion"] = params["risk_aversion"]
        s["max_weight"] = params["max_weight"]
        results.append(s)

    rdf = pd.DataFrame(results)
    print(f"\n  {'risk_aversion':>14s}  {'max_weight':>10s}  {'excess_net':>12s}  {'sharpe':>8s}  {'turnover':>10s}")
    print(f"  {'-'*14:>14s}  {'-'*10:>10s}  {'-'*12:>12s}  {'-'*8:>8s}  {'-'*10:>10s}")
    for _, row in rdf.iterrows():
        print(f"  {row['risk_aversion']:14.1f}  {row['max_weight']:10.2f}  "
              f"{row['avg_excess_net']:+12.2%}  {row['sharpe_ann']:8.2f}  {row['avg_turnover']:10.0%}")

    excess_spread = rdf["avg_excess_net"].max() - rdf["avg_excess_net"].min()
    if excess_spread > 0.10:
        print(f"  ⚠ FRAGILE: excess range={excess_spread:.2%} across grid — alpha is parameter-sensitive.")
    else:
        print(f"  ✓ Robust: excess range={excess_spread:.2%} — stable across parameter choices.")
    return rdf


def robustness_31_permutation_test(n_perms=100):
    """Proper permutation test: shuffle stock↔prediction mapping each quarter."""
    print(f"\n>>> ROBUSTNESS #31.7: Permutation Test ({n_perms} shuffles)")

    # Real baseline
    real = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        label="real",
    )
    real_excess = (real["net_ret"] - real["mkt_ret"]).mean()

    # Permuted: same predictions, random cross-sectional assignment
    perm_excesses = []
    for i in range(n_perms):
        perm = run_walk_forward(
            risk_model_df, feature_cols_all, close_prices, PROD_CFG,
            permute_seed=i * 7 + 1,
            label=f"perm_{i}",
        )
        perm_excess = (perm["net_ret"] - perm["mkt_ret"]).mean()
        perm_excesses.append(perm_excess)
        print(f"  Perm {i+1}/{n_perms}: excess={perm_excess:+.2%}")

    perm_arr = np.array(perm_excesses)
    p_value = np.mean(perm_arr >= real_excess)
    print(f"\n  Real strategy excess: {real_excess:+.2%}")
    print(f"  Permutations mean:    {perm_arr.mean():+.2%}")
    print(f"  Permutations std:     {perm_arr.std():.2%}")
    print(f"  P-value (perm >= real): {p_value:.3f}")

    if p_value < 0.05:
        print("  ✓ Strategy significantly outperforms shuffled signals (p<0.05).")
    elif p_value < 0.10:
        print("  ~ Marginal significance (0.05 < p < 0.10).")
    else:
        print("  ⚠ Strategy does NOT significantly outperform shuffled signals.")

    return real_excess, perm_excesses, p_value


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}
    t0 = time.time()

    # Baseline
    baseline = experiment_baseline()
    all_results["baseline"] = baseline

    # Issue #21: Ensemble weights
    print("\n" + "=" * 100)
    print("EXPERIMENT GROUP: Issue #21 — Ensemble Weights")
    print("=" * 100)
    all_results["equal_1/3"] = experiment_21_equal_weights()
    all_results["adaptive"] = experiment_21_adaptive_weights()
    all_results["ridge_heavy"] = experiment_21_ridge_heavy()

    print_comparison([baseline, all_results["equal_1/3"],
                      all_results["adaptive"], all_results["ridge_heavy"]])

    # Issue #32: Vol model
    print("\n" + "=" * 100)
    print("EXPERIMENT GROUP: Issue #32 — Volatility Model")
    print("=" * 100)
    all_results["vol_deep"] = experiment_32_vol_separate_params()
    all_results["vol_deep+gate"] = experiment_32_vol_with_gate()

    print_comparison([baseline, all_results["vol_deep"], all_results["vol_deep+gate"]])

    # Issue #34: Feature sets
    print("\n" + "=" * 100)
    print("EXPERIMENT GROUP: Issue #34 — Feature Sets")
    print("=" * 100)
    all_results["model_specific"] = experiment_34_model_specific_features()
    all_results["raw_only"] = experiment_34_raw_only()
    all_results["rank_only"] = experiment_34_rank_only()

    print_comparison([baseline, all_results["model_specific"],
                      all_results["raw_only"], all_results["rank_only"]])

    # Per-quarter detail for adaptive weights
    print("\n" + "=" * 100)
    print("ADAPTIVE WEIGHTS DETAIL (per-quarter)")
    print("=" * 100)
    ada = all_results["adaptive"]
    if len(ada) > 0:
        for _, row in ada.iterrows():
            print(f"  {row['test_date'].date()} | w_xgb={row['ens_w_xgb']:.2f} w_ridge={row['ens_w_ridge']:.2f} "
                  f"w_rf={row['ens_w_rf']:.2f} | rc_ens={row['ret_rc']:.3f} rc_xgb={row['ret_rc_xgb']:.3f} "
                  f"rc_rdg={row['ret_rc_ridge']:.3f} rc_rf={row['ret_rc_rf']:.3f}")

    # Final combined comparison
    print("\n" + "=" * 100)
    print("FULL COMPARISON — ALL VARIANTS")
    print("=" * 100)
    all_dfs = [v for v in all_results.values()]
    final_table = print_comparison(all_dfs)

    elapsed = time.time() - t0
    print(f"\nTotal experiment time: {elapsed:.1f}s")

    # Save raw results
    out_dir = Path(__file__).resolve().parent
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "experiment_results.csv", index=False)
    print(f"Raw results saved to {out_dir / 'experiment_results.csv'}")
