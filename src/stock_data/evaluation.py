"""Strategy evaluation, simulation, and reporting functions.

Extracts the analysis logic that was previously inline in notebook cells
into reusable source code — per CCDS: source code for repetition,
notebooks for exploration and communication.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from stock_data.config import (
    EARNINGS_LAG_DAYS, ENS_W, PROD_CFG, XGB_PARAMS, RIDGE_PARAMS, RF_PARAMS,
    COST_BPS, WEIGHT_THRESHOLD, N_BOOT,
)
from stock_data.modeling.predict import bootstrap_ci, block_bootstrap_ci, power_analysis_quarters, safe_spearmanr


# ── Walk-forward summary ───────────────────────────────────────────────────────


def summarize_walk_forward(prod_df, prod_fi, feature_cols_all):
    """Print production walk-forward summary with feature importance stability."""
    ex_g = prod_df["gross_ret"] - prod_df["mkt_ret"]
    ex_n = prod_df["net_ret"] - prod_df["mkt_ret"]

    print(f"\n{'='*80}")
    print(f"PRODUCTION RESULTS ({len(prod_df)} quarters)")
    print(f"{'='*80}")

    metrics = [
        ("Avg return (gross)",      f"{prod_df['gross_ret'].mean():+.2%}"),
        ("Avg return (net)",        f"{prod_df['net_ret'].mean():+.2%}"),
        ("Avg market return (EW)",  f"{prod_df['mkt_ret'].mean():+.2%}"),
        ("Avg excess vs EW (gross)",f"{ex_g.mean():+.2%}"),
        ("Avg excess vs EW (net)",  f"{ex_n.mean():+.2%}"),
        ("Win rate (net > EW)",     f"{(ex_n > 0).sum()}/{len(prod_df)} ({(ex_n > 0).mean():.0%})"),
        ("Avg holdings",            f"{prod_df['n_held'].mean():.0f}"),
        ("Avg max weight",          f"{prod_df['max_wt'].mean():.2%}"),
        ("Avg turnover (one-way)",  f"{prod_df['turnover'].mean():.0%}"),
        ("Avg tx cost",             f"{prod_df['tx_cost'].mean():.2%}"),
        ("Ledoit-Wolf used",        f"{prod_df['used_lw'].sum()}/{len(prod_df)}"),
    ]
    if "spx_ret" in prod_df.columns and prod_df["spx_ret"].notna().any():
        spx_valid = prod_df[prod_df["spx_ret"].notna()]
        ex_spx = spx_valid["net_ret"] - spx_valid["spx_ret"]
        metrics.insert(3, ("Avg S&P 500 return", f"{spx_valid['spx_ret'].mean():+.2%}"))
        metrics.insert(6, ("Avg excess vs SPX (net)", f"{ex_spx.mean():+.2%}"))
        metrics.insert(7, ("Win rate (net > SPX)",
                           f"{(ex_spx > 0).sum()}/{len(spx_valid)} ({(ex_spx > 0).mean():.0%})"))
    for label, val in metrics:
        print(f"  {label:30s} {val:>10s}")

    print(f"\n  Model quality (avg rank correlation):")
    print(f"    Vol model:       {prod_df['vol_rc'].mean():.3f}")
    print(f"    Ret ensemble:    {prod_df['ret_rc'].mean():.3f}")
    print(f"    Ret XGBoost:     {prod_df['ret_rc_xgb'].mean():.3f}")
    print(f"    Ret Ridge:       {prod_df['ret_rc_ridge'].mean():.3f}")
    print(f"    Ret RF:          {prod_df['ret_rc_rf'].mean():.3f}")

    # Feature importance stability (combined multi-source)
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE STABILITY (top 15 by average)")
    print(f"{'='*80}\n")

    fi_wide = pd.DataFrame({r["date"].date(): r["fi"] for r in prod_fi})
    top15 = fi_wide.mean(axis=1).sort_values(ascending=False).head(15)
    print(f"  {'Feature':38s} {'Mean':>7s} {'Std':>7s} {'SNR':>7s}")
    print(f"  {'-'*62}")
    for f in top15.index:
        m, s = fi_wide.loc[f].mean(), fi_wide.loc[f].std()
        print(f"  {f:38s} {m:.4f}  {s:.4f}  {m / (s + 1e-10):.2f}")

    # Multi-source FI breakdown (if available)
    if prod_fi and "fi_detail" in prod_fi[0]:
        print(f"\n{'='*80}")
        print("MULTI-SOURCE FEATURE IMPORTANCE (last quarter)")
        print(f"{'='*80}\n")
        last_detail = prod_fi[-1]["fi_detail"]
        top15_detail = last_detail.sort_values("combined", ascending=False).head(15)
        print(f"  {'Feature':38s} {'XGB':>7s} {'Ridge':>7s} {'RF':>7s} {'Comb':>7s}")
        print(f"  {'-'*70}")
        for feat_name, row in top15_detail.iterrows():
            print(f"  {feat_name:38s} {row['xgb_gain']:.4f}  {row['ridge_coef']:.4f}  "
                  f"{row['rf_impurity']:.4f}  {row['combined']:.4f}")


# ── Factor benchmarks + bootstrap ─────────────────────────────────────────────


def evaluate_factors(prod_df, factor_results, n_boot=N_BOOT):
    """Print factor benchmark comparison, bootstrap CIs, and final assessment.

    Returns (ci_lo, ci_hi, boot_means, ex_n) for downstream plotting.
    """
    ex_n = prod_df["net_ret"] - prod_df["mkt_ret"]

    # SPX (cap-weighted) benchmark excess, if available
    has_spx = "spx_ret" in prod_df.columns and prod_df["spx_ret"].notna().any()
    if has_spx:
        ex_spx = prod_df["net_ret"] - prod_df["spx_ret"]

    print("=" * 80)
    print("FACTOR BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"\n  {'Strategy':25s} {'Avg Q Ret':>9s} {'Excess':>8s} {'Win%':>6s} {'N Q':>4s}")
    print(f"  {'-'*55}")

    print(f"  {'Production MV (net)':25s} {prod_df['net_ret'].mean():+8.2%} "
          f"{ex_n.mean():+7.2%} {(ex_n > 0).mean():5.0%} {len(prod_df):3d}")
    print(f"  {'Market (equal wt)':25s} {prod_df['mkt_ret'].mean():+8.2%} "
          f"{'---':>8s} {'---':>6s} {len(prod_df):3d}")

    if has_spx:
        spx_valid = prod_df[prod_df["spx_ret"].notna()]
        print(f"  {'S&P 500 (cap wt)':25s} {spx_valid['spx_ret'].mean():+8.2%} "
              f"{'---':>8s} {'---':>6s} {len(spx_valid):3d}")
        print(f"  {'MV excess vs SPX':25s} {'---':>9s} "
              f"{ex_spx.dropna().mean():+7.2%} {(ex_spx.dropna() > 0).mean():5.0%} "
              f"{ex_spx.notna().sum():3d}")

    factor_excess_for_boot = {}
    for fname, results in factor_results.items():
        if not results:
            continue
        fdf = pd.DataFrame(results)
        fdf_match = fdf[fdf["test_date"].isin(prod_df["test_date"].values)].sort_values("test_date")
        if len(fdf_match) < 2:
            continue
        avg_ret = fdf_match["port_ret"].mean()
        avg_ex = fdf_match["excess"].mean()
        win = (fdf_match["excess"] > 0).mean()
        # Show net returns if turnover was tracked
        if "net_ret" in fdf_match.columns:
            avg_net = fdf_match["net_ret"].mean()
            avg_ex_net = fdf_match["excess_net"].mean()
            avg_to = fdf_match["turnover"].mean()
            print(f"  {fname:25s} {avg_ret:+8.2%} {avg_ex:+7.2%} {win:5.0%} {len(fdf_match):3d}"
                  f"  (net: {avg_ex_net:+.2%}, TO: {avg_to:.0%})")
            factor_excess_for_boot[fname] = fdf_match["excess_net"].values
        else:
            print(f"  {fname:25s} {avg_ret:+8.2%} {avg_ex:+7.2%} {win:5.0%} {len(fdf_match):3d}")
            factor_excess_for_boot[fname] = fdf_match["excess"].values

    # Bootstrap
    print(f"\n{'='*80}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_boot:,} resamples)")
    print(f"{'='*80}")

    ex_net_vals = ex_n.values
    ci_lo, ci_hi, boot_p, boot_means = bootstrap_ci(ex_net_vals, n_boot)

    print(f"\n  Production MV (net of costs):")
    print(f"    Point estimate:   {ex_net_vals.mean():+.2%}")
    print(f"    95% CI:           [{ci_lo:+.2%}, {ci_hi:+.2%}]")
    print(f"    P(excess <= 0):   {boot_p:.3f}")
    print(f"    N quarters:       {len(ex_net_vals)}")

    if has_spx:
        spx_vals = ex_spx.dropna().values
        if len(spx_vals) >= 3:
            slo, shi, sp, _ = bootstrap_ci(spx_vals, n_boot)
            print(f"\n  Production MV excess vs S&P 500:")
            print(f"    Point estimate:   {spx_vals.mean():+.2%}")
            print(f"    95% CI:           [{slo:+.2%}, {shi:+.2%}]")
            print(f"    P(excess <= 0):   {sp:.3f}")
            print(f"    N quarters:       {len(spx_vals)}")

    # Statistical power warning
    n_q = len(ex_net_vals)
    if ex_n.std() > 0 and ex_n.mean() != 0:
        n_needed = power_analysis_quarters(ex_n.mean(), ex_n.std())
        years_needed = n_needed / 4
        print(f"\n  POWER ANALYSIS:")
        if n_q < 20:
            print(f"    ⚠  N={n_q} — INSUFFICIENT for statistical significance at conventional levels")
        print(f"    Given observed excess vol ({ex_n.std():.2%}):")
        print(f"    Need N={n_needed} quarters ({years_needed:.0f} years) for 80% power at α=0.05")

    # Block bootstrap (preserving quarterly autocorrelation)
    if n_q >= 8:
        blo, bhi, bp, _ = block_bootstrap_ci(ex_net_vals, block_size=4, n_boot=n_boot)
        print(f"\n  Block bootstrap (block_size=4, preserving autocorrelation):")
        print(f"    95% CI:           [{blo:+.2%}, {bhi:+.2%}]")
        print(f"    P(excess <= 0):   {bp:.3f}")

    for fname, fex in factor_excess_for_boot.items():
        if len(fex) < 3:
            continue
        flo, fhi, fp, _ = bootstrap_ci(fex, n_boot)
        print(f"\n  {fname}:")
        print(f"    Excess: {fex.mean():+.2%}  CI: [{flo:+.2%}, {fhi:+.2%}]  P(<=0): {fp:.3f}")

    # Final assessment
    ann_excess_net = (1 + ex_n.mean())**4 - 1
    net_ir = (ex_n.mean() / ex_n.std() * np.sqrt(4)) if ex_n.std() > 0 else 0
    net_sharpe = (prod_df["net_ret"].mean() / prod_df["net_ret"].std() * np.sqrt(4)) if prod_df["net_ret"].std() > 0 else 0
    wealth_net = (1 + prod_df["net_ret"]).cumprod()
    max_dd = ((wealth_net / wealth_net.cummax()) - 1).min()
    ann_ret_net = (1 + prod_df["net_ret"].mean())**4 - 1
    calmar = ann_ret_net / abs(max_dd) if max_dd != 0 else 0

    if len(ex_n) >= 3:
        t_stat, p_val = stats.ttest_1samp(ex_n, 0)
    else:
        t_stat, p_val = 0, 1.0

    print(f"\n{'='*80}")
    print("FINAL PRODUCTION ASSESSMENT")
    print(f"{'='*80}")
    spx_section = ""
    if has_spx:
        spx_ex = ex_spx.dropna()
        ann_excess_spx = (1 + spx_ex.mean())**4 - 1
        spx_ir = (spx_ex.mean() / spx_ex.std() * np.sqrt(4)) if spx_ex.std() > 0 else 0
        spx_section = f"""
  --- vs S&P 500 (cap-weighted) ---
  Ann excess vs SPX:     {ann_excess_spx:+.1%}
  IR vs SPX:             {spx_ir:.2f}
  Win rate vs SPX:       {(spx_ex > 0).sum()}/{len(spx_ex)} ({(spx_ex > 0).mean():.0%})"""

    print(f"""
  --- vs Equal-Weight Universe ---
  Ann excess (EW):       {ann_excess_net:+.1%}
  Sharpe (ann):          {net_sharpe:.2f}
  IR (vs EW):            {net_ir:.2f}
  Calmar ratio:          {calmar:.2f}
  Win rate vs EW:        {(ex_n > 0).sum()}/{len(prod_df)} ({(ex_n > 0).mean():.0%})
  Max drawdown:          {max_dd:+.2%}
  t-statistic:           {t_stat:.2f}
  p-value:               {p_val:.3f}
  Bootstrap 95% CI:      [{ci_lo:+.2%}, {ci_hi:+.2%}]
  Bootstrap P(<=0):      {boot_p:.3f}
{spx_section}
""")

    return ci_lo, ci_hi, boot_means, ex_n


# ── External validity ──────────────────────────────────────────────────────────


def assess_external_validity(
    prod_df, prod_fi, prod_weights_history,
    risk_model_df, feature_cols_all, close_prices,
):
    """Print external validity assessment: regime, prediction quality,
    concentration, FI stability, and return attribution."""
    X_p = risk_model_df[feature_cols_all].copy()
    dp = X_p.index.get_level_values("date")

    print("=" * 80)
    print("EXTERNAL VALIDITY ASSESSMENT")
    print("=" * 80)

    # 1. Regime dependence
    print("\n1. REGIME DEPENDENCE")
    print("-" * 60)

    if len(prod_df) >= 2:
        prod_df = prod_df.copy()
        prod_df["mkt_regime"] = np.where(prod_df["mkt_ret"] >= 0, "Bull", "Bear")
        prod_df["excess_net"] = prod_df["net_ret"] - prod_df["mkt_ret"]

        for regime in ["Bull", "Bear"]:
            subset = prod_df[prod_df["mkt_regime"] == regime]
            if len(subset) > 0:
                print(f"\n  {regime} quarters ({len(subset)}):")
                print(f"    Avg excess: {subset['excess_net'].mean():+.2%}")
                print(f"    Win rate:   {(subset['excess_net'] > 0).mean():.0%}")

        if len(prod_df) >= 3:
            corr, pval = pearsonr(prod_df["mkt_ret"], prod_df["excess_net"])
            print(f"\n  Market-excess correlation: {corr:.3f} (p={pval:.3f})")

    # 2. Prediction quality
    print(f"\n2. PREDICTION QUALITY ANALYSIS")
    print("-" * 60)
    for _, row in prod_df.iterrows():
        d = row["test_date"].date()
        print(f"    {d}: vol_rc={row['vol_rc']:.3f}  ret_rc={row['ret_rc']:.3f}  "
              f"ret_xgb={row['ret_rc_xgb']:.3f}")

    # 3. Concentration risk
    print(f"\n3. CONCENTRATION RISK")
    print("-" * 60)
    for td, wdata in prod_weights_history.items():
        w = wdata["weights"]
        n_held = (w > 0.001).sum()
        top5_wt = np.sort(w)[-5:].sum()
        eff_n = 1.0 / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0
        print(f"  {td.date()}: {n_held} holdings, top-5={top5_wt:.1%}, eff_N={eff_n:.0f}")

    # 4. Feature importance stability
    print(f"\n4. FEATURE IMPORTANCE STABILITY")
    print("-" * 60)
    if len(prod_fi) >= 2:
        fi_df = pd.DataFrame({r["date"].date(): r["fi"] for r in prod_fi})
        fi_ranks = fi_df.rank(ascending=False)
        top10 = fi_df.mean(axis=1).sort_values(ascending=False).head(10).index
        print(f"\n  {'Feature':38s} {'Avg Rank':>8s} {'Rank Std':>8s}")
        for feat_name in top10:
            ranks = fi_ranks.loc[feat_name]
            print(f"  {feat_name:38s} {ranks.mean():7.1f} {ranks.std():8.1f}")

        if fi_df.shape[1] >= 2:
            cols = fi_df.columns.tolist()
            qq_corrs = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c, _ = spearmanr(fi_df[cols[i]], fi_df[cols[j]])
                    qq_corrs.append(c)
            print(f"\n  Feature importance rank corr across quarters: "
                  f"{np.mean(qq_corrs):.3f} (avg)")

    # 5. Return attribution
    print(f"\n5. RETURN ATTRIBUTION")
    print("-" * 60)
    for td, wdata in prod_weights_history.items():
        w, syms = wdata["weights"], wdata["symbols"]
        te_m = dp == td
        Xte_attr = X_p[te_m]
        act_ret_attr = risk_model_df["next_q_return"].loc[Xte_attr.index].values
        sym_idx = {s: i for i, s in enumerate(Xte_attr.index.get_level_values("symbol"))}
        mkt_r = act_ret_attr.mean()
        contrib = []
        for i, s in enumerate(syms):
            if s in sym_idx:
                idx = sym_idx[s]
                stock_ex = act_ret_attr[idx] - mkt_r
                contrib.append((s, w[i], act_ret_attr[idx], stock_ex, w[i] * stock_ex))
        contrib.sort(key=lambda x: -abs(x[4]))
        port_excess = sum(c[4] for c in contrib)
        print(f"\n  {td.date()} — Portfolio excess: {port_excess:+.2%}")
        for s, wt, ret_s, ex, c in contrib[:5]:
            print(f"    {s:>8s} {wt:6.1%} {ret_s:+6.2%} {ex:+6.2%} {c:+7.3%}")


# ── Portfolio simulation ───────────────────────────────────────────────────────


def simulate_portfolio(
    prod_df, prod_weights_history, risk_model_df, close_prices,
    initial_capital, cost_bps=COST_BPS, weight_threshold=WEIGHT_THRESHOLD,
):
    """Run a realistic daily portfolio simulation.

    Returns (sim_df, mkt_sim, qlog) DataFrames.
    """
    daily_records = []
    portfolio_value = initial_capital
    prev_weights_map = {}
    quarter_log = []

    for _, row in prod_df.iterrows():
        td = row["test_date"]
        if td not in prod_weights_history:
            continue
        wdata = prod_weights_history[td]
        raw_w, raw_s = wdata["weights"], wdata["symbols"]
        mask = raw_w > weight_threshold
        s = [sym for sym, m in zip(raw_s, mask) if m]
        w = raw_w[mask]
        w = w / w.sum()

        buy_dt = td + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        sell_dt = td + pd.DateOffset(months=3) + pd.Timedelta(days=EARNINGS_LAG_DAYS)

        held_prices = close_prices[
            (close_prices["symbol"].isin(s)) &
            (close_prices["date"] >= buy_dt) &
            (close_prices["date"] <= sell_dt)
        ].pivot(index="date", columns="symbol", values="close").sort_index()
        if len(held_prices) < 5:
            continue
        avail = [sym for sym in s if sym in held_prices.columns]
        if len(avail) < 10:
            continue

        w_map = dict(zip(s, w))
        weights_vec = np.array([w_map.get(sym, 0) for sym in avail])
        weights_vec = weights_vec / weights_vec.sum()
        daily_ret = held_prices[avail].pct_change().dropna()
        if len(daily_ret) < 3:
            continue

        turnover = sum(abs(w_map.get(sym, 0) - prev_weights_map.get(sym, 0))
                       for sym in set(avail) | set(prev_weights_map.keys())) / 2
        if not prev_weights_map:
            turnover = 1.0
        tx_cost = turnover * cost_bps / 10000
        portfolio_value *= (1 - tx_cost)
        start_val = portfolio_value
        current_weights = weights_vec.copy()

        for day_idx in range(len(daily_ret)):
            day_return = (current_weights * daily_ret.iloc[day_idx][avail].values).sum()
            portfolio_value *= (1 + day_return)
            stock_vals = current_weights * (1 + daily_ret.iloc[day_idx][avail].values)
            current_weights = stock_vals / stock_vals.sum()
            daily_records.append({
                "date": daily_ret.index[day_idx], "quarter": td,
                "portfolio_value": portfolio_value, "daily_return": day_return,
            })

        quarter_ret = portfolio_value / start_val - 1
        quarter_log.append({
            "quarter": td, "start_value": start_val, "end_value": portfolio_value,
            "sim_return": quarter_ret, "prod_net_return": row["net_ret"],
            "market_return": row["mkt_ret"], "turnover": turnover,
            "tx_cost": tx_cost, "n_stocks": len(avail), "n_days": len(daily_ret),
        })
        prev_weights_map = dict(zip(avail, weights_vec))
        print(f"  Q {td.date()}: ${start_val:>12,.0f} → ${portfolio_value:>12,.0f} "
              f"({quarter_ret:+.2%}) | {len(avail)} stocks")

    # Market benchmark
    mkt_value = initial_capital
    mkt_daily = []
    for _, row in prod_df.iterrows():
        td = row["test_date"]
        buy_dt = td + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        sell_dt = td + pd.DateOffset(months=3) + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        all_syms = risk_model_df.loc[
            risk_model_df.index.get_level_values("date") == td
        ].index.get_level_values("symbol").tolist()
        mkt_px = close_prices[
            (close_prices["symbol"].isin(all_syms)) &
            (close_prices["date"] >= buy_dt) & (close_prices["date"] <= sell_dt)
        ].pivot(index="date", columns="symbol", values="close").sort_index()
        if len(mkt_px) < 5:
            continue
        mkt_ret = mkt_px.pct_change().dropna().mean(axis=1)
        for day_idx in range(len(mkt_ret)):
            mkt_value *= (1 + mkt_ret.iloc[day_idx])
            mkt_daily.append({"date": mkt_ret.index[day_idx], "market_value": mkt_value})

    sim_df = pd.DataFrame(daily_records).set_index("date").sort_index()
    mkt_sim = pd.DataFrame(mkt_daily).set_index("date").sort_index()
    qlog = pd.DataFrame(quarter_log)
    return sim_df, mkt_sim, qlog


def print_simulation_summary(sim_df, mkt_sim, initial_capital):
    """Print final simulation results."""
    combined = sim_df[["portfolio_value"]].join(mkt_sim[["market_value"]], how="inner")
    final_port = combined["portfolio_value"].iloc[-1]
    final_mkt = combined["market_value"].iloc[-1]
    n_days = len(combined)
    n_years = n_days / 252
    port_daily_ret = combined["portfolio_value"].pct_change().dropna()
    mkt_daily_ret = combined["market_value"].pct_change().dropna()
    port_ann_ret = (final_port / initial_capital) ** (1 / max(n_years, 0.1)) - 1
    mkt_ann_ret = (final_mkt / initial_capital) ** (1 / max(n_years, 0.1)) - 1
    port_ann_vol = port_daily_ret.std() * np.sqrt(252)

    # Information ratio (excess return / tracking error, annualized)
    excess_daily = port_daily_ret - mkt_daily_ret
    tracking_error = excess_daily.std() * np.sqrt(252)
    ir = (excess_daily.mean() * 252) / tracking_error if tracking_error > 0 else 0

    # Max drawdown from daily simulation
    cum_ret = (1 + port_daily_ret).cumprod()
    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min()

    # Calmar ratio (annualized return / abs(max drawdown))
    calmar = port_ann_ret / abs(max_dd) if max_dd != 0 else 0

    print(f"\n{'='*80}")
    print(f"SIMULATION RESULTS")
    print(f"{'='*80}")
    print(f"  Final value:  ${final_port:>11,.0f} (strategy)  ${final_mkt:>11,.0f} (market)")
    print(f"  Ann return:   {port_ann_ret:>11.2%} (strategy)  {mkt_ann_ret:>11.2%} (market)")
    print(f"  Ann vol:      {port_ann_vol:>11.2%}")
    if port_ann_vol > 0:
        print(f"  Sharpe:       {port_ann_ret / port_ann_vol:>11.2f}")
    print(f"  Info ratio:   {ir:>11.2f}")
    print(f"  Max drawdown: {max_dd:>11.2%}")
    print(f"  Calmar:       {calmar:>11.2f}")


# ── Iteration analysis ─────────────────────────────────────────────────────────


def run_iteration_analysis(
    prod_df, prod_fi, prod_weights_history,
    risk_model_df, feature_cols_all, close_prices,
):
    """Stable-feature model + factor-neutral attribution + vol model comparison."""
    X_p = risk_model_df[feature_cols_all].copy()
    y_ret_p = risk_model_df["next_q_return"].copy()
    y_vol_p = risk_model_df["realized_vol"].copy()
    ok_mask = y_ret_p.notna() & y_vol_p.notna()
    X_p, y_ret_p, y_vol_p = X_p[ok_mask], y_ret_p[ok_mask], y_vol_p[ok_mask]
    dp = X_p.index.get_level_values("date")

    print("=" * 80)
    print("ITERATION ON EXTERNAL VALIDITY FINDINGS")
    print("=" * 80)

    # A. Stable-feature model
    print("\nA. STABLE-FEATURE MODEL")
    print("-" * 60)

    if len(prod_fi) >= 2:
        fi_df = pd.DataFrame({r["date"].date(): r["fi"] for r in prod_fi})
        fi_ranks = fi_df.rank(ascending=False)
        rank_std = fi_ranks.std(axis=1)
        mean_imp = fi_df.mean(axis=1)
        top50 = mean_imp.sort_values(ascending=False).head(50).index
        stable_top = [f for f in top50 if rank_std.get(f, 999) < 20]
        stable_feat = [f for f in stable_top if f in X_p.columns]

        if len(stable_feat) >= 5:
            print(f"  Re-running with {len(stable_feat)} stable features...")
            unique_dates = sorted(dp.unique())
            max_q = PROD_CFG.get("max_train_q")
            stable_results = []
            for _, row in prod_df.iterrows():
                td = row["test_date"]
                tr_dates = [d for d in unique_dates if d < td]
                if max_q and len(tr_dates) > max_q:
                    tr_dates = tr_dates[-max_q:]
                tr_m = dp.isin(tr_dates)
                te_m = dp == td
                if tr_m.sum() < 50:
                    continue
                Xtr_raw = X_p.loc[tr_m, stable_feat]
                Xte_raw = X_p.loc[te_m, stable_feat]
                # XGBoost handles NaN natively (matches train.py)
                xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
                xgb_m.fit(Xtr_raw, y_ret_p[tr_m], verbose=0)
                # Ridge and RF get median-imputed + scaled data
                imp = SimpleImputer(strategy="median")
                sc = StandardScaler()
                X_tr_n = sc.fit_transform(imp.fit_transform(Xtr_raw.values))
                X_te_n = sc.transform(imp.transform(Xte_raw.values))
                rdg_m = Ridge(alpha=RIDGE_PARAMS["alpha"])
                rdg_m.fit(X_tr_n, y_ret_p[tr_m])
                rf_m = RandomForestRegressor(**RF_PARAMS)
                rf_m.fit(X_tr_n, y_ret_p[tr_m])
                p_ens = (ENS_W["xgb"] * xgb_m.predict(Xte_raw)
                         + ENS_W["ridge"] * rdg_m.predict(X_te_n)
                         + ENS_W["rf"] * rf_m.predict(X_te_n))
                rrc, _ = spearmanr(p_ens, y_ret_p[te_m])
                stable_results.append({
                    "quarter": td, "ret_rc_stable": rrc, "ret_rc_full": row["ret_rc"],
                })

            if stable_results:
                sdf = pd.DataFrame(stable_results)
                print(f"\n  {'Quarter':>12s} {'Stable RC':>10s} {'Full RC':>10s}")
                for _, sr in sdf.iterrows():
                    print(f"  {str(sr['quarter'].date()):>12s} "
                          f"{sr['ret_rc_stable']:>+9.3f} {sr['ret_rc_full']:>+9.3f}")
                print(f"\n  Average: stable={sdf['ret_rc_stable'].mean():.3f}, "
                      f"full={sdf['ret_rc_full'].mean():.3f}")

    # B. Factor-neutral attribution
    print(f"\n{'='*60}")
    print("B. FACTOR-NEUTRAL ATTRIBUTION")
    print("-" * 60)

    factor_cols_check = {
        "Momentum (3m)": "momentum_3m_rank", "Low Vol": "hist_vol_3m_rank",
        "Quality": "roe_rank", "Value": "fcf_to_assets_rank",
        "Size": "log_total_assets_rank",
    }
    factor_defs_iter = {k: v for k, v in factor_cols_check.items() if v in X_p.columns}
    factor_tilts = {fn: [] for fn in factor_defs_iter}

    for td, wdata in prod_weights_history.items():
        raw_w, raw_s = wdata["weights"], wdata["symbols"]
        mask = raw_w > WEIGHT_THRESHOLD
        w = raw_w[mask]
        w = w / w.sum()
        s = [sym for sym, m in zip(raw_s, mask) if m]
        te_m = dp == td
        X_te_q = X_p.loc[te_m]
        sym_data = X_te_q.index.get_level_values("symbol")
        for fn, fc in factor_defs_iter.items():
            univ_vals = X_te_q[fc].dropna()
            univ_mean, univ_std = univ_vals.mean(), univ_vals.std()
            port_vals = []
            for i, sym in enumerate(s):
                sym_mask = sym_data == sym
                if sym_mask.any():
                    val = X_te_q.loc[sym_mask, fc].values[0]
                    if pd.notna(val):
                        port_vals.append(w[i] * val)
            port_avg = sum(port_vals) if port_vals else univ_mean
            z = (port_avg - univ_mean) / univ_std if univ_std > 0 else 0
            factor_tilts[fn].append(z)

    print(f"\n  Factor tilt summary (|z| > 0.3 = meaningful):")
    for fn in factor_defs_iter:
        avg_z = np.mean(factor_tilts[fn])
        direction = "overweight" if avg_z > 0 else "underweight"
        marker = "\u26a0" if abs(avg_z) > 0.3 else "\u2713"
        print(f"    {marker} {fn}: {direction} (avg z={avg_z:+.2f})")

    # C. Vol model vs naive
    print(f"\n{'='*60}")
    print("C. VOL MODEL: ML vs NAIVE")
    print("-" * 60)
    for _, row in prod_df.iterrows():
        td = row["test_date"]
        te_m = dp == td
        Xte_v = X_p.loc[te_m]
        act_vol = risk_model_df["realized_vol"].loc[Xte_v.index].values
        if "hist_vol_3m" in Xte_v.columns:
            naive_pred = Xte_v["hist_vol_3m"].values
            valid = ~np.isnan(naive_pred) & ~np.isnan(act_vol)
            if valid.sum() > 10:
                naive_rc, _ = spearmanr(naive_pred[valid], act_vol[valid])
                print(f"  {td.date()}: ML={row['vol_rc']:.3f}  "
                      f"Naive={naive_rc:.3f}  \u0394={row['vol_rc']-naive_rc:+.3f}")


# ── Cost sensitivity analysis ──────────────────────────────────────────────────


def cost_sensitivity_analysis(prod_df, cost_bps_list=None):
    """Report strategy net performance at multiple cost assumptions.

    Parameters
    ----------
    prod_df : DataFrame with columns gross_ret, mkt_ret, turnover
    cost_bps_list : list of ints, cost assumptions in basis points
    """
    from stock_data.config import COST_SENSITIVITY_BPS

    if cost_bps_list is None:
        cost_bps_list = COST_SENSITIVITY_BPS

    print("\n" + "=" * 80)
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\n  {'Cost (bps)':>10s} {'Avg Net':>9s} {'Avg Excess':>11s} "
          f"{'Sharpe':>7s} {'IR':>7s} {'Win%':>6s}")
    print(f"  {'-'*55}")

    for bps in cost_bps_list:
        txc = prod_df["turnover"] * bps / 10000
        net = prod_df["gross_ret"] - txc
        ex = net - prod_df["mkt_ret"]
        sharpe = (net.mean() / net.std() * np.sqrt(4)) if net.std() > 0 else 0
        ir = (ex.mean() / ex.std() * np.sqrt(4)) if ex.std() > 0 else 0
        win = (ex > 0).mean()
        print(f"  {bps:>10d} {net.mean():+8.2%} {ex.mean():+10.2%} "
              f"{sharpe:>7.2f} {ir:>7.2f} {win:5.0%}")
