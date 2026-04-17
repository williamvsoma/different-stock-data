"""Feature engineering: fundamental ratios, momentum, macro, risk, and ranks."""

import numpy as np
import pandas as pd

from stock_data.config import ANNUAL_FILING_LAG, EARNINGS_LAG_DAYS


# ── Helpers ────────────────────────────────────────────────────────────────────


def gcol(df, name, default=np.nan):
    """Get a column safely, returning *default* if not present."""
    if name in df.columns:
        return df[name]
    if isinstance(default, pd.Series):
        return default
    return pd.Series(default, index=df.index)


def qoq_growth(s):
    return s.groupby(level="symbol").pct_change()


def qoq_diff(s):
    return s.groupby(level="symbol").diff()


# ── Core feature blocks ───────────────────────────────────────────────────────


def profitability_features(features_raw: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=features_raw.index)
    rev = gcol(features_raw, "Total Revenue", None)
    if rev is None or rev.isna().all():
        rev = gcol(features_raw, "Operating Revenue")
    ni = gcol(features_raw, "Net Income", 0)
    ebit = gcol(features_raw, "EBIT", gcol(features_raw, "Operating Income", 0))
    ebitda = gcol(features_raw, "EBITDA", 0)
    gp = gcol(features_raw, "Gross Profit", 0)
    rev_safe = rev.replace(0, np.nan)

    feat["gross_margin"] = gp / rev_safe
    feat["operating_margin"] = ebit / rev_safe
    feat["net_margin"] = ni / rev_safe
    feat["ebitda_margin"] = ebitda / rev_safe
    feat["tax_rate"] = (
        gcol(features_raw, "Tax Provision", 0)
        / gcol(features_raw, "Pretax Income").replace(0, np.nan)
    )
    feat["interest_coverage"] = (
        ebit / gcol(features_raw, "Interest Expense").replace(0, np.nan)
    )
    feat["rd_intensity"] = gcol(features_raw, "Research And Development", 0) / rev_safe
    feat["sga_intensity"] = (
        gcol(features_raw, "Selling General And Administration", 0) / rev_safe
    )
    feat["cost_efficiency"] = gcol(features_raw, "Cost Of Revenue", 0) / rev_safe
    return feat


def size_features(features_raw: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=features_raw.index)
    rev = gcol(features_raw, "Total Revenue", gcol(features_raw, "Operating Revenue"))
    ni = gcol(features_raw, "Net Income", 0)
    ebitda = gcol(features_raw, "EBITDA", 0)

    feat["log_revenue"] = np.log1p(rev.clip(lower=0))
    feat["log_net_income"] = np.sign(ni) * np.log1p(ni.abs())
    feat["log_ebitda"] = np.sign(ebitda) * np.log1p(ebitda.abs())
    feat["diluted_eps"] = gcol(features_raw, "Diluted EPS")
    return feat


def balance_sheet_features(features_raw, bs_raw) -> pd.DataFrame:
    feat = pd.DataFrame(index=features_raw.index)
    bs = bs_raw.reindex(features_raw.index)

    rev = gcol(features_raw, "Total Revenue", gcol(features_raw, "Operating Revenue"))
    ni = gcol(features_raw, "Net Income", 0)
    ebit = gcol(features_raw, "EBIT", gcol(features_raw, "Operating Income", 0))

    total_assets = gcol(bs, "Total Assets")
    total_equity = gcol(bs, "Stockholders Equity",
                        gcol(bs, "Total Equity Gross Minority Interest"))
    total_debt = gcol(bs, "Total Debt", 0)
    current_assets = gcol(bs, "Current Assets")
    current_liabilities = gcol(bs, "Current Liabilities")
    cash = gcol(bs, "Cash And Cash Equivalents", 0)
    inventory = gcol(bs, "Inventory", 0)
    receivables = gcol(bs, "Accounts Receivable", 0)

    feat["roe"] = ni / total_equity.replace(0, np.nan)
    feat["roa"] = ni / total_assets.replace(0, np.nan)
    feat["roic"] = ebit / (total_equity + total_debt).replace(0, np.nan)
    feat["debt_to_equity"] = total_debt / total_equity.replace(0, np.nan)
    feat["debt_to_assets"] = total_debt / total_assets.replace(0, np.nan)
    feat["leverage_ratio"] = total_assets / total_equity.replace(0, np.nan)
    feat["current_ratio"] = current_assets / current_liabilities.replace(0, np.nan)
    feat["quick_ratio"] = (
        (current_assets - inventory) / current_liabilities.replace(0, np.nan)
    )
    feat["cash_ratio"] = cash / current_liabilities.replace(0, np.nan)
    feat["asset_turnover"] = rev / total_assets.replace(0, np.nan)
    feat["receivables_turnover"] = rev / receivables.replace(0, np.nan)
    feat["inventory_turnover"] = (
        gcol(features_raw, "Cost Of Revenue", 0) / inventory.replace(0, np.nan)
    )
    feat["working_capital_ratio"] = (
        (current_assets - current_liabilities) / total_assets.replace(0, np.nan)
    )
    feat["cash_to_assets"] = cash / total_assets.replace(0, np.nan)
    feat["log_total_assets"] = np.log1p(total_assets.clip(lower=0))
    feat["log_equity"] = np.sign(total_equity) * np.log1p(total_equity.abs())
    return feat


def cashflow_features(features_raw, cf_raw, bs_raw=None) -> pd.DataFrame:
    feat = pd.DataFrame(index=features_raw.index)
    cf = cf_raw.reindex(features_raw.index)

    rev = gcol(features_raw, "Total Revenue", gcol(features_raw, "Operating Revenue"))
    ni = gcol(features_raw, "Net Income", 0)
    ebitda = gcol(features_raw, "EBITDA", 0)
    rev_safe = rev.replace(0, np.nan)

    if bs_raw is not None:
        total_assets = gcol(bs_raw.reindex(features_raw.index), "Total Assets", np.nan)
    else:
        total_assets = gcol(
            cf_raw.reindex(features_raw.index) if "Total Assets" in cf_raw.columns
            else pd.DataFrame(index=features_raw.index),
            "Total Assets",
            np.nan,
        )

    ocf = gcol(cf, "Cash Flow From Continuing Operating Activities",
               gcol(cf, "Operating Cash Flow", 0))
    capex = gcol(cf, "Capital Expenditure", 0).abs()
    fcf = ocf - capex
    dep = gcol(cf, "Depreciation And Amortization",
               gcol(cf, "Depreciation Amortization Depletion", 0))
    dividends = gcol(cf, "Cash Dividends Paid", 0).abs()
    repurchases = gcol(cf, "Repurchase Of Capital Stock",
                       gcol(cf, "Common Stock Issuance", 0)).abs()

    feat["ocf_to_revenue"] = ocf / rev_safe
    feat["fcf_to_revenue"] = fcf / rev_safe
    feat["fcf_to_assets"] = fcf / total_assets.replace(0, np.nan)
    feat["ocf_to_ni"] = ocf / ni.replace(0, np.nan)
    feat["capex_to_revenue"] = capex / rev_safe
    feat["dep_to_revenue"] = dep / rev_safe
    feat["dep_to_capex"] = dep / capex.replace(0, np.nan)
    feat["dividend_payout"] = dividends / ni.clip(lower=0).replace(0, np.nan)
    feat["cash_conversion"] = ocf / ebitda.replace(0, np.nan)
    return feat


def annual_growth_features(feat, annual_raw) -> pd.DataFrame:
    """Map year-over-year growth from annual financials onto quarterly dates."""
    if annual_raw is None or len(annual_raw) == 0:
        return feat

    ann_rev_g = gcol(annual_raw, "Total Revenue",
                     gcol(annual_raw, "Operating Revenue", np.nan)).groupby(level="symbol").pct_change(1)
    ann_ni_g = gcol(annual_raw, "Net Income", np.nan).groupby(level="symbol").pct_change(1)
    ann_ebitda_g = gcol(annual_raw, "EBITDA", np.nan).groupby(level="symbol").pct_change(1)

    ann_growth = pd.DataFrame({
        "annual_rev_growth": ann_rev_g,
        "annual_ni_growth": ann_ni_g,
        "annual_ebitda_growth": ann_ebitda_g,
    }).reset_index()
    ann_growth["date"] = pd.to_datetime(ann_growth["date"])
    ann_growth["date"] = ann_growth["date"] + pd.Timedelta(days=ANNUAL_FILING_LAG)

    feat_reset = feat.reset_index()
    feat_reset["date"] = pd.to_datetime(feat_reset["date"])

    merged = []
    for sym in feat_reset["symbol"].unique():
        fq = feat_reset[feat_reset["symbol"] == sym][["symbol", "date"]].sort_values("date")
        aq = ann_growth[ann_growth["symbol"] == sym].sort_values("date")
        if len(aq) == 0:
            continue
        m = pd.merge_asof(
            fq, aq.drop(columns="symbol"), on="date",
            direction="backward", tolerance=pd.Timedelta(days=400),
        )
        merged.append(m)

    if merged:
        mapped = pd.concat(merged).set_index(["symbol", "date"])
        for c in ["annual_rev_growth", "annual_ni_growth", "annual_ebitda_growth"]:
            if c in mapped.columns:
                feat[c] = mapped[c]
    return feat


def qoq_features(feat, features_raw) -> pd.DataFrame:
    """Quarter-over-quarter growth and margin changes."""
    rev = gcol(features_raw, "Total Revenue", gcol(features_raw, "Operating Revenue"))
    ni = gcol(features_raw, "Net Income", 0)
    ebitda = gcol(features_raw, "EBITDA", 0)
    gp = gcol(features_raw, "Gross Profit", 0)

    feat["revenue_growth_qoq"] = qoq_growth(rev)
    feat["ni_growth_qoq"] = qoq_growth(ni.replace(0, np.nan))
    feat["ebitda_growth_qoq"] = qoq_growth(ebitda.replace(0, np.nan))
    feat["gp_growth_qoq"] = qoq_growth(gp.replace(0, np.nan))
    feat["eps_growth_qoq"] = qoq_growth(
        gcol(features_raw, "Diluted EPS").replace(0, np.nan)
    )
    feat["gross_margin_chg"] = qoq_diff(feat["gross_margin"])
    feat["operating_margin_chg"] = qoq_diff(feat["operating_margin"])
    feat["net_margin_chg"] = qoq_diff(feat["net_margin"])
    feat["roe_chg"] = qoq_diff(feat["roe"])
    feat["roa_chg"] = qoq_diff(feat["roa"])
    feat["leverage_chg"] = qoq_diff(feat["debt_to_equity"])
    feat["current_ratio_chg"] = qoq_diff(feat["current_ratio"])
    feat["ocf_margin_chg"] = qoq_diff(feat["ocf_to_revenue"])
    feat["fcf_margin_chg"] = qoq_diff(feat["fcf_to_revenue"])
    feat["cash_conversion_chg"] = qoq_diff(feat["cash_conversion"])
    return feat


def momentum_features(features_raw, close_prices) -> pd.DataFrame:
    """Price momentum and short-term volatility."""
    records = []
    for symbol, q_date in features_raw.index.tolist():
        sp = close_prices[
            (close_prices["symbol"] == symbol) & (close_prices["date"] <= q_date)
        ].sort_values("date")
        if len(sp) < 5:
            records.append({"symbol": symbol, "date": q_date})
            continue
        last_px = sp["close"].iloc[-1]
        rec = {"symbol": symbol, "date": q_date}
        for label, days in [("momentum_1m", 35), ("momentum_3m", 95),
                            ("momentum_6m", 185), ("momentum_12m", 370)]:
            window = sp[sp["date"] >= q_date - pd.Timedelta(days=days)]
            if len(window) > 0:
                rec[label] = last_px / window["close"].iloc[0] - 1
        m1 = sp[sp["date"] >= q_date - pd.Timedelta(days=35)]
        if len(m1) > 5:
            rec["volatility_1m"] = m1["close"].pct_change().dropna().std()
        m3 = sp[sp["date"] >= q_date - pd.Timedelta(days=95)]
        if len(m3) > 10:
            rec["volatility_3m"] = m3["close"].pct_change().dropna().std()
        if rec.get("momentum_1m") and rec.get("volatility_1m", 0) > 0:
            rec["risk_adj_momentum"] = rec["momentum_1m"] / rec["volatility_1m"]
        if "momentum_12m" in rec and "momentum_1m" in rec:
            rec["momentum_12_1"] = rec["momentum_12m"] - rec["momentum_1m"]
        records.append(rec)
    return pd.DataFrame(records).set_index(["symbol", "date"])


def macro_features(features_raw, macro_df) -> pd.DataFrame:
    """Map daily macro indicators onto (symbol, date) index."""
    if macro_df is None or len(macro_df) == 0:
        return pd.DataFrame(index=features_raw.index)

    cache = {}
    records = []
    for symbol, q_date in features_raw.index:
        if q_date not in cache:
            lookback = macro_df[macro_df.index <= q_date]
            cache[q_date] = lookback.iloc[-1].to_dict() if len(lookback) > 0 else {}
        rec = {"symbol": symbol, "date": q_date}
        rec.update(cache[q_date])
        records.append(rec)
    return pd.DataFrame(records).set_index(["symbol", "date"])


def cross_sectional_ranks(feat: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks within each quarter."""
    rank_df = feat.groupby(level="date").rank(pct=True)
    rank_df.columns = [f"{c}_rank" for c in rank_df.columns]
    return rank_df


def clip_outliers(features_all: pd.DataFrame, raw_cols, n_std=5):
    """Clip features beyond ±n_std standard deviations within each quarter."""
    for c in raw_cols:
        if c in features_all.columns:
            grp = features_all[c].groupby(level="date")
            mu = grp.transform("mean")
            sigma = grp.transform("std")
            features_all[c] = features_all[c].clip(
                lower=mu - n_std * sigma, upper=mu + n_std * sigma
            )
    return features_all


# ── Risk features ──────────────────────────────────────────────────────────────


def risk_features(returns_full, close_prices) -> pd.DataFrame:
    """Historical vol, beta, drawdown, and higher moments (no lookahead)."""
    mkt_pivot = close_prices.pivot(index="date", columns="symbol", values="close")
    mkt_mean = mkt_pivot.mean(axis=1)
    mkt_daily_ret = mkt_mean.pct_change().dropna()

    records = []
    for sym, q_date in returns_full[["symbol", "date"]].values.tolist():
        buy_dt = q_date + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        lookback_start = buy_dt - pd.Timedelta(days=180)

        hist = close_prices[
            (close_prices["symbol"] == sym)
            & (close_prices["date"] >= lookback_start)
            & (close_prices["date"] < buy_dt)
        ].sort_values("date").set_index("date")

        rec = {"symbol": sym, "date": q_date}
        if len(hist) < 30:
            records.append(rec)
            continue

        dr = hist["close"].pct_change().dropna()
        rec["hist_vol_6m"] = dr.std() * np.sqrt(252)
        if len(dr) >= 63:
            rec["hist_vol_3m"] = dr.iloc[-63:].std() * np.sqrt(252)
        if len(dr) >= 21:
            rec["hist_vol_1m"] = dr.iloc[-21:].std() * np.sqrt(252)

        neg = dr[dr < 0]
        rec["hist_downside_vol"] = neg.std() * np.sqrt(252) if len(neg) > 5 else np.nan

        cum_p = hist["close"].values
        running_max = np.maximum.accumulate(cum_p)
        rec["hist_max_dd"] = (cum_p / running_max - 1).min()
        rec["hist_skew"] = dr.skew()
        rec["hist_kurtosis"] = dr.kurtosis()
        rec["hist_var5"] = dr.quantile(0.05)

        common = dr.index.intersection(mkt_daily_ret.index)
        if len(common) > 30:
            s_vals = dr.loc[common].values
            m_vals = mkt_daily_ret.loc[common].values
            cov_mat = np.cov(s_vals, m_vals)
            if cov_mat[1, 1] > 0:
                rec["hist_beta"] = cov_mat[0, 1] / cov_mat[1, 1]
                resid = s_vals - rec["hist_beta"] * m_vals
                rec["hist_idio_vol"] = np.std(resid) * np.sqrt(252)

        if len(dr) >= 42:
            rolling_vol = dr.rolling(21).std() * np.sqrt(252)
            rec["hist_vol_of_vol"] = rolling_vol.dropna().std()
        if len(dr) >= 63:
            recent = dr.iloc[-21:].std()
            older = dr.iloc[-63:-21].std()
            rec["vol_trend"] = (recent / older - 1) if older > 0 else np.nan
        rec["avg_abs_return"] = dr.abs().mean()
        records.append(rec)

    return pd.DataFrame(records).set_index(["symbol", "date"])


# ── Master feature builder ─────────────────────────────────────────────────────


def build_features(
    features_raw, bs_raw, cf_raw, annual_raw,
    close_prices, macro_df, returns_df, returns_full,
):
    """Assemble the full feature matrix + targets for modeling.

    Returns (risk_model_df, feature_cols_all).
    """
    # Fundamental features
    feat = profitability_features(features_raw)
    feat = feat.join(size_features(features_raw))
    feat = feat.join(balance_sheet_features(features_raw, bs_raw))
    feat = feat.join(cashflow_features(features_raw, cf_raw, bs_raw))
    feat = annual_growth_features(feat, annual_raw)
    feat = qoq_features(feat, features_raw)

    # Momentum
    mom = momentum_features(features_raw, close_prices)
    feat = feat.join(mom)

    # Macro
    mac = macro_features(features_raw, macro_df)
    feat = feat.join(mac)

    raw_cols = list(feat.columns)

    # Cross-sectional ranks
    ranks = cross_sectional_ranks(feat)
    features_all = pd.concat([feat, ranks], axis=1)
    features_all = features_all.replace([np.inf, -np.inf], np.nan)
    features_all = clip_outliers(features_all, raw_cols)

    # Add risk features
    risk_feat = risk_features(returns_full, close_prices)
    risk_model_df = features_all.join(risk_feat, how="inner")
    risk_rank = risk_feat.groupby(level="date").rank(pct=True)
    risk_rank.columns = [f"{c}_rank" for c in risk_rank.columns]
    risk_model_df = risk_model_df.join(risk_rank, how="left")

    # Add targets
    targets = returns_full.set_index(["symbol", "date"])[
        ["next_q_return", "return_quantile", "realized_vol",
         "realized_downside_vol", "realized_max_dd"]
    ]
    risk_model_df = risk_model_df.join(targets, how="inner")
    risk_model_df["vol_quantile"] = (
        risk_model_df.groupby(level="date")["realized_vol"].rank(pct=True)
    )
    risk_model_df["risk_adj_return"] = (
        risk_model_df["next_q_return"]
        / risk_model_df["realized_vol"].replace(0, np.nan)
    )

    target_cols = {
        "next_q_return", "return_quantile", "realized_vol",
        "realized_downside_vol", "realized_max_dd", "vol_quantile",
        "risk_adj_return",
    }
    feature_cols_all = [c for c in risk_model_df.columns if c not in target_cols]

    print(f"  Features: {len(feature_cols_all)} | Rows: {len(risk_model_df)}")
    return risk_model_df, feature_cols_all
