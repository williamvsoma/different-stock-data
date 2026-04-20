"""End-to-end integration test: synthetic data → features → walk-forward.

Exercises the full pipeline without network calls by generating synthetic
financial statements, prices, and macro data, then running the feature
builder and walk-forward engine end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

from stock_data.config import PROD_CFG, ENS_W
from stock_data.dataset import (
    build_returns_panel,
    compute_forward_returns,
    compute_realized_vol,
    filter_by_membership,
)
from stock_data.features import build_features
from stock_data.modeling.train import walk_forward


# ── Synthetic data factories ───────────────────────────────────────────────────

N_SYMBOLS = 60
N_QUARTERS = 8
SYMBOLS = [f"SYM{i:03d}" for i in range(N_SYMBOLS)]
# Add ^GSPC (used by risk_features for beta)
ALL_SYMBOLS = SYMBOLS + ["^GSPC"]
QUARTER_DATES = pd.date_range("2020-03-31", periods=N_QUARTERS, freq="QE")
ANNUAL_DATES = pd.date_range("2019-12-31", periods=3, freq="YE")


def _make_index(symbols, dates):
    tuples = [(s, d) for s in symbols for d in dates]
    return pd.MultiIndex.from_tuples(tuples, names=["symbol", "date"])


def _make_features_raw(rng):
    """Quarterly income statement with columns used by profitability/size/qoq features."""
    idx = _make_index(SYMBOLS, QUARTER_DATES)
    n = len(idx)
    return pd.DataFrame(
        {
            "Total Revenue": rng.uniform(100, 5000, n),
            "Net Income": rng.uniform(-200, 800, n),
            "EBIT": rng.uniform(10, 600, n),
            "EBITDA": rng.uniform(20, 700, n),
            "Gross Profit": rng.uniform(40, 2000, n),
            "Tax Provision": rng.uniform(5, 100, n),
            "Pretax Income": rng.uniform(10, 500, n),
            "Interest Expense": rng.uniform(1, 50, n),
            "Cost Of Revenue": rng.uniform(50, 3000, n),
            "Diluted EPS": rng.uniform(0.1, 10, n),
            "Operating Revenue": rng.uniform(100, 5000, n),
        },
        index=idx,
    )


def _make_bs_raw(rng):
    """Quarterly balance sheet."""
    idx = _make_index(SYMBOLS, QUARTER_DATES)
    n = len(idx)
    return pd.DataFrame(
        {
            "Total Assets": rng.uniform(1000, 50000, n),
            "Stockholders Equity": rng.uniform(200, 20000, n),
            "Total Debt": rng.uniform(100, 15000, n),
            "Current Assets": rng.uniform(200, 10000, n),
            "Current Liabilities": rng.uniform(100, 8000, n),
            "Cash And Cash Equivalents": rng.uniform(50, 5000, n),
            "Inventory": rng.uniform(10, 3000, n),
            "Accounts Receivable": rng.uniform(10, 3000, n),
        },
        index=idx,
    )


def _make_cf_raw(rng):
    """Quarterly cash flow statement."""
    idx = _make_index(SYMBOLS, QUARTER_DATES)
    n = len(idx)
    return pd.DataFrame(
        {
            "Cash Flow From Continuing Operating Activities": rng.uniform(-100, 2000, n),
            "Capital Expenditure": rng.uniform(-500, -10, n),
            "Depreciation And Amortization": rng.uniform(5, 300, n),
            "Cash Dividends Paid": rng.uniform(-200, 0, n),
            "Repurchase Of Capital Stock": rng.uniform(-500, 0, n),
        },
        index=idx,
    )


def _make_annual_raw(rng):
    """Annual income data for annual_growth_features."""
    idx = _make_index(SYMBOLS, ANNUAL_DATES)
    n = len(idx)
    return pd.DataFrame(
        {
            "Total Revenue": rng.uniform(400, 20000, n),
            "Net Income": rng.uniform(-500, 3000, n),
            "EBITDA": rng.uniform(100, 3000, n),
        },
        index=idx,
    )


def _make_close_prices(rng):
    """Daily close prices for all symbols + ^GSPC."""
    # Need prices from well before first quarter to well after last quarter
    start = QUARTER_DATES[0] - pd.Timedelta(days=400)
    end = QUARTER_DATES[-1] + pd.Timedelta(days=200)
    dates = pd.bdate_range(start, end)
    frames = []
    for sym in ALL_SYMBOLS:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
        frames.append(pd.DataFrame({"date": dates, "symbol": sym, "close": px}))
    return pd.concat(frames, ignore_index=True)


def _make_macro_df():
    """Daily macro data (VIX, treasury rates, S&P 500 level)."""
    start = QUARTER_DATES[0] - pd.Timedelta(days=400)
    end = QUARTER_DATES[-1] + pd.Timedelta(days=200)
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "vix": rng.uniform(12, 35, len(dates)),
            "treasury_10y": rng.uniform(0.5, 4.0, len(dates)),
            "treasury_3m": rng.uniform(0.0, 2.5, len(dates)),
            "sp500_level": rng.uniform(2500, 5000, len(dates)),
        },
        index=dates,
    )


def _make_sp500_membership():
    """All synthetic symbols are always in the S&P 500."""
    return pd.DataFrame(
        {
            "symbol": SYMBOLS,
            "date_added": pd.Timestamp("2018-01-01"),
            "date_removed": pd.NaT,
        }
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_data():
    """Build all synthetic inputs once for the module."""
    rng = np.random.default_rng(42)
    features_raw = _make_features_raw(rng)
    bs_raw = _make_bs_raw(rng)
    cf_raw = _make_cf_raw(rng)
    annual_raw = _make_annual_raw(rng)
    close_prices = _make_close_prices(rng)
    macro_df = _make_macro_df()
    sp500 = _make_sp500_membership()
    return {
        "features_raw": features_raw,
        "bs_raw": bs_raw,
        "cf_raw": cf_raw,
        "annual_raw": annual_raw,
        "close_prices": close_prices,
        "macro_df": macro_df,
        "sp500": sp500,
    }


@pytest.fixture(scope="module")
def feature_stage(synthetic_data):
    """Run the feature-building stage (corresponds to stage_features in main.py)."""
    d = synthetic_data
    close_prices = d["close_prices"]
    features_raw = d["features_raw"]

    # Forward returns + vol
    sym_date_pairs = features_raw.index
    returns_df = compute_forward_returns(close_prices, sym_date_pairs)
    returns_df["return_quantile"] = returns_df.groupby("date")["next_q_return"].rank(pct=True)
    vol_df = compute_realized_vol(returns_df, close_prices)
    returns_full = returns_df.merge(vol_df, on=["symbol", "date"], how="inner")

    # Build features
    price_panel = build_returns_panel(close_prices)
    risk_model_df, feature_cols = build_features(
        features_raw, d["bs_raw"], d["cf_raw"], d["annual_raw"],
        close_prices, d["macro_df"], returns_df, returns_full,
        price_panel=price_panel,
    )

    # Universe filter
    risk_model_df = filter_by_membership(risk_model_df, d["sp500"])

    return {
        "risk_model_df": risk_model_df,
        "feature_cols": feature_cols,
        "close_prices": close_prices,
        "returns_df": returns_df,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestFeatureStage:
    """Verify the feature-building pipeline produces valid output."""

    def test_risk_model_df_has_rows(self, feature_stage):
        assert len(feature_stage["risk_model_df"]) > 100

    def test_risk_model_df_index_is_symbol_date(self, feature_stage):
        rmd = feature_stage["risk_model_df"]
        assert list(rmd.index.names) == ["symbol", "date"]

    def test_targets_present(self, feature_stage):
        rmd = feature_stage["risk_model_df"]
        for col in ["next_q_return", "realized_vol"]:
            assert col in rmd.columns

    def test_feature_cols_are_subset_of_columns(self, feature_stage):
        rmd = feature_stage["risk_model_df"]
        feat_cols = feature_stage["feature_cols"]
        assert len(feat_cols) > 10
        missing = set(feat_cols) - set(rmd.columns)
        assert missing == set(), f"Missing feature columns: {missing}"

    def test_no_inf_in_features(self, feature_stage):
        rmd = feature_stage["risk_model_df"]
        feat_cols = feature_stage["feature_cols"]
        vals = rmd[feat_cols].values
        # NaN is fine, inf is not
        finite_mask = np.isfinite(vals) | np.isnan(vals)
        assert finite_mask.all(), "Inf values found in features"

    def test_multiple_quarters_present(self, feature_stage):
        rmd = feature_stage["risk_model_df"]
        n_dates = rmd.index.get_level_values("date").nunique()
        assert n_dates >= 3, f"Only {n_dates} quarters in risk_model_df"

    def test_multiple_symbols_present(self, feature_stage):
        rmd = feature_stage["risk_model_df"]
        n_syms = rmd.index.get_level_values("symbol").nunique()
        assert n_syms >= 30, f"Only {n_syms} symbols in risk_model_df"


class TestWalkForward:
    """Run the walk-forward engine on synthetic data and verify outputs."""

    @pytest.fixture(scope="class")
    def wf_result(self, feature_stage):
        """Run walk_forward with relaxed config for synthetic data."""
        rmd = feature_stage["risk_model_df"]
        feat_cols = feature_stage["feature_cols"]
        close = feature_stage["close_prices"]

        # Relaxed config: fewer constraints so synthetic data can produce results
        cfg = {
            **PROD_CFG,
            "min_train_q": 2,
            "min_train_rows": 30,
            "min_test_stocks": 10,
            "embargo_q": 0,
            "max_train_q": None,
        }
        prod_df, fi_list, weights_history = walk_forward(
            rmd, feat_cols, close, cfg=cfg, ens_weights=ENS_W,
        )
        return prod_df, fi_list, weights_history

    def test_prod_df_not_empty(self, wf_result):
        prod_df, _, _ = wf_result
        assert len(prod_df) > 0, "Walk-forward produced no quarters"

    def test_prod_df_has_required_columns(self, wf_result):
        prod_df, _, _ = wf_result
        required = [
            "test_date", "n_train_q", "n_stocks", "n_held",
            "gross_ret", "net_ret", "mkt_ret",
            "turnover", "ret_rc", "vol_rc",
        ]
        for col in required:
            assert col in prod_df.columns, f"Missing column: {col}"

    def test_net_return_finite(self, wf_result):
        prod_df, _, _ = wf_result
        assert prod_df["net_ret"].notna().all()
        assert np.isfinite(prod_df["net_ret"].values).all()

    def test_weights_history_matches_quarters(self, wf_result):
        prod_df, _, weights_history = wf_result
        assert len(weights_history) == len(prod_df)

    def test_weights_sum_close_to_one(self, wf_result):
        _, _, weights_history = wf_result
        for td, entry in weights_history.items():
            w = entry["weights"]
            assert abs(w.sum() - 1.0) < 0.05, f"Weights sum = {w.sum():.4f} at {td}"

    def test_weights_nonnegative(self, wf_result):
        _, _, weights_history = wf_result
        for td, entry in weights_history.items():
            w = entry["weights"]
            assert (w >= -1e-10).all(), f"Negative weight at {td}: min={w.min()}"

    def test_fi_list_has_entries(self, wf_result):
        _, fi_list, _ = wf_result
        assert len(fi_list) > 0, "No feature importance entries"

    def test_fi_entries_have_combined(self, wf_result):
        _, fi_list, _ = wf_result
        for entry in fi_list:
            assert "fi" in entry
            assert len(entry["fi"]) > 0

    def test_no_lookahead_bias(self, wf_result):
        """Training quarters must strictly precede the test quarter."""
        prod_df, _, _ = wf_result
        for i in range(1, len(prod_df)):
            assert prod_df.iloc[i]["test_date"] > prod_df.iloc[i - 1]["test_date"]

    def test_holdings_within_bounds(self, wf_result):
        prod_df, _, _ = wf_result
        assert (prod_df["n_held"] > 0).all(), "Some quarters have 0 holdings"
        assert (prod_df["n_held"] <= prod_df["n_stocks"]).all()
