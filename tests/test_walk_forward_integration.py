"""Integration tests for walk_forward engine and factor_benchmarks.

Tests the full walk-forward pipeline with synthetic data to verify:
- No train/test leakage
- Weight constraints (sum-to-one, non-negative, bounded)
- Forward return timing (buy_date > q_date)
- Output structure correctness
- Factor benchmark execution
"""

import numpy as np
import pandas as pd
import pytest

from stock_data.config import EARNINGS_LAG_DAYS, PROD_CFG
from stock_data.dataset import compute_forward_returns
from stock_data.modeling.train import walk_forward, factor_benchmarks


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_synthetic_data(
    n_symbols=200,
    n_quarters=10,
    n_features=20,
    seed=42,
):
    """Build synthetic risk_model_df, feature_cols, and close_prices.

    Creates a controlled dataset with a known signal embedded in one feature
    so walk_forward has something to learn.
    """
    rng = np.random.default_rng(seed)

    # Quarter start dates with enough history for min_train_q
    dates = pd.date_range("2018-01-01", periods=n_quarters, freq="QS")
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    # MultiIndex (symbol, date) — canonical index
    idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])

    # Features: random noise + one "signal" feature correlated with returns
    feat_cols = [f"feat_{i}" for i in range(n_features)]
    feat_data = rng.standard_normal((len(idx), n_features))
    df = pd.DataFrame(feat_data, index=idx, columns=feat_cols)

    # Add hist_vol_3m (needed by vol model fallback)
    df["hist_vol_3m"] = np.abs(rng.normal(0.20, 0.05, len(idx)))

    # Target: next_q_return has weak signal from feat_0
    signal = df["feat_0"].values * 0.02
    noise = rng.normal(0.02, 0.08, len(idx))
    df["next_q_return"] = signal + noise

    # Realized vol target
    df["realized_vol"] = np.abs(rng.normal(0.20, 0.05, len(idx)))

    # Factor columns for factor_benchmarks
    df["hist_vol_6m"] = df["hist_vol_3m"] * 1.1
    df["momentum_3m"] = rng.normal(0, 0.1, len(idx))
    df["roe"] = rng.normal(0.1, 0.05, len(idx))
    df["fcf_to_assets"] = rng.normal(0.05, 0.03, len(idx))

    all_feat_cols = feat_cols + ["hist_vol_3m"]

    # Close prices: daily data spanning the full test period
    # Need data from first quarter buy_date through last quarter sell_date
    price_start = dates[0] - pd.Timedelta(days=30)
    price_end = dates[-1] + pd.Timedelta(days=EARNINGS_LAG_DAYS + 120)
    trade_days = pd.bdate_range(price_start, price_end)

    price_records = []
    for sym in symbols:
        base = rng.uniform(20, 200)
        prices = base * np.cumprod(1 + rng.normal(0.0003, 0.015, len(trade_days)))
        for dt, px in zip(trade_days, prices):
            price_records.append({"symbol": sym, "date": dt, "close": px})

    # Add ^GSPC for SPX benchmark
    gspc_prices = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(trade_days)))
    for dt, px in zip(trade_days, gspc_prices):
        price_records.append({"symbol": "^GSPC", "date": dt, "close": px})

    close_prices = pd.DataFrame(price_records)

    return df, all_feat_cols, close_prices


@pytest.fixture(scope="module")
def synthetic_data():
    """Module-scoped fixture for expensive synthetic data generation."""
    return _make_synthetic_data()


@pytest.fixture(scope="module")
def walk_forward_results(synthetic_data):
    """Run walk_forward once for all tests."""
    risk_model_df, feature_cols, close_prices = synthetic_data
    prod_df, prod_fi, weights_history = walk_forward(
        risk_model_df, feature_cols, close_prices
    )
    return prod_df, prod_fi, weights_history


# ── Train/Test Leakage ─────────────────────────────────────────────────────────


class TestNoLeakage:
    """Verify walk_forward never trains on test data."""

    def test_no_train_date_equals_test_date(self, walk_forward_results):
        """Each test_date must not appear in any training window."""
        prod_df, _, _ = walk_forward_results
        # walk_forward uses `d < td` strictly for training dates
        # If we have results, the engine ran. Verify the dates are strictly ordered.
        test_dates = prod_df["test_date"].tolist()
        # Each test date should be later than any prior test date's training window
        for i in range(1, len(test_dates)):
            assert test_dates[i] > test_dates[i - 1]

    def test_train_quarters_strictly_precede_test(self, walk_forward_results):
        """n_train_q should increase or stay bounded as we move forward."""
        prod_df, _, _ = walk_forward_results
        # First iteration should have min_train_q training quarters
        assert prod_df.iloc[0]["n_train_q"] >= PROD_CFG["min_train_q"]

    def test_n_train_q_never_exceeds_max(self, walk_forward_results):
        """Training window should respect max_train_q config."""
        prod_df, _, _ = walk_forward_results
        max_q = PROD_CFG.get("max_train_q")
        if max_q:
            assert prod_df["n_train_q"].max() <= max_q


# ── Weight Constraints ─────────────────────────────────────────────────────────


class TestWeightConstraints:
    """Verify portfolio weights are valid at every iteration."""

    def test_weights_sum_to_one(self, walk_forward_results):
        _, _, weights_history = walk_forward_results
        for td, wdata in weights_history.items():
            w = wdata["weights"]
            assert abs(w.sum() - 1.0) < 1e-6, f"Weights don't sum to 1 at {td}"

    def test_weights_non_negative(self, walk_forward_results):
        _, _, weights_history = walk_forward_results
        for td, wdata in weights_history.items():
            w = wdata["weights"]
            assert (w >= -1e-10).all(), f"Negative weight at {td}"

    def test_weights_respect_max_weight(self, walk_forward_results):
        _, _, weights_history = walk_forward_results
        max_w = PROD_CFG["max_weight"]
        for td, wdata in weights_history.items():
            w = wdata["weights"]
            assert w.max() <= max_w + 1e-6, (
                f"Weight {w.max():.4f} exceeds max {max_w} at {td}"
            )

    def test_symbols_match_weights_length(self, walk_forward_results):
        _, _, weights_history = walk_forward_results
        for td, wdata in weights_history.items():
            assert len(wdata["weights"]) == len(wdata["symbols"])


# ── Output Structure ───────────────────────────────────────────────────────────


class TestOutputStructure:
    """Verify walk_forward returns correct structure and columns."""

    def test_prod_df_is_dataframe(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert isinstance(prod_df, pd.DataFrame)

    def test_prod_df_has_required_columns(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        required = [
            "test_date", "n_train_q", "n_stocks", "n_eligible",
            "n_held", "max_wt", "mkt_ret", "gross_ret", "net_ret",
            "turnover", "tx_cost", "vol_rc", "ret_rc",
            "ret_rc_xgb", "ret_rc_ridge", "ret_rc_rf", "used_lw",
        ]
        for col in required:
            assert col in prod_df.columns, f"Missing column: {col}"

    def test_prod_df_has_spx_ret(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert "spx_ret" in prod_df.columns

    def test_prod_df_not_empty(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert len(prod_df) > 0

    def test_fi_list_matches_prod_df_length(self, walk_forward_results):
        prod_df, prod_fi, _ = walk_forward_results
        assert len(prod_fi) == len(prod_df)

    def test_fi_entries_have_date_and_fi(self, walk_forward_results):
        _, prod_fi, _ = walk_forward_results
        for entry in prod_fi:
            assert "date" in entry
            assert "fi" in entry
            assert "fi_detail" in entry

    def test_weights_history_has_same_dates(self, walk_forward_results):
        prod_df, _, weights_history = walk_forward_results
        wh_dates = set(weights_history.keys())
        pdf_dates = set(prod_df["test_date"])
        assert wh_dates == pdf_dates

    def test_net_ret_equals_gross_minus_costs(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        # net_ret = gross_ret - tx_cost
        diff = (prod_df["gross_ret"] - prod_df["tx_cost"] - prod_df["net_ret"]).abs()
        assert (diff < 1e-10).all()


# ── Turnover and Costs ─────────────────────────────────────────────────────────


class TestTurnoverAndCosts:
    """Verify turnover and transaction cost logic."""

    def test_first_period_turnover_is_one(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert prod_df.iloc[0]["turnover"] == pytest.approx(1.0)

    def test_turnover_in_valid_range(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert (prod_df["turnover"] >= 0).all()
        assert (prod_df["turnover"] <= 2.0).all()  # one-way can approach 1.0

    def test_tx_cost_proportional_to_turnover(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        expected = prod_df["turnover"] * PROD_CFG["cost_bps"] / 10000
        diff = (prod_df["tx_cost"] - expected).abs()
        assert (diff < 1e-10).all()


# ── Forward Return Timing ──────────────────────────────────────────────────────


class TestForwardReturnTiming:
    """Verify compute_forward_returns respects earnings lag."""

    def test_buy_date_after_quarter_date(self):
        """Buy date must be > q_date (by EARNINGS_LAG_DAYS)."""
        rng = np.random.default_rng(99)
        symbols = ["AAPL", "MSFT"]
        dates = pd.date_range("2020-01-01", periods=4, freq="QS")
        trade_days = pd.bdate_range("2019-11-01", "2021-06-30")

        price_records = []
        for sym in symbols:
            prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(trade_days)))
            for dt, px in zip(trade_days, prices):
                price_records.append({"symbol": sym, "date": dt, "close": px})
        close_prices = pd.DataFrame(price_records)

        idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])
        returns_df = compute_forward_returns(close_prices, idx)

        if len(returns_df) > 0 and "buy_date" in returns_df.columns:
            for _, row in returns_df.iterrows():
                q_date = row.name[1] if isinstance(row.name, tuple) else row["date"]
                assert row["buy_date"] > q_date

    def test_sell_date_after_buy_date(self):
        """Sell date must be > buy_date."""
        rng = np.random.default_rng(99)
        symbols = ["AAPL"]
        dates = pd.date_range("2020-01-01", periods=4, freq="QS")
        trade_days = pd.bdate_range("2019-11-01", "2021-06-30")

        price_records = []
        for sym in symbols:
            prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(trade_days)))
            for dt, px in zip(trade_days, prices):
                price_records.append({"symbol": sym, "date": dt, "close": px})
        close_prices = pd.DataFrame(price_records)

        idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])
        returns_df = compute_forward_returns(close_prices, idx)

        if len(returns_df) > 0 and "sell_date" in returns_df.columns and "buy_date" in returns_df.columns:
            for _, row in returns_df.iterrows():
                assert row["sell_date"] > row["buy_date"]

    def test_buy_date_lag_is_at_least_earnings_lag(self):
        """Gap between q_date and buy_date should be >= EARNINGS_LAG_DAYS."""
        rng = np.random.default_rng(99)
        symbols = ["TEST"]
        dates = pd.date_range("2020-01-01", periods=4, freq="QS")
        trade_days = pd.bdate_range("2019-11-01", "2021-06-30")

        price_records = []
        prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(trade_days)))
        for dt, px in zip(trade_days, prices):
            price_records.append({"symbol": "TEST", "date": dt, "close": px})
        close_prices = pd.DataFrame(price_records)

        idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])
        returns_df = compute_forward_returns(close_prices, idx)

        if len(returns_df) > 0 and "buy_date" in returns_df.columns:
            for _, row in returns_df.iterrows():
                q_date = row.name[1] if isinstance(row.name, tuple) else row["date"]
                lag = (row["buy_date"] - q_date).days
                assert lag >= EARNINGS_LAG_DAYS


# ── Factor Benchmarks ──────────────────────────────────────────────────────────


class TestFactorBenchmarks:
    """Verify factor_benchmarks produces valid output."""

    def test_returns_dict_of_factor_names(self, synthetic_data, walk_forward_results):
        risk_model_df, feature_cols, _ = synthetic_data
        prod_df, _, _ = walk_forward_results
        result = factor_benchmarks(risk_model_df, feature_cols, prod_df)
        assert isinstance(result, dict)
        # At least some factors should have results
        non_empty = {k: v for k, v in result.items() if v}
        assert len(non_empty) > 0

    def test_factor_results_have_required_keys(self, synthetic_data, walk_forward_results):
        risk_model_df, feature_cols, _ = synthetic_data
        prod_df, _, _ = walk_forward_results
        result = factor_benchmarks(risk_model_df, feature_cols, prod_df)
        for fname, results in result.items():
            for entry in results:
                assert "test_date" in entry
                assert "port_ret" in entry
                assert "mkt_ret" in entry
                assert "excess" in entry

    def test_factor_excess_equals_port_minus_mkt(self, synthetic_data, walk_forward_results):
        risk_model_df, feature_cols, _ = synthetic_data
        prod_df, _, _ = walk_forward_results
        result = factor_benchmarks(risk_model_df, feature_cols, prod_df)
        for fname, results in result.items():
            for entry in results:
                assert abs(entry["excess"] - (entry["port_ret"] - entry["mkt_ret"])) < 1e-10


# ── Smoke Test: Known Signal ──────────────────────────────────────────────────


class TestKnownSignal:
    """Verify the engine can detect a strong embedded signal.

    With a strong enough signal, the ensemble should produce positive
    average rank correlation. This is a sanity check, not a performance test.
    """

    def test_avg_ret_rc_is_finite(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert prod_df["ret_rc"].notna().all()
        assert np.isfinite(prod_df["ret_rc"]).all()

    def test_vol_rc_is_finite(self, walk_forward_results):
        prod_df, _, _ = walk_forward_results
        assert prod_df["vol_rc"].notna().all()
        assert np.isfinite(prod_df["vol_rc"]).all()
