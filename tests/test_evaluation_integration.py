"""Integration tests for the evaluation pipeline.

Tests simulate_portfolio and evaluate_factors end-to-end with
synthetic data produced by walk_forward.
"""

import numpy as np
import pandas as pd
import pytest

from stock_data.config import EARNINGS_LAG_DAYS, PROD_CFG
from stock_data.evaluation import evaluate_factors, simulate_portfolio
from stock_data.modeling.train import walk_forward, factor_benchmarks


# ── Fixtures (reuse same synthetic data pattern from walk_forward tests) ───────


def _make_synthetic_data(n_symbols=200, n_quarters=10, n_features=20, seed=42):
    """Same as test_walk_forward_integration but duplicated for isolation."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_quarters, freq="QS")
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])

    feat_cols = [f"feat_{i}" for i in range(n_features)]
    feat_data = rng.standard_normal((len(idx), n_features))
    df = pd.DataFrame(feat_data, index=idx, columns=feat_cols)
    df["hist_vol_3m"] = np.abs(rng.normal(0.20, 0.05, len(idx)))

    signal = df["feat_0"].values * 0.02
    noise = rng.normal(0.02, 0.08, len(idx))
    df["next_q_return"] = signal + noise
    df["realized_vol"] = np.abs(rng.normal(0.20, 0.05, len(idx)))
    df["hist_vol_6m"] = df["hist_vol_3m"] * 1.1
    df["momentum_3m"] = rng.normal(0, 0.1, len(idx))
    df["roe"] = rng.normal(0.1, 0.05, len(idx))
    df["fcf_to_assets"] = rng.normal(0.05, 0.03, len(idx))

    all_feat_cols = feat_cols + ["hist_vol_3m"]

    price_start = dates[0] - pd.Timedelta(days=30)
    price_end = dates[-1] + pd.Timedelta(days=EARNINGS_LAG_DAYS + 120)
    trade_days = pd.bdate_range(price_start, price_end)

    price_records = []
    for sym in symbols:
        base = rng.uniform(20, 200)
        prices = base * np.cumprod(1 + rng.normal(0.0003, 0.015, len(trade_days)))
        for dt, px in zip(trade_days, prices):
            price_records.append({"symbol": sym, "date": dt, "close": px})
    gspc_prices = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(trade_days)))
    for dt, px in zip(trade_days, gspc_prices):
        price_records.append({"symbol": "^GSPC", "date": dt, "close": px})
    close_prices = pd.DataFrame(price_records)

    return df, all_feat_cols, close_prices


@pytest.fixture(scope="module")
def full_pipeline_results():
    """Run walk_forward + factor_benchmarks once for all eval tests."""
    risk_model_df, feature_cols, close_prices = _make_synthetic_data()
    prod_df, prod_fi, weights_history = walk_forward(
        risk_model_df, feature_cols, close_prices
    )
    factor_results = factor_benchmarks(risk_model_df, feature_cols, prod_df)
    return prod_df, prod_fi, weights_history, factor_results, risk_model_df, close_prices


# ── evaluate_factors integration ───────────────────────────────────────────────


class TestEvaluateFactorsIntegration:
    """Verify evaluate_factors runs end-to-end with real walk_forward output."""

    def test_returns_four_tuple(self, full_pipeline_results, capsys):
        prod_df, _, _, factor_results, _, _ = full_pipeline_results
        result = evaluate_factors(prod_df, factor_results, n_boot=200)
        assert len(result) == 4

    def test_ci_lo_le_ci_hi(self, full_pipeline_results, capsys):
        prod_df, _, _, factor_results, _, _ = full_pipeline_results
        ci_lo, ci_hi, _, _ = evaluate_factors(prod_df, factor_results, n_boot=200)
        assert ci_lo <= ci_hi

    def test_boot_means_length(self, full_pipeline_results, capsys):
        prod_df, _, _, factor_results, _, _ = full_pipeline_results
        _, _, boot_means, _ = evaluate_factors(prod_df, factor_results, n_boot=200)
        assert len(boot_means) == 200

    def test_excess_returns_length_matches_prod_df(self, full_pipeline_results, capsys):
        prod_df, _, _, factor_results, _, _ = full_pipeline_results
        _, _, _, ex_n = evaluate_factors(prod_df, factor_results, n_boot=200)
        assert len(ex_n) == len(prod_df)


# ── simulate_portfolio integration ─────────────────────────────────────────────


class TestSimulatePortfolioIntegration:
    """Verify simulate_portfolio runs end-to-end with real walk_forward output."""

    def test_returns_three_dataframes(self, full_pipeline_results):
        prod_df, _, weights_history, _, risk_model_df, close_prices = full_pipeline_results
        sim_df, mkt_sim, qlog = simulate_portfolio(
            prod_df, weights_history, risk_model_df, close_prices,
            initial_capital=1_000_000,
        )
        assert isinstance(sim_df, pd.DataFrame)
        assert isinstance(mkt_sim, pd.DataFrame)
        assert isinstance(qlog, pd.DataFrame)

    def test_sim_df_has_portfolio_value(self, full_pipeline_results):
        prod_df, _, weights_history, _, risk_model_df, close_prices = full_pipeline_results
        sim_df, _, _ = simulate_portfolio(
            prod_df, weights_history, risk_model_df, close_prices,
            initial_capital=1_000_000,
        )
        if len(sim_df) > 0:
            assert "portfolio_value" in sim_df.columns
            assert "daily_return" in sim_df.columns

    def test_portfolio_value_always_positive(self, full_pipeline_results):
        prod_df, _, weights_history, _, risk_model_df, close_prices = full_pipeline_results
        sim_df, _, _ = simulate_portfolio(
            prod_df, weights_history, risk_model_df, close_prices,
            initial_capital=1_000_000,
        )
        if len(sim_df) > 0:
            assert (sim_df["portfolio_value"] > 0).all()

    def test_qlog_returns_are_finite(self, full_pipeline_results):
        prod_df, _, weights_history, _, risk_model_df, close_prices = full_pipeline_results
        _, _, qlog = simulate_portfolio(
            prod_df, weights_history, risk_model_df, close_prices,
            initial_capital=1_000_000,
        )
        if len(qlog) > 0:
            assert qlog["sim_return"].notna().all()
            assert np.isfinite(qlog["sim_return"]).all()

    def test_qlog_end_value_gt_zero(self, full_pipeline_results):
        prod_df, _, weights_history, _, risk_model_df, close_prices = full_pipeline_results
        _, _, qlog = simulate_portfolio(
            prod_df, weights_history, risk_model_df, close_prices,
            initial_capital=1_000_000,
        )
        if len(qlog) > 0:
            assert (qlog["end_value"] > 0).all()

    def test_market_sim_tracks_equal_weight(self, full_pipeline_results):
        prod_df, _, weights_history, _, risk_model_df, close_prices = full_pipeline_results
        _, mkt_sim, _ = simulate_portfolio(
            prod_df, weights_history, risk_model_df, close_prices,
            initial_capital=1_000_000,
        )
        if len(mkt_sim) > 0:
            assert "market_value" in mkt_sim.columns
            assert (mkt_sim["market_value"] > 0).all()
