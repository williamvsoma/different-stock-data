"""Unit tests for stock_data.features."""

import numpy as np
import pandas as pd
import pytest

from stock_data.features import (
    cashflow_features,
    cross_sectional_ranks,
    gcol,
    profitability_features,
    qoq_diff,
    qoq_growth,
    size_features,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_index(symbols, dates):
    """Build a MultiIndex[(symbol, date)] from lists."""
    tuples = [(s, d) for s in symbols for d in dates]
    return pd.MultiIndex.from_tuples(tuples, names=["symbol", "date"])


def _basic_income_df(n_symbols=2, n_dates=3):
    """Return a minimal income-statement DataFrame with required columns."""
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="QE")
    idx = _make_index(symbols, dates)
    rng = np.random.default_rng(0)
    size = len(idx)
    return pd.DataFrame(
        {
            "Total Revenue": rng.integers(100, 1000, size).astype(float),
            "Net Income": rng.integers(-50, 200, size).astype(float),
            "EBIT": rng.integers(10, 300, size).astype(float),
            "EBITDA": rng.integers(20, 350, size).astype(float),
            "Gross Profit": rng.integers(40, 500, size).astype(float),
            "Tax Provision": rng.integers(5, 50, size).astype(float),
            "Pretax Income": rng.integers(10, 250, size).astype(float),
            "Interest Expense": rng.integers(1, 30, size).astype(float),
        },
        index=idx,
    )


# ── gcol ───────────────────────────────────────────────────────────────────────


class TestGcol:
    def test_present_column_returns_series(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = gcol(df, "a")
        pd.testing.assert_series_equal(result, df["a"])

    def test_absent_column_returns_scalar_default(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = gcol(df, "b", default=99.0)
        assert (result == 99.0).all()
        assert len(result) == len(df)

    def test_absent_column_default_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = gcol(df, "missing")
        assert result.isna().all()

    def test_absent_column_series_default(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        default_series = pd.Series([7, 8, 9], index=df.index)
        result = gcol(df, "nope", default=default_series)
        pd.testing.assert_series_equal(result, default_series)


# ── profitability_features ─────────────────────────────────────────────────────


class TestProfitabilityFeatures:
    def test_output_columns_present(self):
        df = _basic_income_df()
        feat = profitability_features(df)
        expected = {
            "gross_margin", "operating_margin", "net_margin",
            "ebitda_margin", "tax_rate", "interest_coverage",
            "rd_intensity", "sga_intensity", "cost_efficiency",
        }
        assert expected.issubset(set(feat.columns))

    def test_no_inf_values(self):
        df = _basic_income_df()
        feat = profitability_features(df)
        assert not np.isinf(feat.values).any(), "profitability_features must not contain inf"

    def test_zero_revenue_guard_no_inf(self):
        """When revenue is zero, ratios should be NaN, never ±inf."""
        df = _basic_income_df()
        df["Total Revenue"] = 0.0
        feat = profitability_features(df)
        assert not np.isinf(feat.values).any()
        # With zero revenue, margin columns should be all-NaN
        assert feat["gross_margin"].isna().all()
        assert feat["net_margin"].isna().all()

    def test_margins_bounded_when_positive_revenue(self):
        """Gross margin should be in (-inf, 1] when revenue > gross profit >= 0."""
        df = _basic_income_df()
        # Ensure GP <= Rev so gross_margin <= 1
        df["Gross Profit"] = df["Total Revenue"] * 0.4
        feat = profitability_features(df)
        valid = feat["gross_margin"].dropna()
        assert (valid <= 1.0 + 1e-9).all()

    def test_output_index_matches_input(self):
        df = _basic_income_df()
        feat = profitability_features(df)
        assert feat.index.equals(df.index)


# ── size_features ──────────────────────────────────────────────────────────────


class TestSizeFeatures:
    def test_output_columns_present(self):
        df = _basic_income_df()
        feat = size_features(df)
        assert {"log_revenue", "log_net_income", "log_ebitda", "diluted_eps"}.issubset(
            set(feat.columns)
        )

    def test_log_revenue_non_negative(self):
        df = _basic_income_df()
        feat = size_features(df)
        assert (feat["log_revenue"].dropna() >= 0).all()

    def test_log_net_income_preserves_sign_positive(self):
        """log_net_income should be positive when net income is positive."""
        df = _basic_income_df()
        df["Net Income"] = 500.0
        feat = size_features(df)
        assert (feat["log_net_income"] > 0).all()

    def test_log_net_income_preserves_sign_negative(self):
        """log_net_income should be negative when net income is negative."""
        df = _basic_income_df()
        df["Net Income"] = -500.0
        feat = size_features(df)
        assert (feat["log_net_income"] < 0).all()

    def test_log_net_income_zero_gives_zero(self):
        df = _basic_income_df()
        df["Net Income"] = 0.0
        feat = size_features(df)
        assert (feat["log_net_income"] == 0.0).all()

    def test_output_index_matches_input(self):
        df = _basic_income_df()
        feat = size_features(df)
        assert feat.index.equals(df.index)


# ── cashflow_features ──────────────────────────────────────────────────────────


class TestCashflowFeatures:
    def _make_cf_df(self, income_df):
        rng = np.random.default_rng(1)
        size = len(income_df)
        return pd.DataFrame(
            {
                "Cash Flow From Continuing Operating Activities": rng.integers(
                    10, 200, size
                ).astype(float),
                "Capital Expenditure": rng.integers(1, 80, size).astype(float),
                "Depreciation And Amortization": rng.integers(5, 60, size).astype(float),
                "Cash Dividends Paid": rng.integers(0, 30, size).astype(float),
            },
            index=income_df.index,
        )

    def test_output_columns_present(self):
        inc = _basic_income_df()
        cf = self._make_cf_df(inc)
        feat = cashflow_features(inc, cf)
        assert {"ocf_to_revenue", "fcf_to_revenue"}.issubset(set(feat.columns))

    def test_no_inf_values(self):
        inc = _basic_income_df()
        cf = self._make_cf_df(inc)
        feat = cashflow_features(inc, cf)
        assert not np.isinf(feat.values).any()

    def test_zero_revenue_gives_nan_not_inf(self):
        inc = _basic_income_df()
        inc["Total Revenue"] = 0.0
        cf = self._make_cf_df(inc)
        feat = cashflow_features(inc, cf)
        assert not np.isinf(feat.values).any()
        assert feat["ocf_to_revenue"].isna().all()

    def test_output_index_matches_input(self):
        inc = _basic_income_df()
        cf = self._make_cf_df(inc)
        feat = cashflow_features(inc, cf)
        assert feat.index.equals(inc.index)


# ── cross_sectional_ranks ──────────────────────────────────────────────────────


class TestCrossSectionalRanks:
    def _make_feat_df(self):
        symbols = ["A", "B", "C", "D"]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        rng = np.random.default_rng(2)
        return pd.DataFrame(
            {
                "gross_margin": rng.standard_normal(len(idx)),
                "net_margin": rng.standard_normal(len(idx)),
            },
            index=idx,
        )

    def test_output_shape_matches_input(self):
        feat = self._make_feat_df()
        ranks = cross_sectional_ranks(feat)
        assert ranks.shape == feat.shape

    def test_column_names_end_with_rank(self):
        feat = self._make_feat_df()
        ranks = cross_sectional_ranks(feat)
        assert all(c.endswith("_rank") for c in ranks.columns)

    def test_rank_values_in_zero_one(self):
        feat = self._make_feat_df()
        ranks = cross_sectional_ranks(feat)
        vals = ranks.values.flatten()
        valid = vals[~np.isnan(vals)]
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_output_index_matches_input(self):
        feat = self._make_feat_df()
        ranks = cross_sectional_ranks(feat)
        assert ranks.index.equals(feat.index)


# ── qoq_growth / qoq_diff ─────────────────────────────────────────────────────


class TestQoqGrowth:
    def _make_series(self):
        symbols = ["X", "Y"]
        dates = pd.date_range("2022-01-01", periods=4, freq="QE")
        idx = _make_index(symbols, dates)
        values = [100.0, 110.0, 121.0, 133.1] * 2
        return pd.Series(values, index=idx, name="rev")

    def test_first_period_is_nan(self):
        s = self._make_series()
        g = qoq_growth(s)
        # First date for each symbol should be NaN
        first_dates = s.groupby(level="symbol").apply(lambda x: x.index[0])
        for sym, first_idx in first_dates.items():
            assert pd.isna(g.loc[first_idx])

    def test_second_period_correct_growth(self):
        """Growth from 100 → 110 should be ~10%."""
        symbols = ["X"]
        dates = pd.date_range("2022-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        s = pd.Series([100.0, 110.0], index=idx)
        g = qoq_growth(s)
        assert abs(g.iloc[1] - 0.10) < 1e-9

    def test_negative_base_growth(self):
        """pct_change from -100 → -50 is (-50 - (-100)) / (-100) = -0.5."""
        symbols = ["X"]
        dates = pd.date_range("2022-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        s = pd.Series([-100.0, -50.0], index=idx)
        g = qoq_growth(s)
        assert abs(g.iloc[1] - (-0.50)) < 1e-9


class TestQoqDiff:
    def test_first_period_is_nan(self):
        symbols = ["A"]
        dates = pd.date_range("2022-01-01", periods=3, freq="QE")
        idx = _make_index(symbols, dates)
        s = pd.Series([10.0, 12.0, 15.0], index=idx)
        d = qoq_diff(s)
        assert pd.isna(d.iloc[0])

    def test_second_period_correct_diff(self):
        """Diff from 10 → 12 should be 2."""
        symbols = ["A"]
        dates = pd.date_range("2022-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        s = pd.Series([10.0, 12.0], index=idx)
        d = qoq_diff(s)
        assert d.iloc[1] == pytest.approx(2.0)

    def test_diff_across_symbols_independent(self):
        """Each symbol's diff should be computed independently."""
        symbols = ["A", "B"]
        dates = pd.date_range("2022-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        # A: 10→20 (diff=10), B: 5→15 (diff=10)
        s = pd.Series([10.0, 20.0, 5.0, 15.0], index=idx)
        d = qoq_diff(s)
        assert d.loc["A"].iloc[1] == pytest.approx(10.0)
        assert d.loc["B"].iloc[1] == pytest.approx(10.0)
