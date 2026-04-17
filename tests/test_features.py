"""Unit tests for stock_data.features."""

import numpy as np
import pandas as pd
import pytest

from stock_data.features import (
    annual_growth_features,
    balance_sheet_features,
    cashflow_features,
    clip_outliers,
    cross_sectional_ranks,
    gcol,
    macro_features,
    momentum_features,
    profitability_features,
    qoq_diff,
    qoq_features,
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


# ── balance_sheet helpers ──────────────────────────────────────────────────────


def _basic_bs_df(idx):
    """Return a minimal balance-sheet DataFrame."""
    rng = np.random.default_rng(3)
    size = len(idx)
    return pd.DataFrame(
        {
            "Total Assets": rng.integers(500, 5000, size).astype(float),
            "Stockholders Equity": rng.integers(100, 2000, size).astype(float),
            "Total Debt": rng.integers(50, 1000, size).astype(float),
            "Current Assets": rng.integers(100, 1500, size).astype(float),
            "Current Liabilities": rng.integers(50, 800, size).astype(float),
            "Cash And Cash Equivalents": rng.integers(20, 500, size).astype(float),
            "Inventory": rng.integers(10, 200, size).astype(float),
            "Accounts Receivable": rng.integers(10, 300, size).astype(float),
        },
        index=idx,
    )


# ── balance_sheet_features ─────────────────────────────────────────────────────


class TestBalanceSheetFeatures:
    def test_output_columns_present(self):
        inc = _basic_income_df()
        bs = _basic_bs_df(inc.index)
        feat = balance_sheet_features(inc, bs)
        expected = {
            "roe", "roa", "roic", "debt_to_equity", "debt_to_assets",
            "leverage_ratio", "current_ratio", "quick_ratio", "cash_ratio",
            "asset_turnover", "receivables_turnover", "inventory_turnover",
            "working_capital_ratio", "cash_to_assets", "log_total_assets",
            "log_equity",
        }
        assert expected.issubset(set(feat.columns))

    def test_no_inf_values(self):
        inc = _basic_income_df()
        bs = _basic_bs_df(inc.index)
        feat = balance_sheet_features(inc, bs)
        assert not np.isinf(feat.values).any()

    def test_zero_equity_gives_nan_not_inf(self):
        inc = _basic_income_df()
        bs = _basic_bs_df(inc.index)
        bs["Stockholders Equity"] = 0.0
        feat = balance_sheet_features(inc, bs)
        assert not np.isinf(feat.values).any()
        assert feat["roe"].isna().all()

    def test_zero_assets_gives_nan_not_inf(self):
        inc = _basic_income_df()
        bs = _basic_bs_df(inc.index)
        bs["Total Assets"] = 0.0
        feat = balance_sheet_features(inc, bs)
        assert not np.isinf(feat.values).any()
        assert feat["roa"].isna().all()

    def test_output_index_matches_input(self):
        inc = _basic_income_df()
        bs = _basic_bs_df(inc.index)
        feat = balance_sheet_features(inc, bs)
        assert feat.index.equals(inc.index)

    def test_current_ratio_positive(self):
        """With positive assets and liabilities, current ratio > 0."""
        inc = _basic_income_df()
        bs = _basic_bs_df(inc.index)
        feat = balance_sheet_features(inc, bs)
        valid = feat["current_ratio"].dropna()
        assert (valid > 0).all()


# ── qoq_features ──────────────────────────────────────────────────────────────


class TestQoqFeatures:
    def _make_feat_with_prereqs(self):
        """Build feat DataFrame with columns that qoq_features reads."""
        inc = _basic_income_df(n_symbols=2, n_dates=4)
        bs = _basic_bs_df(inc.index)
        feat = profitability_features(inc)
        feat = feat.join(size_features(inc))
        feat = feat.join(balance_sheet_features(inc, bs))
        cf_rng = np.random.default_rng(10)
        size = len(inc)
        cf = pd.DataFrame(
            {
                "Cash Flow From Continuing Operating Activities": cf_rng.integers(10, 200, size).astype(float),
                "Capital Expenditure": cf_rng.integers(1, 80, size).astype(float),
                "Depreciation And Amortization": cf_rng.integers(5, 60, size).astype(float),
                "Cash Dividends Paid": cf_rng.integers(0, 30, size).astype(float),
            },
            index=inc.index,
        )
        feat = feat.join(cashflow_features(inc, cf))
        return feat, inc

    def test_output_columns_present(self):
        feat, inc = self._make_feat_with_prereqs()
        result = qoq_features(feat, inc)
        expected = {
            "revenue_growth_qoq", "ni_growth_qoq", "ebitda_growth_qoq",
            "gross_margin_chg", "operating_margin_chg", "net_margin_chg",
            "roe_chg", "roa_chg", "leverage_chg",
        }
        assert expected.issubset(set(result.columns))

    def test_first_quarter_is_nan(self):
        """QoQ changes should be NaN for first quarter per symbol."""
        feat, inc = self._make_feat_with_prereqs()
        result = qoq_features(feat, inc)
        for sym in ["S0", "S1"]:
            first_val = result.loc[sym, "revenue_growth_qoq"].iloc[0]
            assert pd.isna(first_val)

    def test_no_inf_values(self):
        feat, inc = self._make_feat_with_prereqs()
        result = qoq_features(feat, inc)
        for c in result.columns:
            vals = result[c].dropna().values
            if len(vals) > 0:
                assert not np.isinf(vals).any(), f"inf found in {c}"

    def test_output_index_matches_input(self):
        feat, inc = self._make_feat_with_prereqs()
        result = qoq_features(feat, inc)
        assert result.index.equals(feat.index)


# ── clip_outliers ──────────────────────────────────────────────────────────────


class TestClipOutliers:
    def _make_df_with_outliers(self):
        symbols = [f"S{i}" for i in range(20)]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {"feat_a": rng.standard_normal(len(idx)),
             "feat_b": rng.standard_normal(len(idx))},
            index=idx,
        )
        # Inject extreme outlier well beyond 5σ
        df.iloc[0, 0] = 1000.0
        return df

    def test_outlier_clipped(self):
        df = self._make_df_with_outliers()
        original_max = df["feat_a"].max()
        result = clip_outliers(df.copy(), ["feat_a", "feat_b"], n_std=2)
        assert result["feat_a"].max() < original_max

    def test_shape_preserved(self):
        df = self._make_df_with_outliers()
        result = clip_outliers(df.copy(), ["feat_a"], n_std=5)
        assert result.shape == df.shape

    def test_missing_column_ignored(self):
        df = self._make_df_with_outliers()
        result = clip_outliers(df.copy(), ["feat_a", "nonexistent"], n_std=5)
        assert result.shape == df.shape

    def test_no_inf_after_clip(self):
        df = self._make_df_with_outliers()
        df.iloc[1, 1] = np.inf
        df = df.replace([np.inf, -np.inf], np.nan)
        result = clip_outliers(df.copy(), ["feat_a", "feat_b"], n_std=5)
        assert not np.isinf(result.values[~np.isnan(result.values)]).any()


# ── momentum_features ──────────────────────────────────────────────────────────


def _make_close_prices(symbols, n_days=400):
    """Build a tidy close_prices DataFrame."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    frames = []
    for sym in symbols:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        frames.append(pd.DataFrame({"date": dates, "symbol": sym, "close": px}))
    return pd.concat(frames, ignore_index=True)


class TestMomentumFeatures:
    def test_output_has_momentum_columns(self):
        symbols = ["A", "B"]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        inc = pd.DataFrame({"Total Revenue": [100.0] * len(idx)}, index=idx)
        cp = _make_close_prices(symbols)
        mom = momentum_features(inc, cp)
        assert "momentum_1m" in mom.columns

    def test_output_index_is_symbol_date(self):
        symbols = ["X"]
        dates = pd.date_range("2023-06-30", periods=1, freq="QE")
        idx = _make_index(symbols, dates)
        inc = pd.DataFrame({"Total Revenue": [100.0]}, index=idx)
        cp = _make_close_prices(symbols)
        mom = momentum_features(inc, cp)
        assert list(mom.index.names) == ["symbol", "date"]

    def test_no_inf_values(self):
        symbols = ["A", "B"]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        inc = pd.DataFrame({"Total Revenue": [100.0] * len(idx)}, index=idx)
        cp = _make_close_prices(symbols)
        mom = momentum_features(inc, cp)
        for c in mom.columns:
            vals = mom[c].dropna().values
            if len(vals) > 0:
                assert not np.isinf(vals).any(), f"inf in {c}"

    def test_uses_decision_date_cutoff(self):
        """Momentum should use prices up to q_date + EARNINGS_LAG_DAYS, not q_date."""
        from stock_data.config import EARNINGS_LAG_DAYS

        q_date = pd.Timestamp("2022-06-30")
        decision_date = q_date + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        idx = _make_index(["A"], [q_date])
        inc = pd.DataFrame({"Total Revenue": [100.0]}, index=idx)

        # Prices only after q_date but before decision_date
        dates = pd.bdate_range(q_date + pd.Timedelta(days=1), decision_date)
        cp = pd.DataFrame({
            "date": dates, "symbol": "A", "close": range(100, 100 + len(dates)),
        })
        # With old cutoff (<= q_date) this would have no data; with decision_date it should
        mom = momentum_features(inc, cp)
        assert len(mom) == 1
        # Should have used the post-q_date prices
        has_values = mom.drop(columns=[], errors="ignore").notna().any(axis=1)
        assert has_values.iloc[0], "momentum should use prices up to decision_date"


# ── macro_features ─────────────────────────────────────────────────────────────


class TestMacroFeatures:
    def test_output_maps_macro_to_index(self):
        symbols = ["A", "B"]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        inc = pd.DataFrame({"Total Revenue": [100.0] * len(idx)}, index=idx)
        macro_df = pd.DataFrame(
            {"vix": [20.0, 22.0, 25.0], "yield_curve": [0.5, 0.4, 0.3]},
            index=pd.date_range("2023-01-01", periods=3, freq="ME"),
        )
        result = macro_features(inc, macro_df)
        assert "vix" in result.columns
        assert result.index.equals(inc.index)

    def test_empty_macro_returns_empty_cols(self):
        symbols = ["A"]
        dates = pd.date_range("2023-01-01", periods=1, freq="QE")
        idx = _make_index(symbols, dates)
        inc = pd.DataFrame({"Total Revenue": [100.0]}, index=idx)
        result = macro_features(inc, None)
        assert len(result.columns) == 0
        assert result.index.equals(inc.index)

    def test_none_macro_returns_empty(self):
        symbols = ["A"]
        dates = pd.date_range("2023-01-01", periods=1, freq="QE")
        idx = _make_index(symbols, dates)
        inc = pd.DataFrame({"Total Revenue": [100.0]}, index=idx)
        result = macro_features(inc, pd.DataFrame())
        assert len(result.columns) == 0

    def test_uses_decision_date_cutoff(self):
        """Macro should use data up to q_date + EARNINGS_LAG_DAYS, not q_date."""
        from stock_data.config import EARNINGS_LAG_DAYS

        q_date = pd.Timestamp("2023-03-31")
        decision_date = q_date + pd.Timedelta(days=EARNINGS_LAG_DAYS)
        idx = _make_index(["A"], [q_date])
        inc = pd.DataFrame({"Total Revenue": [100.0]}, index=idx)

        # Macro data: one entry before q_date, one between q_date and decision_date
        macro_df = pd.DataFrame(
            {"vix": [20.0, 30.0]},
            index=[q_date - pd.Timedelta(days=10), q_date + pd.Timedelta(days=20)],
        )
        result = macro_features(inc, macro_df)
        # Should pick up the more recent value (30.0) available after q_date
        assert result.loc[("A", q_date), "vix"] == 30.0


# ── annual_growth_features ─────────────────────────────────────────────────────


class TestAnnualGrowthFeatures:
    def test_none_annual_returns_unchanged(self):
        symbols = ["A"]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        feat = pd.DataFrame({"gross_margin": [0.3, 0.4]}, index=idx)
        result = annual_growth_features(feat, None)
        assert result.equals(feat)

    def test_empty_annual_returns_unchanged(self):
        symbols = ["A"]
        dates = pd.date_range("2023-01-01", periods=2, freq="QE")
        idx = _make_index(symbols, dates)
        feat = pd.DataFrame({"gross_margin": [0.3, 0.4]}, index=idx)
        result = annual_growth_features(feat, pd.DataFrame())
        assert result.equals(feat)

    def test_annual_growth_columns_added(self):
        symbols = ["A"]
        q_dates = pd.date_range("2023-01-01", periods=4, freq="QE")
        idx = _make_index(symbols, q_dates)
        feat = pd.DataFrame({"gross_margin": [0.3, 0.35, 0.32, 0.38]}, index=idx)

        a_dates = pd.date_range("2021-12-31", periods=3, freq="YE")
        a_idx = _make_index(symbols, a_dates)
        annual_raw = pd.DataFrame(
            {"Total Revenue": [100.0, 120.0, 150.0],
             "Net Income": [10.0, 15.0, 20.0],
             "EBITDA": [30.0, 40.0, 55.0]},
            index=a_idx,
        )
        result = annual_growth_features(feat, annual_raw)
        assert "annual_rev_growth" in result.columns
