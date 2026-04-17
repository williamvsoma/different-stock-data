"""Unit tests for pure-computation functions in stock_data.dataset."""

import numpy as np
import pandas as pd
import pytest

from stock_data.dataset import compute_forward_returns, compute_realized_vol


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_close_prices(symbols, n_days=400, start="2022-01-01", seed=0):
    """Build a tidy close_prices DataFrame with realistic random walk prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    frames = []
    for sym in symbols:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        frames.append(pd.DataFrame({"date": dates, "symbol": sym, "close": px}))
    return pd.concat(frames, ignore_index=True)


def _make_sym_date_pairs(symbols, dates):
    """Build a (symbol, date) MultiIndex."""
    tuples = [(s, d) for s in symbols for d in dates]
    return pd.MultiIndex.from_tuples(tuples, names=["symbol", "date"])


# ── compute_forward_returns ────────────────────────────────────────────────────


class TestComputeForwardReturns:
    def test_returns_dataframe_with_expected_columns(self):
        symbols = ["A", "B"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=3, freq="QE")
        pairs = _make_sym_date_pairs(symbols, q_dates)
        result = compute_forward_returns(cp, pairs)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"symbol", "date", "buy_date", "sell_date", "next_q_return"}
        assert expected_cols.issubset(set(result.columns))

    def test_returns_are_finite(self):
        symbols = ["X"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=3, freq="QE")
        pairs = _make_sym_date_pairs(symbols, q_dates)
        result = compute_forward_returns(cp, pairs)
        assert np.isfinite(result["next_q_return"]).all()

    def test_buy_date_after_quarter_end(self):
        """Buy date should be after quarter end (EARNINGS_LAG_DAYS delay)."""
        symbols = ["A"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        pairs = _make_sym_date_pairs(symbols, q_dates)
        result = compute_forward_returns(cp, pairs)
        if len(result) > 0:
            for _, row in result.iterrows():
                assert row["buy_date"] > row["date"]

    def test_sell_date_after_buy_date(self):
        symbols = ["A"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        pairs = _make_sym_date_pairs(symbols, q_dates)
        result = compute_forward_returns(cp, pairs)
        if len(result) > 0:
            for _, row in result.iterrows():
                assert row["sell_date"] > row["buy_date"]

    def test_no_results_for_symbol_not_in_prices(self):
        cp = _make_close_prices(["A"], n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        pairs = _make_sym_date_pairs(["MISSING"], q_dates)
        result = compute_forward_returns(cp, pairs)
        assert len(result) == 0

    def test_buy_date_on_or_after_decision_date(self):
        """Buy date should be >= q_date + EARNINGS_LAG_DAYS (no backward slack)."""
        from stock_data.config import EARNINGS_LAG_DAYS

        symbols = ["A"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        pairs = _make_sym_date_pairs(symbols, q_dates)
        result = compute_forward_returns(cp, pairs)
        for _, row in result.iterrows():
            decision_date = row["date"] + pd.Timedelta(days=EARNINGS_LAG_DAYS)
            assert row["buy_date"] >= decision_date, (
                f"buy_date {row['buy_date']} is before decision_date {decision_date}"
            )


# ── compute_realized_vol ───────────────────────────────────────────────────────


class TestComputeRealizedVol:
    def _make_returns_df(self, symbols, q_dates, close_prices):
        """Build a returns_df with buy/sell dates matching compute_forward_returns output."""
        from stock_data.config import EARNINGS_LAG_DAYS

        records = []
        for sym in symbols:
            grp = close_prices[close_prices["symbol"] == sym].set_index("date").sort_index()
            for q_date in q_dates:
                buy_date = q_date + pd.Timedelta(days=EARNINGS_LAG_DAYS)
                sell_date = buy_date + pd.DateOffset(months=3)
                buy_w = grp.loc[buy_date - pd.Timedelta(days=5):buy_date + pd.Timedelta(days=5)]
                sell_w = grp.loc[sell_date - pd.Timedelta(days=5):sell_date + pd.Timedelta(days=5)]
                if len(buy_w) > 0 and len(sell_w) > 0:
                    records.append({
                        "symbol": sym, "date": q_date,
                        "buy_date": buy_w.index[0], "sell_date": sell_w.index[0],
                    })
        return pd.DataFrame(records)

    def test_returns_expected_columns(self):
        symbols = ["A"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        ret_df = self._make_returns_df(symbols, q_dates, cp)
        if len(ret_df) == 0:
            pytest.skip("no valid return windows")
        result = compute_realized_vol(ret_df, cp)
        expected = {"symbol", "date", "realized_vol", "realized_downside_vol", "realized_max_dd"}
        assert expected.issubset(set(result.columns))

    def test_realized_vol_non_negative(self):
        symbols = ["A", "B"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        ret_df = self._make_returns_df(symbols, q_dates, cp)
        if len(ret_df) == 0:
            pytest.skip("no valid return windows")
        result = compute_realized_vol(ret_df, cp)
        assert (result["realized_vol"].dropna() >= 0).all()

    def test_max_drawdown_non_positive(self):
        symbols = ["A"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        ret_df = self._make_returns_df(symbols, q_dates, cp)
        if len(ret_df) == 0:
            pytest.skip("no valid return windows")
        result = compute_realized_vol(ret_df, cp)
        assert (result["realized_max_dd"].dropna() <= 0).all()

    def test_no_inf_values(self):
        symbols = ["A"]
        cp = _make_close_prices(symbols, n_days=500)
        q_dates = pd.date_range("2022-06-30", periods=2, freq="QE")
        ret_df = self._make_returns_df(symbols, q_dates, cp)
        if len(ret_df) == 0:
            pytest.skip("no valid return windows")
        result = compute_realized_vol(ret_df, cp)
        for c in result.select_dtypes(include=[np.number]).columns:
            vals = result[c].dropna().values
            assert not np.isinf(vals).any(), f"inf in {c}"

    def test_empty_returns_df(self):
        cp = _make_close_prices(["A"], n_days=100)
        ret_df = pd.DataFrame(columns=["symbol", "date", "buy_date", "sell_date"])
        result = compute_realized_vol(ret_df, cp)
        assert len(result) == 0
