"""Unit tests for stock_data.dataset (pure-computation functions only)."""

import numpy as np
import pandas as pd
import pytest

from stock_data.dataset import (
    drop_sparse_pairs,
    pivot_statements,
    reshape_annual_income,
    reshape_statements,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _wide_stmt(n_items=4, n_dates=3, seed=0):
    """Return a DataFrame shaped like a yfinance quarterly statement.

    Shape: (items, dates) — rows are financial line-items, columns are dates.
    """
    rng = np.random.default_rng(seed)
    items = [f"Item{i}" for i in range(n_items)]
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="QE")
    return pd.DataFrame(rng.integers(10, 500, (n_items, n_dates)).astype(float),
                        index=items, columns=dates)


def _make_raw_dict(symbols=("AAA", "BBB"), n_items=4, n_dates=3):
    return {sym: _wide_stmt(n_items=n_items, n_dates=n_dates) for sym in symbols}


# ── reshape_statements ─────────────────────────────────────────────────────────


class TestReshapeStatements:
    def test_result_has_symbol_date_item_index(self):
        raw = _make_raw_dict()
        combined = reshape_statements(raw)
        assert list(combined.index.names) == ["symbol", "date", "item"]

    def test_all_symbols_present(self):
        symbols = ("X", "Y", "Z")
        raw = _make_raw_dict(symbols=symbols)
        combined = reshape_statements(raw)
        in_index = combined.index.get_level_values("symbol").unique().tolist()
        assert set(in_index) == set(symbols)

    def test_row_count(self):
        symbols = ("A", "B")
        n_items, n_dates = 3, 4
        raw = _make_raw_dict(symbols=symbols, n_items=n_items, n_dates=n_dates)
        combined = reshape_statements(raw)
        # Each symbol contributes n_items * n_dates rows
        assert len(combined) == len(symbols) * n_items * n_dates

    def test_values_are_numeric(self):
        raw = _make_raw_dict()
        combined = reshape_statements(raw)
        assert pd.api.types.is_numeric_dtype(combined["value"])

    def test_correct_value_roundtrip(self):
        """A known value in the input should survive the reshape."""
        raw = _make_raw_dict(symbols=("AAPL",), n_items=2, n_dates=2)
        known_item = raw["AAPL"].index[0]
        known_date = raw["AAPL"].columns[0]
        known_val = raw["AAPL"].loc[known_item, known_date]
        combined = reshape_statements(raw)
        actual = combined.loc[("AAPL", known_date, known_item), "value"]
        assert actual == pytest.approx(known_val)


# ── pivot_statements ───────────────────────────────────────────────────────────


class TestPivotStatements:
    def _stacked(self):
        raw = _make_raw_dict(symbols=("P", "Q"), n_items=3, n_dates=2)
        return reshape_statements(raw)

    def test_result_has_symbol_date_index(self):
        combined = self._stacked()
        wide = pivot_statements(combined)
        assert list(wide.index.names) == ["symbol", "date"]

    def test_result_is_wide_format(self):
        raw = _make_raw_dict(symbols=("P",), n_items=3, n_dates=2)
        combined = reshape_statements(raw)
        wide = pivot_statements(combined)
        # Items should become columns
        assert wide.shape[1] == 3

    def test_no_column_name_artifact(self):
        combined = self._stacked()
        wide = pivot_statements(combined)
        assert wide.columns.name is None

    def test_values_preserved(self):
        raw = _make_raw_dict(symbols=("T",), n_items=2, n_dates=2)
        combined = reshape_statements(raw)
        wide = pivot_statements(combined)
        sym = "T"
        date = raw[sym].columns[0]
        item = raw[sym].index[0]
        expected = raw[sym].loc[item, date]
        assert wide.loc[(sym, date), item] == pytest.approx(expected)


# ── drop_sparse_pairs ──────────────────────────────────────────────────────────


class TestDropSparsePairs:
    def _stacked_with_nans(self):
        """Build a stacked DataFrame where one (symbol, date) group is all-NaN."""
        raw = _make_raw_dict(symbols=("S1", "S2"), n_items=4, n_dates=3)
        # Introduce NaNs for S1 at the first date
        first_date = raw["S1"].columns[0]
        raw["S1"][first_date] = np.nan
        return reshape_statements(raw)

    def test_dense_pairs_retained(self):
        combined = self._stacked_with_nans()
        before_count = len(combined)
        filtered = drop_sparse_pairs(combined, threshold=0.5)
        # Dense rows should still exist
        assert len(filtered) > 0

    def test_all_nan_pair_dropped(self):
        raw = _make_raw_dict(symbols=("S1",), n_items=4, n_dates=2)
        first_date = raw["S1"].columns[0]
        raw["S1"][first_date] = np.nan
        combined = reshape_statements(raw)
        # threshold=0.5 → 100% NaN group gets dropped
        filtered = drop_sparse_pairs(combined, threshold=0.5)
        remaining_dates = filtered.index.get_level_values("date").unique()
        assert first_date not in remaining_dates

    def test_threshold_zero_drops_any_nan(self):
        raw = _make_raw_dict(symbols=("A",), n_items=4, n_dates=2)
        # Set one value to NaN in first date (25% NaN fraction)
        first_date = raw["A"].columns[0]
        raw["A"].iloc[0, 0] = np.nan
        combined = reshape_statements(raw)
        # threshold=0 → even 25% NaN triggers a drop
        filtered = drop_sparse_pairs(combined, threshold=0.0)
        remaining_dates = filtered.index.get_level_values("date").unique()
        assert first_date not in remaining_dates

    def test_high_threshold_retains_sparse_pairs(self):
        """With threshold=1.0, only fully-NaN groups are dropped."""
        raw = _make_raw_dict(symbols=("A",), n_items=4, n_dates=2)
        first_date = raw["A"].columns[0]
        raw["A"].iloc[0, 0] = np.nan  # only 25% NaN
        combined = reshape_statements(raw)
        filtered = drop_sparse_pairs(combined, threshold=1.0)
        assert len(filtered) == len(combined)


# ── reshape_annual_income ──────────────────────────────────────────────────────


class TestReshapeAnnualIncome:
    def _make_annual_dict(self, symbols=("MSFT", "GOOG"), n_years=3):
        """Build a dict like {symbol: DataFrame(items x years)}."""
        rng = np.random.default_rng(5)
        items = ["Total Revenue", "Net Income", "EBITDA"]
        dates = pd.date_range("2020-01-01", periods=n_years, freq="YE")
        return {
            sym: pd.DataFrame(
                rng.integers(100, 2000, (len(items), n_years)).astype(float),
                index=items,
                columns=dates,
            )
            for sym in symbols
        }

    def test_empty_dict_returns_empty_dataframe(self):
        result = reshape_annual_income({})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_result_has_symbol_date_index(self):
        annual_dict = self._make_annual_dict()
        result = reshape_annual_income(annual_dict)
        assert "symbol" in result.index.names
        assert "date" in result.index.names

    def test_all_symbols_present(self):
        symbols = ("AAPL", "TSLA")
        annual_dict = self._make_annual_dict(symbols=symbols)
        result = reshape_annual_income(annual_dict)
        in_index = result.index.get_level_values("symbol").unique().tolist()
        assert set(in_index) == set(symbols)

    def test_numeric_columns_only(self):
        annual_dict = self._make_annual_dict()
        result = reshape_annual_income(annual_dict)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Column '{col}' is not numeric"
            )

    def test_row_count(self):
        symbols = ("A", "B")
        n_years = 3
        annual_dict = self._make_annual_dict(symbols=symbols, n_years=n_years)
        result = reshape_annual_income(annual_dict)
        # Each symbol × year produces one row
        assert len(result) == len(symbols) * n_years
