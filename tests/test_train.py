"""Unit tests for stock_data.modeling.train."""

import numpy as np
import pandas as pd
import pytest

from stock_data.modeling.train import _select_features


# ── _select_features ───────────────────────────────────────────────────────────


class TestSelectFeatures:
    def _make_df(self, n_samples, n_features, rng=None):
        rng = rng or np.random.default_rng(42)
        data = rng.standard_normal((n_samples, n_features))
        cols = [f"f{i}" for i in range(n_features)]
        return pd.DataFrame(data, columns=cols)

    def test_all_features_kept_below_threshold(self):
        """When ratio is safe, all columns are returned."""
        df = self._make_df(1000, 100)  # ratio = 0.1
        result = _select_features(df, threshold=0.3)
        assert len(result) == 100
        assert set(result) == set(df.columns)

    def test_features_reduced_above_threshold(self):
        """When ratio is too high, features are reduced."""
        df = self._make_df(100, 180)  # ratio = 1.8
        result = _select_features(df, threshold=0.3)
        assert len(result) == int(100 * 0.3)
        assert all(c in df.columns for c in result)

    def test_keeps_highest_variance_columns(self):
        """Selected features should be the highest-variance ones."""
        rng = np.random.default_rng(0)
        n_samples, n_features = 50, 20  # ratio = 0.4 > 0.3
        data = rng.standard_normal((n_samples, n_features))
        # Make first column have very high variance
        data[:, 0] *= 100
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=cols)
        result = _select_features(df, threshold=0.3)
        assert "f0" in result

    def test_nan_variance_columns_ranked_last(self):
        """All-NaN columns should not be selected."""
        rng = np.random.default_rng(1)
        n_samples, n_features = 50, 20
        data = rng.standard_normal((n_samples, n_features))
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=cols)
        df["f0"] = np.nan  # all-NaN column
        result = _select_features(df, threshold=0.3)
        assert "f0" not in result

    def test_single_feature_always_kept(self):
        """A single-feature DataFrame should pass through."""
        df = self._make_df(10, 1)
        result = _select_features(df, threshold=0.3)
        assert len(result) == 1

    def test_exact_threshold_boundary(self):
        """At exactly the threshold, all features should be kept."""
        df = self._make_df(100, 30)  # ratio = 0.3 exactly
        result = _select_features(df, threshold=0.3)
        assert len(result) == 30

    def test_returns_list_of_column_names(self):
        df = self._make_df(100, 180)
        result = _select_features(df, threshold=0.3)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_n_keep_at_least_one(self):
        """Even with very few samples, at least one feature is kept."""
        df = self._make_df(2, 100)  # ratio = 50
        result = _select_features(df, threshold=0.3)
        assert len(result) >= 1
