"""Unit tests for stock_data.modeling.train."""

import numpy as np
import pandas as pd

from stock_data.modeling.train import _select_features


# ── _select_features ───────────────────────────────────────────────────────────


class TestSelectFeatures:
    def _make_df_and_y(self, n_samples, n_features, rng=None):
        rng = rng or np.random.default_rng(42)
        data = rng.standard_normal((n_samples, n_features))
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=cols)
        y = pd.Series(rng.standard_normal(n_samples))
        return df, y

    def test_all_features_kept_below_threshold(self):
        """When ratio is safe, all columns are returned."""
        df, y = self._make_df_and_y(1000, 100)  # ratio = 0.1
        result = _select_features(df, y, threshold=0.3)
        assert len(result) == 100
        assert set(result) == set(df.columns)

    def test_features_reduced_above_threshold(self):
        """When ratio is too high, features are reduced."""
        df, y = self._make_df_and_y(100, 180)  # ratio = 1.8
        result = _select_features(df, y, threshold=0.3)
        assert len(result) == int(100 * 0.3)
        assert all(c in df.columns for c in result)

    def test_keeps_most_predictive_feature(self):
        """A feature linearly correlated with target should be selected.

        This directly tests the F-regression criterion: the feature with
        the highest linear relationship to y must be in the selected set,
        regardless of its variance.
        """
        rng = np.random.default_rng(0)
        n_samples, n_features = 100, 20
        noise = rng.standard_normal((n_samples, n_features)) * 0.01
        y = pd.Series(rng.standard_normal(n_samples))
        data = noise.copy()
        # f0 is strongly correlated with y; scaled down (low variance)
        data[:, 0] = y.values * 0.1 + rng.standard_normal(n_samples) * 0.001
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=cols)
        # Force selection (threshold=0.05 → keep top 5 of 20)
        result = _select_features(df, y, threshold=0.05)
        assert "f0" in result

    def test_rank_features_not_penalised(self):
        """Rank (percentile) features with bounded variance should be selected
        when they are predictive, not dropped due to low spread.

        Variance-based selection would drop rank features (var ≈ 1/12 for
        uniform [0,1]) in favour of high-magnitude raw features. F-regression
        must select the rank feature if it is actually predictive.
        """
        rng = np.random.default_rng(7)
        n_samples = 100
        # Rank feature: uniform on [0, 1] — low variance, but correlated with y
        rank_feat = rng.uniform(0, 1, n_samples)
        y = pd.Series(rank_feat + rng.standard_normal(n_samples) * 0.1)
        # Raw features: high variance (std ~ 100), zero correlation with y
        raw_feats = rng.standard_normal((n_samples, 19)) * 100
        data = np.column_stack([rank_feat, raw_feats])
        cols = ["rank_f"] + [f"raw_{i}" for i in range(19)]
        df = pd.DataFrame(data, columns=cols)
        # threshold=0.05 → keep top 5 of 20 features
        result = _select_features(df, y, threshold=0.05)
        assert "rank_f" in result, (
            "Rank feature with low variance but high predictive power must be "
            "selected (variance-based selection would incorrectly drop it)"
        )

    def test_nan_columns_ranked_last(self):
        """All-NaN columns should not be selected (F-score → 0 after imputation)."""
        rng = np.random.default_rng(1)
        n_samples, n_features = 100, 50  # ratio = 0.5 > 0.3 → selection triggers
        data = rng.standard_normal((n_samples, n_features))
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=cols)
        df["f0"] = np.nan  # all-NaN → imputed to constant → F = 0
        y = pd.Series(rng.standard_normal(n_samples))
        result = _select_features(df, y, threshold=0.3)
        assert "f0" not in result

    def test_single_feature_always_kept(self):
        """A single-feature DataFrame should pass through."""
        df, y = self._make_df_and_y(10, 1)
        result = _select_features(df, y, threshold=0.3)
        assert len(result) == 1

    def test_exact_threshold_boundary(self):
        """At exactly the threshold, all features should be kept."""
        df, y = self._make_df_and_y(100, 30)  # ratio = 0.3 exactly
        result = _select_features(df, y, threshold=0.3)
        assert len(result) == 30

    def test_returns_list_of_column_names(self):
        df, y = self._make_df_and_y(100, 180)
        result = _select_features(df, y, threshold=0.3)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_n_keep_at_least_one(self):
        """Even with very few samples, at least one feature is kept."""
        df, y = self._make_df_and_y(2, 100)  # ratio = 50
        result = _select_features(df, y, threshold=0.3)
        assert len(result) >= 1
