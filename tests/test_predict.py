"""Unit tests for stock_data.modeling.predict."""

import numpy as np
import pytest

from stock_data.modeling.predict import (
    bootstrap_ci,
    mv_optimize,
    mv_optimize_diag,
    portfolio_turnover,
    safe_spearmanr,
    shrink_to_mean,
    winsorize,
)


# ── winsorize ──────────────────────────────────────────────────────────────────


class TestWinsorize:
    def test_shape_preserved(self):
        a = np.array([1.0, 2.0, 3.0, 100.0, -50.0])
        out = winsorize(a, pct=0.05)
        assert out.shape == a.shape

    def test_values_clipped_to_quantile_range(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(200)
        pct = 0.05
        out = winsorize(a, pct=pct)
        lo, hi = np.nanquantile(a, [pct, 1 - pct])
        assert out.min() >= lo - 1e-12
        assert out.max() <= hi + 1e-12

    def test_no_change_on_already_clipped_data(self):
        a = np.linspace(0, 1, 100)
        pct = 0.0
        out = winsorize(a, pct=pct)
        np.testing.assert_array_almost_equal(out, a)

    def test_extreme_outlier_clipped(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1_000.0])
        out = winsorize(a, pct=0.10)
        assert out[-1] < 1_000.0

    def test_single_element_array(self):
        a = np.array([42.0])
        out = winsorize(a, pct=0.05)
        assert out[0] == pytest.approx(42.0)


# ── shrink_to_mean ─────────────────────────────────────────────────────────────


class TestShrinkToMean:
    def test_scalar_alpha_zero_returns_mean(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = shrink_to_mean(a, alpha=0.0)
        np.testing.assert_allclose(out, np.mean(a))

    def test_scalar_alpha_one_returns_original(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = shrink_to_mean(a, alpha=1.0)
        np.testing.assert_array_almost_equal(out, a)

    def test_scalar_alpha_half_midpoint(self):
        a = np.array([0.0, 10.0])
        out = shrink_to_mean(a, alpha=0.5)
        # mean = 5, shrunk = 0.5*[0,10] + 0.5*5 = [2.5, 7.5]
        expected = np.array([2.5, 7.5])
        np.testing.assert_array_almost_equal(out, expected)

    def test_shape_preserved(self):
        a = np.arange(20, dtype=float)
        out = shrink_to_mean(a, alpha=0.3)
        assert out.shape == a.shape


# ── mv_optimize ────────────────────────────────────────────────────────────────


class TestMvOptimize:
    def _simple_case(self, n=5, max_w=0.3):
        rng = np.random.default_rng(42)
        mu = rng.standard_normal(n)
        # Positive semi-definite covariance via A'A
        A = rng.standard_normal((n, n))
        cov = A.T @ A / n + np.eye(n) * 0.01
        return mu, cov, max_w

    def test_weights_sum_to_one(self):
        mu, cov, max_w = self._simple_case()
        w = mv_optimize(mu, cov, max_w, lam=1.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self):
        mu, cov, max_w = self._simple_case()
        w = mv_optimize(mu, cov, max_w, lam=1.0)
        assert (w >= -1e-8).all()

    def test_weights_respect_max_w(self):
        max_w = 0.3
        mu, cov, _ = self._simple_case(max_w=max_w)
        w = mv_optimize(mu, cov, max_w, lam=1.0)
        assert (w <= max_w + 1e-6).all()

    def test_output_length_matches_input(self):
        n = 7
        mu, cov, max_w = self._simple_case(n=n)
        w = mv_optimize(mu, cov, max_w, lam=1.0)
        assert len(w) == n


# ── mv_optimize_diag ───────────────────────────────────────────────────────────


class TestMvOptimizeDiag:
    def _simple_case(self, n=5, max_w=0.4):
        rng = np.random.default_rng(7)
        mu = rng.standard_normal(n)
        vol = rng.uniform(0.1, 0.5, n)
        return mu, vol, max_w

    def test_weights_sum_to_one(self):
        mu, vol, max_w = self._simple_case()
        w = mv_optimize_diag(mu, vol, max_w, lam=1.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self):
        mu, vol, max_w = self._simple_case()
        w = mv_optimize_diag(mu, vol, max_w, lam=1.0)
        assert (w >= -1e-8).all()

    def test_weights_respect_max_w(self):
        max_w = 0.4
        mu, vol, _ = self._simple_case(max_w=max_w)
        w = mv_optimize_diag(mu, vol, max_w, lam=1.0)
        assert (w <= max_w + 1e-6).all()

    def test_output_length_matches_input(self):
        n = 8
        mu, vol, max_w = self._simple_case(n=n)
        w = mv_optimize_diag(mu, vol, max_w, lam=1.0)
        assert len(w) == n


# ── portfolio_turnover ─────────────────────────────────────────────────────────


class TestPortfolioTurnover:
    def test_zero_turnover_same_portfolio(self):
        syms = ["A", "B", "C"]
        w = np.array([0.4, 0.4, 0.2])
        to = portfolio_turnover(w, syms, w.copy(), syms)
        assert to == pytest.approx(0.0, abs=1e-9)

    def test_complete_replacement_turnover_is_one(self):
        """Selling everything and buying all new stocks → turnover = 1."""
        prev_w = np.array([0.5, 0.5])
        prev_syms = ["A", "B"]
        cur_w = np.array([0.5, 0.5])
        cur_syms = ["C", "D"]
        to = portfolio_turnover(prev_w, prev_syms, cur_w, cur_syms)
        assert to == pytest.approx(1.0, abs=1e-9)

    def test_partial_turnover(self):
        """Shifting 20 percentage points from A to B → turnover = 0.2."""
        prev_w = np.array([0.5, 0.5])
        prev_syms = ["A", "B"]
        cur_w = np.array([0.3, 0.7])
        cur_syms = ["A", "B"]
        to = portfolio_turnover(prev_w, prev_syms, cur_w, cur_syms)
        assert to == pytest.approx(0.2, abs=1e-9)

    def test_new_position_added(self):
        """Adding a new position; turnover reflects both the buy and sell."""
        prev_w = np.array([1.0])
        prev_syms = ["A"]
        cur_w = np.array([0.5, 0.5])
        cur_syms = ["A", "B"]
        to = portfolio_turnover(prev_w, prev_syms, cur_w, cur_syms)
        # |0.5-1.0| + |0.5-0| = 0.5+0.5 = 1.0, /2 = 0.5
        assert to == pytest.approx(0.5, abs=1e-9)


# ── bootstrap_ci ───────────────────────────────────────────────────────────────


class TestBootstrapCi:
    def test_lo_le_hi(self):
        vals = np.arange(1, 101, dtype=float)
        lo, hi, p_neg, _ = bootstrap_ci(vals, n_boot=500, seed=0)
        assert lo <= hi

    def test_p_neg_in_zero_one(self):
        vals = np.arange(1, 101, dtype=float)
        lo, hi, p_neg, _ = bootstrap_ci(vals, n_boot=500, seed=0)
        assert 0.0 <= p_neg <= 1.0

    def test_boot_means_length(self):
        vals = np.arange(1, 51, dtype=float)
        n_boot = 300
        _, _, _, boot_means = bootstrap_ci(vals, n_boot=n_boot, seed=42)
        assert len(boot_means) == n_boot

    def test_all_positive_values_p_neg_is_zero(self):
        vals = np.ones(50) * 10.0
        _, _, p_neg, _ = bootstrap_ci(vals, n_boot=200, seed=1)
        assert p_neg == 0.0

    def test_all_negative_values_p_neg_is_one(self):
        vals = np.ones(50) * -10.0
        _, _, p_neg, _ = bootstrap_ci(vals, n_boot=200, seed=1)
        assert p_neg == 1.0

    def test_reproducible_with_seed(self):
        vals = np.arange(1, 101, dtype=float)
        r1 = bootstrap_ci(vals, n_boot=100, seed=99)
        r2 = bootstrap_ci(vals, n_boot=100, seed=99)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]


# ── safe_spearmanr ─────────────────────────────────────────────────────────────


class TestSafeSpearmanr:
    def test_perfect_positive_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert safe_spearmanr(a, a) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert safe_spearmanr(a, b) == pytest.approx(-1.0)

    def test_too_few_elements_returns_nan(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert np.isnan(safe_spearmanr(a, b))

    def test_result_in_valid_range(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        r = safe_spearmanr(a, b)
        assert -1.0 <= r <= 1.0

    def test_constant_input_returns_nan(self):
        a = np.ones(10)
        b = np.arange(10, dtype=float)
        r = safe_spearmanr(a, b)
        assert np.isnan(r)


# ── mv_optimize_turnover ──────────────────────────────────────────────────────

from stock_data.modeling.predict import mv_optimize_turnover


class TestMvOptimizeTurnover:
    def _simple_case(self, n=5, max_w=0.4):
        rng = np.random.default_rng(42)
        mu = rng.standard_normal(n)
        A = rng.standard_normal((n, n))
        cov = A.T @ A / n + np.eye(n) * 0.01
        prev_w = np.ones(n) / n
        return mu, cov, max_w, prev_w

    def test_weights_sum_to_one(self):
        mu, cov, max_w, prev_w = self._simple_case()
        w = mv_optimize_turnover(mu, cov, max_w, lam=1.0, prev_w=prev_w, max_turnover=0.3)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_weights_non_negative(self):
        mu, cov, max_w, prev_w = self._simple_case()
        w = mv_optimize_turnover(mu, cov, max_w, lam=1.0, prev_w=prev_w, max_turnover=0.3)
        assert (w >= -1e-6).all()

    def test_turnover_respects_constraint(self):
        mu, cov, max_w, prev_w = self._simple_case()
        max_to = 0.2
        w = mv_optimize_turnover(mu, cov, max_w, lam=1.0, prev_w=prev_w, max_turnover=max_to)
        actual_to = np.abs(w - prev_w).sum() / 2
        assert actual_to <= max_to + 1e-3

    def test_none_prev_w_defaults_to_equal(self):
        mu, cov, max_w, _ = self._simple_case()
        w = mv_optimize_turnover(mu, cov, max_w, lam=1.0, prev_w=None, max_turnover=0.5)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
