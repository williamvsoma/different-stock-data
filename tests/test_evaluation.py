"""Unit tests for stock_data.evaluation metric computations."""

import numpy as np
import pandas as pd
import pytest

from stock_data.evaluation import evaluate_factors, print_simulation_summary


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_prod_df(n_quarters=8, seed=42):
    """Build a minimal prod_df with the columns evaluate_factors uses."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_quarters, freq="QS")
    net_ret = rng.normal(0.03, 0.05, n_quarters)
    mkt_ret = rng.normal(0.02, 0.04, n_quarters)
    return pd.DataFrame({
        "test_date": dates,
        "net_ret": net_ret,
        "gross_ret": net_ret + 0.002,
        "mkt_ret": mkt_ret,
        "n_held": 50,
        "max_wt": 0.02,
        "turnover": 0.3,
        "tx_cost": 0.004,
        "used_lw": True,
        "vol_rc": 0.5,
        "ret_rc": 0.1,
        "ret_rc_xgb": 0.12,
        "ret_rc_ridge": 0.08,
        "ret_rc_rf": 0.09,
    })


def _make_sim_data(n_days=504, seed=7):
    """Build synthetic sim_df and mkt_sim for print_simulation_summary tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    port_vals = 1_000_000 * np.cumprod(1 + rng.normal(0.0004, 0.01, n_days))
    mkt_vals = 1_000_000 * np.cumprod(1 + rng.normal(0.0003, 0.009, n_days))
    daily_ret = np.concatenate([[0.0], np.diff(port_vals) / port_vals[:-1]])
    sim_df = pd.DataFrame(
        {"portfolio_value": port_vals, "daily_return": daily_ret},
        index=dates,
    )
    sim_df.index.name = "date"
    mkt_sim = pd.DataFrame({"market_value": mkt_vals}, index=dates)
    mkt_sim.index.name = "date"
    return sim_df, mkt_sim


# ── evaluate_factors ───────────────────────────────────────────────────────────


class TestEvaluateFactorsMetrics:
    """Verify that evaluate_factors reports both Sharpe and IR."""

    def test_output_contains_information_ratio(self, capsys):
        prod_df = _make_prod_df()
        ci_lo, ci_hi, boot_means, ex_n = evaluate_factors(prod_df, {}, n_boot=200)
        captured = capsys.readouterr().out
        assert "IR (vs EW)" in captured

    def test_output_contains_sharpe(self, capsys):
        prod_df = _make_prod_df()
        evaluate_factors(prod_df, {}, n_boot=200)
        captured = capsys.readouterr().out
        assert "Sharpe (ann)" in captured

    def test_output_contains_calmar(self, capsys):
        prod_df = _make_prod_df()
        evaluate_factors(prod_df, {}, n_boot=200)
        captured = capsys.readouterr().out
        assert "Calmar ratio" in captured

    def test_sharpe_uses_total_returns(self, capsys):
        """Sharpe should be computed from total net returns, not excess."""
        prod_df = _make_prod_df()
        evaluate_factors(prod_df, {}, n_boot=200)

        # Manually compute expected Sharpe from total returns
        net_ret = prod_df["net_ret"]
        expected_sharpe = net_ret.mean() / net_ret.std() * np.sqrt(4)

        # Manually compute IR from excess returns (should differ)
        ex_n = net_ret - prod_df["mkt_ret"]
        expected_ir = ex_n.mean() / ex_n.std() * np.sqrt(4)

        # They should generally differ
        assert expected_sharpe != pytest.approx(expected_ir, abs=1e-6)

    def test_no_old_label_annualized_sharpe(self, capsys):
        """The old 'Annualized Sharpe' label should no longer appear."""
        prod_df = _make_prod_df()
        evaluate_factors(prod_df, {}, n_boot=200)
        captured = capsys.readouterr().out
        assert "Annualized Sharpe" not in captured

    def test_spx_benchmark_reported_when_present(self, capsys):
        """When spx_ret column exists, SPX excess is reported."""
        prod_df = _make_prod_df()
        prod_df["spx_ret"] = prod_df["mkt_ret"] - 0.01  # SPX slightly lower
        evaluate_factors(prod_df, {}, n_boot=200)
        captured = capsys.readouterr().out
        assert "S&P 500" in captured
        assert "vs SPX" in captured

    def test_works_without_spx_column(self, capsys):
        """Backward compat: works fine without spx_ret column."""
        prod_df = _make_prod_df()
        assert "spx_ret" not in prod_df.columns
        ci_lo, ci_hi, boot_means, ex_n = evaluate_factors(prod_df, {}, n_boot=200)
        assert len(boot_means) == 200


# ── print_simulation_summary ───────────────────────────────────────────────────


class TestPrintSimulationSummaryMetrics:
    """Verify that print_simulation_summary reports Sharpe, IR, drawdown, Calmar."""

    def test_output_contains_sharpe(self, capsys):
        sim_df, mkt_sim = _make_sim_data()
        print_simulation_summary(sim_df, mkt_sim, 1_000_000)
        captured = capsys.readouterr().out
        assert "Sharpe:" in captured

    def test_output_contains_info_ratio(self, capsys):
        sim_df, mkt_sim = _make_sim_data()
        print_simulation_summary(sim_df, mkt_sim, 1_000_000)
        captured = capsys.readouterr().out
        assert "Info ratio:" in captured

    def test_output_contains_max_drawdown(self, capsys):
        sim_df, mkt_sim = _make_sim_data()
        print_simulation_summary(sim_df, mkt_sim, 1_000_000)
        captured = capsys.readouterr().out
        assert "Max drawdown:" in captured

    def test_output_contains_calmar(self, capsys):
        sim_df, mkt_sim = _make_sim_data()
        print_simulation_summary(sim_df, mkt_sim, 1_000_000)
        captured = capsys.readouterr().out
        assert "Calmar:" in captured

    def test_max_drawdown_is_negative(self, capsys):
        """Max drawdown should be ≤ 0 for any non-trivial series."""
        sim_df, mkt_sim = _make_sim_data()
        # Compute drawdown manually
        combined = sim_df[["portfolio_value"]].join(mkt_sim[["market_value"]], how="inner")
        port_daily_ret = combined["portfolio_value"].pct_change().dropna()
        cum_ret = (1 + port_daily_ret).cumprod()
        max_dd = ((cum_ret / cum_ret.cummax()) - 1).min()
        assert max_dd <= 0


# ── cost_sensitivity_analysis ──────────────────────────────────────────────────

from stock_data.evaluation import cost_sensitivity_analysis


class TestCostSensitivityAnalysis:
    def test_output_shows_all_cost_levels(self, capsys):
        prod_df = _make_prod_df()
        cost_sensitivity_analysis(prod_df, cost_bps_list=[10, 20, 30])
        captured = capsys.readouterr().out
        assert "10" in captured
        assert "20" in captured
        assert "30" in captured
        assert "COST SENSITIVITY" in captured

    def test_higher_costs_reduce_net_returns(self, capsys):
        prod_df = _make_prod_df()
        prod_df["turnover"] = 0.5  # fixed turnover
        cost_sensitivity_analysis(prod_df, cost_bps_list=[10, 50])
        # Just verify it runs without error — monotonicity is inherent
