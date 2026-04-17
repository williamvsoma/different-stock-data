"""Run the full pipeline: download data → build features → walk-forward backtest.

Saves intermediate artifacts to data/interim/, data/processed/, and models/
so that report notebooks can load results without re-downloading.

Usage:
    python main.py             # run all stages
    python main.py --stage data      # download & save interim data only
    python main.py --stage features  # build features from interim data
    python main.py --stage train     # train models from processed data
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_data.config import EARNINGS_LAG_DAYS, INITIAL_CAPITAL
from stock_data.dataset import (
    compute_forward_returns,
    compute_realized_vol,
    download_macro,
    download_prices,
    fetch_annual_income,
    fetch_quarterly_balance_sheets,
    fetch_quarterly_cashflows,
    fetch_quarterly_income,
    drop_sparse_pairs,
    filter_by_membership,
    get_active_symbols,
    load_sp500_universe,
    pivot_statements,
    reshape_annual_income,
    reshape_statements,
)
from stock_data.features import build_features
from stock_data.modeling.train import factor_benchmarks, walk_forward
from stock_data.evaluation import (
    summarize_walk_forward,
    evaluate_factors,
    assess_external_validity,
    simulate_portfolio,
    print_simulation_summary,
    run_iteration_analysis,
)
from stock_data.plots import plot_walk_forward_diagnostics, plot_simulation

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
MODELS = Path("models")
FIGURES = Path("reports/figures")


def stage_data():
    """Download raw data and save interim parquet files."""
    INTERIM.mkdir(parents=True, exist_ok=True)

    # ── 1. Load universe ──
    sp500 = load_sp500_universe("data/raw/sp500_monthly.csv")
    symbols = get_active_symbols(sp500)
    print(f"S&P 500 symbols (current + historical): {len(symbols)}")

    # ── 2. Fetch financial statements ──
    qi, _ = fetch_quarterly_income(symbols)
    combined = drop_sparse_pairs(reshape_statements(qi))
    features_raw = pivot_statements(combined)

    qbs, _ = fetch_quarterly_balance_sheets(symbols)
    bs_raw = pivot_statements(reshape_statements(qbs))

    qcf, _ = fetch_quarterly_cashflows(symbols)
    cf_raw = pivot_statements(reshape_statements(qcf))

    ai, _ = fetch_annual_income(symbols)
    annual_raw = reshape_annual_income(ai)

    # ── 3. Prices & macro ──
    inc_symbols = combined.index.get_level_values("symbol").unique().tolist()
    all_dates = combined.index.get_level_values("date")
    min_date = all_dates.min() - pd.Timedelta(days=400)
    max_date = all_dates.max() + pd.Timedelta(days=120)

    close_prices = download_prices(inc_symbols, min_date, max_date)
    macro_df = download_macro(min_date, max_date)

    # Save interim data
    features_raw.to_parquet(INTERIM / "features_raw.parquet")
    bs_raw.to_parquet(INTERIM / "bs_raw.parquet")
    cf_raw.to_parquet(INTERIM / "cf_raw.parquet")
    annual_raw.to_parquet(INTERIM / "annual_raw.parquet")
    close_prices.to_parquet(INTERIM / "close_prices.parquet")
    macro_df.to_parquet(INTERIM / "macro_df.parquet")
    sp500.to_parquet(INTERIM / "sp500_membership.parquet")
    print(f"  Interim data saved to {INTERIM}/")


def stage_features():
    """Build features from interim data and save processed parquet files."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load interim data
    features_raw = pd.read_parquet(INTERIM / "features_raw.parquet")
    bs_raw = pd.read_parquet(INTERIM / "bs_raw.parquet")
    cf_raw = pd.read_parquet(INTERIM / "cf_raw.parquet")
    annual_raw = pd.read_parquet(INTERIM / "annual_raw.parquet")
    close_prices = pd.read_parquet(INTERIM / "close_prices.parquet")
    macro_df = pd.read_parquet(INTERIM / "macro_df.parquet")
    sp500 = pd.read_parquet(INTERIM / "sp500_membership.parquet")

    # ── 4. Forward returns & vol ──
    sym_date_pairs = features_raw.index
    returns_df = compute_forward_returns(close_prices, sym_date_pairs)
    returns_df["return_quantile"] = returns_df.groupby("date")["next_q_return"].rank(pct=True)

    vol_df = compute_realized_vol(returns_df, close_prices)
    returns_full = returns_df.merge(vol_df, on=["symbol", "date"], how="inner")

    # ── 5. Build features ──
    risk_model_df, feature_cols = build_features(
        features_raw, bs_raw, cf_raw, annual_raw,
        close_prices, macro_df, returns_df, returns_full,
    )

    # ── 5b. Point-in-time universe filter (removes survivorship bias) ──
    before = len(risk_model_df)
    risk_model_df = filter_by_membership(risk_model_df, sp500)
    pct = 100 * len(risk_model_df) / before if before else 0
    print(f"  Point-in-time membership filter (survivorship bias removal): "
          f"{before:,} → {len(risk_model_df):,} rows ({pct:.1f}% retained)")

    # Save processed data
    risk_model_df.to_parquet(PROCESSED / "risk_model_df.parquet")
    with open(PROCESSED / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"  Processed data saved to {PROCESSED}/")


def stage_train():
    """Run walk-forward backtest, evaluate, and save model outputs."""
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Load processed data
    risk_model_df = pd.read_parquet(PROCESSED / "risk_model_df.parquet")
    with open(PROCESSED / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    close_prices = pd.read_parquet(INTERIM / "close_prices.parquet")

    # ── 6. Walk-forward backtest ──
    prod_df, prod_fi, weights_history = walk_forward(
        risk_model_df, feature_cols, close_prices,
    )
    summarize_walk_forward(prod_df, prod_fi, feature_cols)

    # ── 7. Factor benchmarks & evaluation ──
    factor_results = factor_benchmarks(risk_model_df, feature_cols, prod_df)
    ci_lo, ci_hi, boot_means, ex_n = evaluate_factors(prod_df, factor_results)

    # Save model outputs
    prod_df.to_parquet(MODELS / "prod_df.parquet")
    with open(MODELS / "prod_fi.pkl", "wb") as f:
        pickle.dump(prod_fi, f)
    with open(MODELS / "weights_history.pkl", "wb") as f:
        pickle.dump(weights_history, f)
    with open(MODELS / "factor_results.pkl", "wb") as f:
        pickle.dump(factor_results, f)
    print(f"  Model outputs saved to {MODELS}/")

    # ── 8. Diagnostics & reports ──
    fig = plot_walk_forward_diagnostics(
        prod_df, factor_results, boot_means, ci_lo, ci_hi, ex_n,
    )
    fig.savefig(FIGURES / "walk_forward_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    assess_external_validity(
        prod_df, prod_fi, weights_history,
        risk_model_df, feature_cols, close_prices,
    )

    sim_df, mkt_sim, qlog = simulate_portfolio(
        prod_df, weights_history, risk_model_df, close_prices, INITIAL_CAPITAL,
    )
    if len(sim_df) > 0 and len(mkt_sim) > 0:
        fig2 = plot_simulation(sim_df, mkt_sim, qlog, INITIAL_CAPITAL)
        fig2.savefig(FIGURES / "simulation.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print_simulation_summary(sim_df, mkt_sim, INITIAL_CAPITAL)

    run_iteration_analysis(
        prod_df, prod_fi, weights_history,
        risk_model_df, feature_cols, close_prices,
    )

    print("\nTraining complete. Artifacts saved to models/ and reports/figures/.")


def main():
    parser = argparse.ArgumentParser(description="Stock data pipeline")
    parser.add_argument(
        "--stage",
        choices=["data", "features", "train"],
        default=None,
        help="Run a single pipeline stage instead of the full pipeline",
    )
    args = parser.parse_args()

    if args.stage == "data":
        stage_data()
    elif args.stage == "features":
        stage_features()
    elif args.stage == "train":
        stage_train()
    else:
        stage_data()
        stage_features()
        stage_train()
        print("\nPipeline complete. Artifacts saved to data/ and models/.")


if __name__ == "__main__":
    main()
